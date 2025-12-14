import numpy as np 
from typing import Optional, Tuple
import torch
from solver_utils.consistency import consistency_check_with_depth


def compute_transformed_points(depth1: np.ndarray, transformation1: np.ndarray,
                                transformation2: np.ndarray, intrinsic1: np.ndarray,
                                intrinsic2: Optional[np.ndarray]):
    """
    Computes transformed position for each pixel location
    """
    h, w = depth1.shape
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    transformation = np.matmul(transformation2, np.linalg.inv(transformation1))

    y1d = np.array(range(h))
    x1d = np.array(range(w))
    x2d, y2d = np.meshgrid(x1d, y1d)
    ones_2d = np.ones(shape=(h, w))
    ones_4d = ones_2d[:, :, None, None]
    pos_vectors_homo = np.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]

    intrinsic1_inv = np.linalg.inv(intrinsic1)
    intrinsic1_inv_4d = intrinsic1_inv[None, None]
    intrinsic2_4d = intrinsic2[None, None]
    depth_4d = depth1[:, :, None, None]
    trans_4d = transformation[None, None]

    unnormalized_pos = np.matmul(intrinsic1_inv_4d, pos_vectors_homo)
    world_points = depth_4d * unnormalized_pos
    world_points_homo = np.concatenate([world_points, ones_4d], axis=2)
    trans_world_homo = np.matmul(trans_4d, world_points_homo)
    trans_world = trans_world_homo[:, :, :3]
    trans_norm_points = np.matmul(intrinsic2_4d, trans_world)
    return trans_norm_points,world_points



def bilinear_splatting(frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                        flow12: np.ndarray, flow12_mask: Optional[np.ndarray], is_image: bool = False) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Using inverse bilinear interpolation based splatting
    :param frame1: (h, w, c)
    :param mask1: (h, w): True if known and False if unknown. Optional
    :param depth1: (h, w)
    :param flow12: (h, w, 2)
    :param flow12_mask: (h, w): True if valid and False if invalid. Optional
    :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
    :return: warped_frame2: (h, w, c)
                mask2: (h, w): True if known and False if unknown
    """
    h, w, c = frame1.shape
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if flow12_mask is None:
        flow12_mask = np.ones(shape=(h, w), dtype=bool)
    grid = create_grid(h, w)
    trans_pos = flow12 + grid

    trans_pos_offset = trans_pos + 1
    trans_pos_floor = np.floor(trans_pos_offset).astype('int')
    trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
    trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_floor[:, :, 0] = np.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_floor[:, :, 1] = np.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_ceil[:, :, 0] = np.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_ceil[:, :, 1] = np.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

    prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

    sat_depth1 = np.clip(depth1, a_min=0, a_max=5000)
    log_depth1 = np.log(1 + sat_depth1)
    depth_weights = np.exp(log_depth1 / log_depth1.max() * 50)

    weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
    weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
    weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
    weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights

    weight_nw_3d = weight_nw[:, :, None]
    weight_sw_3d = weight_sw[:, :, None]
    weight_ne_3d = weight_ne[:, :, None]
    weight_se_3d = weight_se[:, :, None]

    warped_image = np.zeros(shape=(h + 2, w + 2, c), dtype=np.float64)
    warped_weights = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)

    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)

    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

    cropped_warped_image = warped_image[1:-1, 1:-1]
    cropped_weights = warped_weights[1:-1, 1:-1]

    mask = cropped_weights > 0 
    # mask2 = cropped_weights <=0.6 #????
    # mask = mask*mask2
    with np.errstate(invalid='ignore'):
        warped_frame2 = np.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)
        # warped_frame2 = np.where(mask[:, :, None], cropped_warped_image / (cropped_weights[:, :, None]+1e-20), 0)


    if is_image:
        try:
            assert np.min(warped_frame2) >= 0, np.min(warped_frame2)
            assert np.max(warped_frame2) <= 256
        except:
            import pdb; pdb.set_trace()
            
        clipped_image = np.clip(warped_frame2, a_min=0, a_max=255)
        warped_frame2 = np.round(clipped_image).astype('uint8')
    return warped_frame2, mask



def create_grid(h, w):
    x_1d = np.arange(0, w)[None]
    y_1d = np.arange(0, h)[:, None]
    x_2d = np.repeat(x_1d, repeats=h, axis=0)
    y_2d = np.repeat(y_1d, repeats=w, axis=1)
    grid = np.stack([x_2d, y_2d], axis=2)
    return grid

def forward_warp(frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                    transformation1: np.ndarray, transformation2: np.ndarray, intrinsic1: np.ndarray,
                    intrinsic2: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                np.ndarray]:
    """
    Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
    bilinear splatting.
    :param frame1: (h, w, 3) uint8 np array
    :param mask1: (h, w) bool np array. Wherever mask1 is False, those pixels are ignored while warping. Optional
    :param depth1: (h, w) float np array.
    :param transformation1: (4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
    :param transformation2: (4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
    :param intrinsic1: (3, 3) camera intrinsic matrix
    :param intrinsic2: (3, 3) camera intrinsic matrix. Optional
    """
    h, w = frame1.shape[:2]
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    assert frame1.shape == (h, w, 3)
    assert mask1.shape == (h, w)
    assert depth1.shape == (h, w)
    assert transformation1.shape == (4, 4)
    assert transformation2.shape == (4, 4)
    assert intrinsic1.shape == (3, 3)
    assert intrinsic2.shape == (3, 3)

    trans_points1,world_points = compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                    intrinsic2)
    
    trans_coordinates = trans_points1[:, :, :2, 0] / (trans_points1[:, :, 2:3, 0])
    
    trans_depth1 = trans_points1[:, :, 2, 0]

    grid = create_grid(h, w)
    flow12 = trans_coordinates - grid


    warped_frame2, mask2 = bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)

    return warped_frame2, mask2,flow12




def inverse_warp(img, depth, depth_pseudo, pose1, pose2, K, bg_mask=None, bandwidth=20):

    '''
    img: origin image of closest view, [3, H, W]
    depth: rendered depth of closest view, [1, H, W]
    depth_pseudo: rendered depth of pseudo view, ; [1, H, W]
    pose1: camera pose of closest view -- w2c? [4, 4]
    pose2: camera pose of pseudo view
    K: camera intrinsic matrix, [3, 3]
    '''

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    

    _, H, W = img.shape
    y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    x = x.float().to(img.device)
    y = y.float().to(img.device)
    z = depth_pseudo[0]
    
    x = (x - cx) / fx
    y = (y - cy) / fy
    coordinates = torch.stack([x, y, torch.ones_like(z)], dim=0)
    coordinates = coordinates * z
    
    coordinates = coordinates.view(3, -1)
    coordinates = torch.cat([coordinates, torch.ones_like(z).view(1, -1)], dim=0)
    pose = torch.matmul(pose1, torch.inverse(pose2))
    coordinates = torch.matmul(pose, coordinates)
    
    coordinates = coordinates[:3, :]
    coordinates = coordinates.view(3, H, W)
    x = fx * coordinates[0, :] / coordinates[2, :] + cx
    y = fy * coordinates[1, :] / coordinates[2, :] + cy

    grid = torch.stack([2.0*x/W - 1.0, 2.0*y/H - 1.0], dim=-1).unsqueeze(0).to('cuda')

    warped_img = torch.nn.functional.grid_sample(img.unsqueeze(0), grid, mode='nearest', padding_mode='zeros').squeeze(0).to('cuda') 
    warped_depth = torch.nn.functional.grid_sample(depth.unsqueeze(0), grid, mode='nearest', padding_mode='zeros').squeeze(0).to('cuda') 
    warped_bg_mask = None
    if not (bg_mask is None):
        warped_bg_mask = torch.nn.functional.grid_sample(bg_mask.unsqueeze(0).float(), grid, mode='nearest', padding_mode='zeros').squeeze(0).to('cuda') 
        warped_bg_mask = (warped_bg_mask > 0.5)
    mask_warp = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    mask_warp = mask_warp.to('cuda')
    
    warped_depth_clone = warped_depth.clone()
    warped_depth_max = warped_depth_clone.max()
    warped_depth_zero = (warped_depth_clone > 0)
    warped_depth[~warped_depth_zero] = 1e4
    warped_depth_min = warped_depth.min()

    norm_warped_depth = (warped_depth[0].detach().clone() - warped_depth_min) / (warped_depth_max - warped_depth_min)
    
    warped_depth[~warped_depth_zero] = 0
    norm_warped_depth[~warped_depth_zero[0]] = 0
    
    # norm_depth_pseudo = (depth_pseudo[0].detach() - depth_pseudo[0].min()) / (depth_pseudo[0].max() - depth_pseudo[0].min()) # this is wrong
    norm_depth_pseudo = (depth_pseudo[0].detach() - warped_depth_min) / (warped_depth_max - warped_depth_min)

    mask_depth = (torch.abs(norm_warped_depth - norm_depth_pseudo) < 0.3)
    mask_depth_strict = (torch.abs(norm_warped_depth - norm_depth_pseudo) < 0.1)
    
    mask_depth = mask_depth.to('cuda')
    mask_depth_strict = mask_depth_strict.to('cuda')
    mask = mask_warp & mask_depth
    mask = mask.to('cuda')

    
    reproj_error = consistency_check_with_depth(depth1=depth_pseudo.squeeze(0), pose1=pose2, intrinsics1=K, depth2=depth.squeeze(0), pose2=pose1, intrinsics2=K)
    mask_reproj = reproj_error<bandwidth # 20

    # soft_mask_reproj = torch.exp(-(reproj_error/10)**2) # p1
    # soft_mask_reproj = torch.exp(-(reproj_error/40)**2) # p2 
    # soft_mask_reproj = torch.exp(-(reproj_error/20)**3) # p3 
    soft_mask_reproj = torch.exp(-(reproj_error/bandwidth)**3) # p3 

    mask_reproj = mask_reproj * mask_warp 
    # mask_depth_strict = mask_reproj * mask_warp # for debugging
    
    warped_masked_img = warped_img * mask
    
    mask_inv = ~mask
    
    return {"warped_img": warped_img, "warped_depth": warped_depth, 
            "mask_warp": mask_warp, "mask_depth": mask_depth, "mask": mask, 
            "warped_masked_img": warped_masked_img, "mask_inv": mask_inv, "mask_depth_strict": mask_depth_strict, "warped_bg_mask": warped_bg_mask,
            "mask_reproj": mask_reproj, 
            "soft_mask_reproj": soft_mask_reproj, 
            } 