
import numpy as np 
import torch 


def get_points_from_depth(depth, intrinsics):
    '''
        Get 3D points from depth map
        Args:
            depth: depth map of shape (h, w)
            intrinsics: camera intrinsics of shape (3, 3)
        Returns:
            pts: 3D points of shape (h, w, 3)
    '''
    h, w = depth.shape

    # Create grid
    grid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    grid = torch.stack([*grid, torch.ones_like(grid[0]) ], axis=-1).to(intrinsics) # hxwx3

    pts = torch.einsum('bc,ijc->ijb', torch.inverse(intrinsics), grid ) * depth[:, :, None]

    return pts

def transform_points(pts, pose1, pose2):
    '''
        Transform points from frame1 to frame2
        Args:
            pts: 3D points of shape (h, w, 3)
            pose1: w2c pose of frame1 of shape (4, 4)
            pose2: w2c pose of frame2 of shape (4, 4)
        Returns:
            pts2: 3D points of shape (h, w, 3) in frame2
    '''
    h, w, _ = pts.shape

    pts = torch.cat([pts, torch.ones((h, w, 1), device=pts.device )], axis=-1) # hxwx4

    pts2 = torch.einsum('mn,ijn->ijm', pose2@torch.inverse(pose1), pts) # 4x4 x hxwx4 -> hxwx4
    pts2 = pts2[:, :, :3] / pts2[:, :, 3:] # normalize

    return pts2

def consistency_check_with_depth(depth1, pose1, intrinsics1, depth2, pose2, intrinsics2):
    '''
        Check the consistency of depth maps with poses and intrinsics
        Args:
            depth1: depth map of frame1 of shape (h, w)
            pose1: w2c pose of frame1
            intrinsics1: intrinsics of frame1
            depth2: depth map of frame2 of shape (h, w) 
            pose2: w2c pose of frame2
            intrinsics2: intrinsics of frame2
    '''

    h,w = depth1.shape

    pts = get_points_from_depth(depth1, intrinsics1)

    # Transform points from frame1 to frame2
    pts2 = transform_points(pts, pose1, pose2)
    
    # pts2_1 = pts2/pts2[:, :, 2:]* depth2 # normalize
    # pts2_1 = transform_points(pts2_1, pose2, pose1)

    # # Get depth map from transformed points
    image_pts2 = torch.einsum('mn,ijn->ijm', intrinsics2, pts2) # 3x3 x hxwx3 -> hxwx3
    image_pts2 = image_pts2[:, :, :2] / image_pts2[:, :, 2:]

    # Normalize image points forg grid sampling
    image_pts2[..., 0] /= ((w - 1) / 2)
    image_pts2[..., 1] /= ((h - 1) / 2)
    image_pts2 -= 1 # Normalize to [-1, 1]

    # Get depth map from transformed points
    depth1_2 = torch.nn.functional.grid_sample(depth2[None,None], image_pts2[None], ) # 1xCxHxW
    depth1_2 = depth1_2.permute(0, 2, 3, 1).squeeze(0) # HxWxC

    pts2_ = pts2/pts2[:, :, 2:]* depth1_2 # HxWx3
    pts2_1 = transform_points(pts2_, pose2, pose1)

    image_pts1_ = torch.einsum('mn,ijn->ijm', intrinsics1, pts2_1) # 3x3 x hxwx3 -> hxwx3
    image_pts1_ = image_pts1_[:, :, :2] / image_pts1_[:, :, 2:] # HxWx2

    # Create grid
    grid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy'  )
    grid = torch.stack(grid , axis=-1).to(image_pts1_ ) # hxwx2

    reprojection_error = torch.linalg.norm(image_pts1_ - grid, axis=-1)

    return reprojection_error












