
# support FSGS 
# add view selection in view densification process 

import numpy as np 

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


import torch 
import torch.nn.functional as F
import cv2 
import os
import matplotlib.pyplot as plt
import PIL
import glob
import copy 
import trimesh 
import open3d as o3d 


from FSGS.utils.trainer import init_GSTrainer, GSTrainer
# from model.SVD_2pass import export_to_video
from solver_utils.forward_warp import forward_warp, inverse_warp

from FSGS.scene.cameras import Camera
# from model.correspondence import MASt3RInference
from functools import partial
from diffusers.utils import load_image, export_to_video





class DiffusionGS:
    def __init__(self, GSTrainer:GSTrainer, num_input_views=12, save_dir=None, diffusion_type='2Pass', interp_type='forward_warp', debug=False, input_args=None):
        
         
        self.args = input_args    
        self.cam_confidence = self.args.cam_confidence 
        self.pseudo_cam_sampling_rate = self.args.pseudo_cam_sampling_rate
        self.fps_keyframe_sampling = self.args.fps_keyframe_sampling

        self.debug = debug

        self.gsTrainer = GSTrainer

        self.dust3r = self.gsTrainer.dust3r
        

        self.num_input_views = num_input_views 
        self.save_dir = save_dir
        self.interp_type = interp_type
        assert self.interp_type in ['forward_warp', 'backward_warp']
        self.densify_type = self.args.densify_type  # 'interpolate' or 'from_single'

        self.refine_epoch = 0 

        if self.args.reorg_train_views:
            self.get_TrainCameras = partial(self.get_GS_TrainCameras, ordered=True)
        else:
            self.get_TrainCameras = partial(self.get_GS_TrainCameras, ordered=False)

        self.gs_intrinsics, self.gs_extrinsics = self.get_TrainCameras()[0].get_calib_matrix_nerf() # K, w2c 
        self.gs_extrinsics, self.gs_intrinsics = self.gs_extrinsics.cpu().numpy(), self.gs_intrinsics.cpu().numpy()

        self.gs_height, self.gs_width = self.get_TrainCameras()[0].image_height,  self.get_TrainCameras()[0].image_width

        self.diffusion_height, self.diffusion_width = 576, 1024
        self.diffusion_intrinsics  = self.gs_intrinsics.copy() *  np.array([self.diffusion_width/self.gs_width, self.diffusion_height/self.gs_height, 1]).reshape([3,1])
        print(f"The intrinsics of GS = {self.gs_intrinsics} \n the intrinsics of SVD={self.diffusion_intrinsics}")


        

        # assert diffusion_type in ['2Pass', '2Pass2Latent', '2Pass4Latent', '1Pass', '1PassProb', '2PassProb']

        self.diffusion_type = diffusion_type
        if self.densify_type in ['from_single', 'from_single_gs']:
            if diffusion_type != '1Pass':
                print("Warning: 2Pass is not supported for densify_type='from_single', we will use 1Pass instead")
            
            # diffusion_type = '1Pass'
            assert '1Pass' in diffusion_type 
            
        
        # if diffusion_type == '2Pass':
        #     from model.SVD_2pass import StableVideoDiffusionPipeline
        #     # raise NotImplementedError
        #     self.diffusion_pipeline = StableVideoDiffusionPipeline
        #     self.latent_num = 1 
        # elif diffusion_type == '2Pass2Latent':
        #     from model.SVD_2pass import StableVideoDiffusionPipeline
        #     self.diffusion_pipeline = StableVideoDiffusionPipeline
        #     self.latent_num = 2
        # elif diffusion_type == '2Pass4Latent':
        #     from model.SVD_2pass import StableVideoDiffusionPipeline
        #     self.diffusion_pipeline = StableVideoDiffusionPipeline
        #     self.latent_num = 4
        # elif diffusion_type == '1Pass':
        #     from model.SVD_1pass import StableVideoDiffusionPipeline
        #     self.diffusion_pipeline = StableVideoDiffusionPipeline
        #     self.latent_num = 1
        # elif diffusion_type == '1PassProb':
        #     from model.SVD_1pass_prob import StableVideoDiffusionPipeline
        #     self.diffusion_pipeline = StableVideoDiffusionPipeline
        #     self.latent_num = 1
        # elif diffusion_type == '2PassProb':
        #     from model.SVD_2pass_prob import StableVideoDiffusionPipeline
        #     self.diffusion_pipeline = StableVideoDiffusionPipeline
        #     self.latent_num = 1
        if diffusion_type == '2PassProbUncertain':
            from model.SVD_2pass_prob_uncertain import StableVideoDiffusionPipeline
            self.diffusion_pipeline = StableVideoDiffusionPipeline
            self.latent_num = 1
        elif diffusion_type == '2PassProbUncertainPost':
            from model.SVD_2pass_prob_uncertain_post import StableVideoDiffusionPipeline
            self.diffusion_pipeline = StableVideoDiffusionPipeline
            self.latent_num = 1
        else:
            raise NotImplementedError
    

    def get_GS_TrainCameras(self, ordered=False):
        '''
            Get the train cameras from the GS scene, if ordered is True, the cameras will be ordered with SalesMan 
        '''
        if ordered:
            return self.gsTrainer.scene.getTrainCameras(ordered=True)
        else:
            return self.gsTrainer.scene.getTrainCameras()

    def init_GS(self, cycle=0):
        gs_start_iter = 0
        epoch_iteration = self.gsTrainer.opt.iterations
        self.gsTrainer.training(gs_start_iter, epoch_indicator=cycle)
        pass
    
    def render_GS(self, idx=None, pose=None, return_alpha=False):
        '''
            idx: the index of the camera in the GS cameras
            pose: the pose of the camera in the GS cameras, world to camera 
        '''
        assert (idx is None and pose is not None) or (idx is not None and pose is None)

        if idx is not None:
            cam = self.get_TrainCameras()[idx]
            pose=cam.world_view_transform.transpose(0,1).cpu().numpy() # world to cam? transpose to make right multiplication->left multiplication
            image = cam.get_image()[0].permute([1,2,0]).cpu().numpy()*255 # HWC

            render_res = self.gsTrainer.render_view(cam)
            # depth = render_res['plane_depth'].detach().squeeze().cpu().numpy()
            depth = render_res['depth'].detach().squeeze().cpu().numpy()

        else:
            cam_template = self.get_TrainCameras()[0]    

            cam = Camera(colmap_id=-1, R=pose[:3,:3].T, T=pose[:3,3], FoVx=cam_template.FoVx, FoVy=cam_template.FoVy, image=cam_template.original_image, gt_alpha_mask=None,
                 image_name=None, uid=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda:0", 
                cam_confidence=1.0, )
            # cam = cam_template # for debugging

            render_res = self.gsTrainer.render_view(cam)

            image = render_res['render'].detach().squeeze().cpu().numpy()
            depth = render_res['depth'].detach().squeeze().cpu().numpy()

        if return_alpha:
            alpha = render_res['alpha'].detach().squeeze().cpu().numpy()
            return pose, image, depth, alpha

        return pose, image, depth



    def densify_views(self, cycle_num, down_sample_rate=1, densify_type='interpolate', num_views_for_pcd_densification=4):
        # densify views based on sparse views

        # assert densify_type in ['interpolate', 'from_single', 'from_single_gs', 'interpolate_gs']


        def view_selection_for_pcd_densification(poses, pose_num, alpha=1.0, beta=1.0):
            # poses: list of (4,4) np.ndarray, w2c
            # pose_num: int, number of poses to select
            # Do furthest pose samplling based on the pose distance metric 1-\text{CovisibilityScore} = 1-\exp(-\alpha ||\mathbf{t}_1-\mathbf{t}_2 ||) \exp(-\beta \arccos(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{||\mathbf{v}_1||~ ||\mathbf{v}_2|| }) ) 
            assert len(poses) > pose_num, "The number of poses should be larger than the number of poses to select"
            N = len(poses)
            # Convert w2c to c2w
            c2w_poses = [np.linalg.inv(p) for p in poses]
            translations = np.array([p[:3, 3] for p in c2w_poses])  # (N,3)
            forwards = np.array([p[:3, 2] for p in c2w_poses])      # (N,3)


            # Compute pairwise distance matrix
            dist_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    t_dist = np.linalg.norm(translations[i] - translations[j])
                    v1 = forwards[i]
                    v2 = forwards[j]
                    cos_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    covis = np.exp(-alpha * t_dist) * np.exp(-beta * angle)
                    dist_matrix[i, j] = 1 - covis

            # Furthest point sampling
            selected = [0]  # Start with the first pose
            for _ in range(1, pose_num):
                min_dists = np.min(dist_matrix[selected], axis=0)
                min_dists[selected] = -np.inf  # Exclude already selected
                next_idx = np.argmax(min_dists)
                selected.append(next_idx)
            # return [poses[i] for i in selected]
            return selected



        dense_views = [] 
        dense_poses = []
        key_frame_mask = []
        input_flags = []

        trimesh_scenes = trimesh.Scene()



        for i in range(self.num_input_views):
            saving_path = os.path.join(self.save_dir, 'dense_views' f'interpolated_dense_views_cyc{cycle_num}_view{i}.pt')
            if os.path.exists(saving_path):
                data = torch.load(saving_path)
                diffused_frames, interpolated_poses = data['views'], data['poses']
                if 'gs_renderings' in data:
                    gs_images = data['gs_renderings']
                print("Load interpolated dense views from ", saving_path)
            else:
                # if densify_type == 'interpolate':
                #     diffused_frames, interpolated_poses = self._interpolate_between(i, (i+1)%self.num_input_views, replace=True)
                # elif densify_type == 'from_single_gs':
                #     diffused_frames, interpolated_poses = self._extrapolate_from_gs(i, replace=True)
                #     pass
                if densify_type == 'interpolate_loop0_gs':
                    if i==self.num_input_views-1:
                        break
                    diffused_frames, interpolated_poses, gs_images = self._interpolate_between_gs_v3(i, (i+1)%self.num_input_views, replace=True, perturb_interp_poses=True )
                elif densify_type == 'interpolate_gs_v2':
                    # diffused_frames, interpolated_poses, gs_images = self._interpolate_between_gs_v2(i, (i+1)%self.num_input_views, replace=True)
                    diffused_frames, interpolated_poses, gs_images = self._interpolate_between_gs_v3(i, (i+1)%self.num_input_views, replace=True)
                # elif densify_type == 'interpolate_gs_unordered':
                #     # diffused_frames, interpolated_poses, gs_images = self._interpolate_between_gs_v2(i, (i+1)%self.num_input_views, replace=True)
                #     diffused_frames, interpolated_poses, gs_images = self._interpolate_between_gs_v3(i, (i+1)%self.num_input_views, replace=True)
                else:
                    raise NotImplementedError(f"{densify_type} not supported")

            
                if down_sample_rate <1:
                    # diffused_frames = [diffused_frames[i] for i in range(0, len(diffused_frames), down_sample_rate)]
                    # interpolated_poses = [interpolated_poses[i] for i in range(0, len(interpolated_poses), down_sample_rate)]
                    num_samples = int(len(diffused_frames) * down_sample_rate)
                    indices = np.linspace(0, len(diffused_frames) - 1, num_samples, dtype=int)
                    diffused_frames = [diffused_frames[i] for i in indices ]
                    interpolated_poses = [interpolated_poses[i] for i in indices ]

            

            input_flags.extend( [True] + [False] * (len(diffused_frames)-2) ) # the last frame is the input frame

            dense_views.extend(diffused_frames[:-1])
            dense_poses.extend(interpolated_poses[:-1])  
            # assert num_views_for_pcd_densification>1

            if self.fps_keyframe_sampling:
                key_inds = view_selection_for_pcd_densification(interpolated_poses, num_views_for_pcd_densification, alpha=1.0, beta=1.0)
                key_inds.sort()
            else:   

                key_inds = np.linspace(0, len(diffused_frames)-1, num_views_for_pcd_densification, dtype=int)

            key_inds = key_inds[:-1]
            template = np.zeros(len(diffused_frames)-1, dtype=bool)
            template[key_inds] = True
            key_frame_mask.extend(list(template)) # the first frame is the key frame

            if densify_type=='interpolate_loop0_gs' and i==self.num_input_views-2:
                input_flags.append(True) # the last frame is the input frame
                dense_views.append(diffused_frames[-1])
                dense_poses.append(interpolated_poses[-1])
                key_frame_mask.append(True) # the last input frame is the key frame



            assert len(dense_views) == len(dense_poses) == len(key_frame_mask) == len(input_flags)


            # TODO:save dense views
            # torch.save( {'views':diffused_frames, 'poses':interpolated_poses, 'gs_renderings': gs_images}, saving_path, )
            torch.save( {'views':diffused_frames, 'poses':interpolated_poses}, saving_path, )

        #densify point clouds
        if num_views_for_pcd_densification > 1:

            trimesh_scene = self.densify_pcds([ dense_views[i] for i in np.nonzero(key_frame_mask)[0] ], 
                                              [dense_poses[i] for i in np.nonzero(key_frame_mask)[0]], 
                                              key_frame_mask=[True]*len(np.nonzero(key_frame_mask)[0]), 
                                              input_flags=[input_flags[i] for i in np.nonzero(key_frame_mask)[0] ], 
                                              win_samples=-1)
            # trimesh_scenes.add_geometry( trimesh_scene )    

            os.makedirs(os.path.join(self.save_dir, 'dense_views'), exist_ok=True)
            dust3r_pcd = trimesh_scene.geometry['geometry_0']

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dust3r_pcd.vertices)
            pcd.colors = o3d.utility.Vector3dVector(dust3r_pcd.colors[:, :3]/255.0)   
        
            if 1:
                every_k_points = len(pcd.points)//100000
                down_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
                cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=3.0)
            else:
                every_k_points = len(pcd.points)//300000
                down_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
                # down_pcd = pcd.voxel_down_sample(voxel_size=(dust3r_pcd.vertices.max(axis=0)- dust3r_pcd.vertices.min(axis=0)).max()/500  )
                cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=1)

            # dense_pcds = inlier_cloud.voxel_down_sample(voxel_size=(dust3r_pcd.vertices.max(axis=0)- dust3r_pcd.vertices.min(axis=0)).max()/1000  )
            # uni_down_pcd = pcd

            inlier_cloud = down_pcd.select_by_index(ind)
            dense_pcds = inlier_cloud 

            o3d.io.write_point_cloud(os.path.join(self.save_dir, 'dense_views', f'dense_views_cyc{cycle_num}.ply'), dense_pcds)
        else:
            dense_pcds = None

 
     

        return dense_views, dense_poses, dense_pcds#, corresp_masks 



    def densify_pcds(self, diffused_frames, interpolated_poses, key_frame_mask=None, input_flags=None, win_samples=-1):
        '''
            interpolated_poses: the poses of the diffused frames, w2c 
        '''


        assert len(diffused_frames) == len(interpolated_poses)# == 25
        if key_frame_mask is not None:
            assert len(diffused_frames) == len(key_frame_mask)

        if isinstance(diffused_frames[0], torch.Tensor):
            diffused_frames = [im.permute([1,2,0]).cpu().numpy()*255 for im in diffused_frames]

        num_frames = len(diffused_frames)

        

        #filter out noisy frames
        filtered_diffused_frames = []
        filtered_interpolated_poses = []
        filtered_intrinsics_list = []
        filtered_key_frame_mask = []
        filtered_key_frame_inds = []
        
        # import pdb; pdb.set_trace()
        # for i in range(1, len(diffused_frames)-1):
        for i in range(0, len(diffused_frames)):
            pose, image, depth, alpha = self.render_GS(pose=interpolated_poses[i], return_alpha=True) 

            
            masks, flow_bwfw = self.gsTrainer.generate_corresp_mask(gs_renderings=[torch.from_numpy(image)], svd_outputs=[torch.from_numpy(diffused_frames[i].transpose([2,0,1]) ) ], dist_thresh=3, desc_only=False)
            mask = masks[0][0].data.cpu().numpy()

            if i ==0 or i==num_frames-1:
                # assert mask.mean() > 0.2, "Weird phenomenon, the first or last frame is not good, please check the input images"
                if mask.mean()<0.2:
                    print( "Warning: Weird phenomenon, the first or last frame is not good, please check the input images")

            if mask.mean()>0.3 or (input_flags is not None and input_flags[i]):  
                if key_frame_mask is not None:
                    if key_frame_mask[i]:
                        filtered_key_frame_inds.append(len(filtered_diffused_frames)) 
                    filtered_key_frame_mask.append(key_frame_mask[i]) 

                filtered_diffused_frames.append(diffused_frames[i])
                filtered_interpolated_poses.append(np.linalg.inv(interpolated_poses[i])) # w2c to c2w
                K = self.gs_intrinsics.copy()
                K[:2] = K[:2] * 512/self.gs_width
                filtered_intrinsics_list.append(K)

                    

        self.dust3r.to('cuda')
        # import pdb; pdb.set_trace()
        pairs_key_frames = self.dust3r.make_pairs([filtered_diffused_frames[i] for i in filtered_key_frame_inds], scene_graph='complete', global_image_inds=filtered_key_frame_inds) 
        '''
        
        pairs_window = []
        for i in range(len(filtered_key_frame_inds)):
            keyframe_idx = filtered_key_frame_inds[i]

            left_radius = (filtered_key_frame_inds[i]  - filtered_key_frame_inds[(i-1)%len(filtered_key_frame_inds) ] )//2
            right_radius = (filtered_key_frame_inds[ (i+1)%len(filtered_key_frame_inds) ] - filtered_key_frame_inds[i])//2

            left_neighbor_inds = list(range(filtered_key_frame_inds[i] - left_radius, filtered_key_frame_inds[i]) ) 
            right_neighbor_inds = list(range(filtered_key_frame_inds[i] + 1, filtered_key_frame_inds[i] + right_radius + 1) )

            left_right_inds = left_neighbor_inds + right_neighbor_inds
            if win_samples> 0:
                sample_inds = np.linspace(0, len(left_right_inds)-1, win_samples-1, dtype=int)
                left_right_inds = [left_right_inds[i] for i in sample_inds]


            window_inds =  [filtered_key_frame_inds[i]] + left_right_inds #+ left_neighbor_inds + right_neighbor_inds
            
            window = [ filtered_diffused_frames[j] for j in window_inds ]

            pairs = self.dust3r.make_pairs(window, scene_graph='oneref-0', global_image_inds=window_inds ) 
            pairs_window.extend(pairs)
        '''        




        scene, trimesh_scene = self.dust3r.run(filtered_diffused_frames, c2w_poses=filtered_interpolated_poses, intrinsics=filtered_intrinsics_list, preset_pairs=pairs_key_frames)
        self.dust3r.to('cpu')
         

        return trimesh_scene



        
        

    def generate_corresp_mask(self, gs_renderings, svd_outputs):
        mast3r = MASt3RInference()

        assert len(gs_renderings) == len(svd_outputs)

        masks = []
        for i in range(len(gs_renderings)):
            image_pair = [gs_renderings[i], svd_outputs[i]]
            matches_im0, matches_im1, images_resized = mast3r.run_inference(image_pair)

            mask =torch.zeros_like(images_resized[0]['img'][0,0])

            mask[matches_im0[:,1], matches_im0[:,0]] = 1

            masks.append(mask)
            
        return masks


    def _extrapolate_from_gs(self, idx1, replace=True,):

        pose1, image1, depth1 = self.render_GS(idx1)
        # pose2, image2, depth2 = self.render_GS(idx2)

        gs_height, gs_width, _ = image1.shape
        assert gs_height == self.gs_height and gs_width == self.gs_width

        
        train_cams = self.get_TrainCameras().copy()
        pseudo_cams = self.gsTrainer.scene.getPseudoCameras().copy()

        # import pdb; pdb.set_trace()
        if self.args.dataset == 'llff':
            self.gsTrainer.find_nearest_cam([train_cams[idx1]], pseudo_cams, multi_view_max_angle=30, multi_view_min_dis=0.01, multi_view_max_dis=1.5) # #TODO: hyperparameters need to be tuned
        elif self.args.dataset == 'dtu':
            self.gsTrainer.find_nearest_cam([train_cams[idx1]], pseudo_cams, multi_view_max_angle=15, multi_view_min_dis=0.01, multi_view_max_dis=0.5)
        else:
            raise NotImplementedError
        # self.gsTrainer.find_nearest_cam([train_cams[idx1]], pseudo_cams, multi_view_max_angle=30, multi_view_min_dis=0.01, multi_view_max_dis=0.5)

        sampled_id = np.linspace(0, len(train_cams[idx1].nearest_id )-1, 24, dtype=int)
        nearest_pseudo_cams = [pseudo_cams[i] for i in sampled_id ] # maybe sub-sampling is needed

        # interpolated_poses = self.pose_interpolation(pose1,pose2)
        interpolated_poses =  [cam.world_view_transform.transpose(0,1).cpu().numpy() for cam in nearest_pseudo_cams] # world to cam? transpose to make right multiplication->left multiplication -- 
        interpolated_poses = [pose1] + interpolated_poses

        # render images from GS
        # for pseudo_cam in nearest_pseudo_cams:
        pseudo_images = []
        for pseudo_pose in interpolated_poses:
            # render_res = self.gsTrainer.render_view(pseudo_cam)
            
            _, pseudo_image, pseudo_depth = self.render_GS(pose=pseudo_pose)
            pseudo_images.append( cv2.resize(pseudo_image.transpose([1,2,0]), dsize=(self.diffusion_width, self.diffusion_height), interpolation=cv2.INTER_LINEAR ) )
            # pseudo_cam.image = image
            # pseudo_cam.depth = depth




        save_path = os.path.join(self.save_dir, 'warp_images')
        os.makedirs(save_path, exist_ok=True)

        if self.interp_type == 'forward_warp':
            image_o, image_o2, masks, cond_image = self.warp_images(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=None, depth_l=depth1, depth_r=None, save_path=save_path, save_file_name_prefix=f'{idx1}') 
        elif self.interp_type == 'backward_warp':   
            image_o, image_o2, masks, cond_image, aux = self.warp_images_bw(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=None, depth_l=depth1, depth_r=None, save_path=save_path, save_file_name_prefix=f'{idx1}', return_ero_mask=True) 
            masks_ero = aux['masks_ero']
            
        else:
            raise NotImplementedError
        assert image_o2 is None # !

        dists, min_indice = self.compute_dists(interpolated_poses, type='single_end') # TODO: We may need to use cam to world transformation matrix

        sigma_list = np.load('sigmas/sigmas_100.npy').tolist()
        lambda_ts = self.search_hypers(interpolated_poses,dists,sigma_list,save_path, type='single_end') # single end

        output_path = os.path.join(self.save_dir, 'render_warp_images')
        os.makedirs(output_path, exist_ok=True)

        #replace cond_image with pseudo_images
        # import pdb; pdb.set_trace()  # check the shape and type of pseudo_images
        # image_o = pseudo_images[0]
        def get_intensity_confidence(warpped_image, pseudo_image, uncertainty_mask, sigma=0.1):
            warpped_image [uncertainty_mask.astype(bool)] == 0
            pseudo_image[uncertainty_mask.astype(bool)] == 0

            # intensity_conf = np.exp( -( (np.linalg.norm(warpped_image-pseudo_image, axis=-1, keepdims=True))/sigma)**2 ) * (1-uncertainty_mask)
            intensity_conf = np.exp( -( (np.linalg.norm(warpped_image-pseudo_image, axis=-1, keepdims=True))/sigma)**3 ) * (1-uncertainty_mask)

            return intensity_conf



        if 'Prob' in self.diffusion_type: # 1PassProb,...
            masks = aux['soft_masks_reproj']
            soft_masks_reproj_ori = aux['soft_masks_reproj_ori']    
            # cond_image = np.stack(pseudo_images[1:])
            
            #integrate the uncertainties based on intensity and geometry 
            # cond_image= np.stack(cond_image)
            cond_image = np.stack(aux['cond_images_ori'])
            gs_images= np.stack(pseudo_images[1:])
            intensity_conf = get_intensity_confidence(cond_image, gs_images, 1-(cond_image>0), sigma=0.5)

            geo_inten_conf = (intensity_conf) * (1-soft_masks_reproj_ori[...,None])
            geo_inten_uncertainty = 1-geo_inten_conf
            #reshape the mask for SVD
            mask_buf = []
            for mask_erosion_ in geo_inten_uncertainty:
                mask_erosion = np.mean(mask_erosion_,axis = -1)
                mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
                mask_erosion = np.mean(mask_erosion,axis=2)
                mask_buf.append(mask_erosion)
                # mask_erosion[mask_erosion < 0.2] = 0
                # mask_erosion[mask_erosion >= 0.2] = 1
            masks = np.stack(mask_buf)
            masks = torch.from_numpy(masks).float()




            # save for debugging
            for i, pseudo_image in enumerate(cond_image):
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_pseudo_image.png'), pseudo_image *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_soft_mask_reproj.png'), aux['soft_masks_reproj_ori'][i] *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_geo_inten_uncertainty.png'), geo_inten_uncertainty[i] *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_inten_uncertainty.png'), (1-intensity_conf[i]) *255)

        else:
            cond_image = np.stack(pseudo_images[1:]) * (1-masks_ero)
        # cond_image = np.clip(cond_image, 0, 1)
        cond_image = [np.clip(im, 0, 1) for im in cond_image]


        diffused_frames = self.svd_render(image_o, image_o2, masks, cond_image, output_path, lambda_ts, num_frames=25, save_prefix=f'{idx1}_',  ) #NOTE:!!!!
        # diffused_frames = svd_render(image_o2, image_o, masks.flip(dims=[0]), cond_image[::-1], output_path,lambda_ts.flip(dims=[1]), num_frames=num_frames) #NOTE:!!!!

        if replace:
            diffused_frames[0] = PIL.Image.fromarray( (image_o*255).astype(np.uint8) ) 
            # diffused_frames[-1] = PIL.Image.fromarray( (image_o2*255).astype(np.uint8) )  #image_o2


        diffused_frames = [fr.resize( (self.gs_width, self.gs_height) ) for fr in diffused_frames]  # resize to GS resolution

        diffused_frames = [torch.from_numpy(np.asarray(fr)).permute([2,0,1])/255. for fr in diffused_frames]   # HWC->CHW

        return diffused_frames, interpolated_poses
    def _extrapolate_from(self, idx1, replace=True,):

        pose1, image1, depth1 = self.render_GS(idx1)
        # pose2, image2, depth2 = self.render_GS(idx2)

        gs_height, gs_width, _ = image1.shape
        assert gs_height == self.gs_height and gs_width == self.gs_width

        
        train_cams = self.get_TrainCameras().copy()
        pseudo_cams = self.gsTrainer.scene.getPseudoCameras().copy()

        if self.args.dataset == 'llff':
            self.gsTrainer.find_nearest_cam([train_cams[idx1]], pseudo_cams, multi_view_max_angle=30, multi_view_min_dis=0.01, multi_view_max_dis=1.5) # #TODO: hyperparameters need to be tuned
        elif self.args.dataset == 'dtu':
            self.gsTrainer.find_nearest_cam([train_cams[idx1]], pseudo_cams, multi_view_max_angle=15, multi_view_min_dis=0.01, multi_view_max_dis=0.5)
        else:
            raise NotImplementedError

        sampled_id = np.linspace(0, len(train_cams[idx1].nearest_id )-1, 24, dtype=int)
        nearest_pseudo_cams = [pseudo_cams[i] for i in sampled_id ] # maybe sub-sampling is needed
        

        # interpolated_poses = self.pose_interpolation(pose1,pose2)
        interpolated_poses =  [cam.world_view_transform.transpose(0,1).cpu().numpy() for cam in nearest_pseudo_cams] # world to cam? transpose to make right multiplication->left multiplication -- 
        interpolated_poses = [pose1] + interpolated_poses

        dists, min_indice = self.compute_dists(interpolated_poses, type='single_end') # TODO: We may need to use cam to world transformation matrix

        save_path = os.path.join(self.save_dir, 'warp_images')
        os.makedirs(save_path, exist_ok=True)


        if self.interp_type == 'forward_warp':
            image_o, image_o2, masks, cond_image = self.warp_images(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=None, depth_l=depth1, depth_r=None, save_path=save_path, save_file_name_prefix=f'{idx1}') 
        elif self.interp_type == 'backward_warp':   
            image_o, image_o2, masks, cond_image = self.warp_images_bw(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=None, depth_l=depth1, depth_r=None, save_path=save_path, save_file_name_prefix=f'{idx1}') 
        else:
            raise NotImplementedError
        assert image_o2 is None


        sigma_list = np.load('sigmas/sigmas_100.npy').tolist()
        lambda_ts = self.search_hypers(interpolated_poses,dists,sigma_list,save_path, type='single_end') # single end

        output_path = os.path.join(self.save_dir, 'render_warp_images')
        os.makedirs(output_path, exist_ok=True)

        diffused_frames = self.svd_render(image_o, image_o2, masks, cond_image, output_path, lambda_ts, num_frames=25, save_prefix=f'{idx1}_',  ) #NOTE:!!!!

        if replace:
            diffused_frames[0] = PIL.Image.fromarray( (image_o*255).astype(np.uint8) ) 
            # diffused_frames[-1] = PIL.Image.fromarray( (image_o2*255).astype(np.uint8) )  #image_o2


        diffused_frames = [fr.resize( (self.gs_width, self.gs_height) ) for fr in diffused_frames]  # resize to GS resolution

        diffused_frames = [torch.from_numpy(np.asarray(fr)).permute([2,0,1])/255. for fr in diffused_frames]   # HWC->CHW

        return diffused_frames, interpolated_poses

    def _perturb_interp_pose_candidates(self, anchor_poses, perturb_num=5):
        '''
            perturb the anchor poses to generate more candidates for interpolation

            anchor_poses: (N, 4, 4) array, the poses to be perturbed
            perturb_num: int, the number of perturbed poses to be generated for each anchor pose
            return: a list of perturbed poses for each anchor pose, each element is a list of perturbed poses
            
            1. compute the nearest neighbor distance for each pose
            2. perturb each pose with a random translation and rotation noise
            3. return a list of perturbed poses for each anchor pose
        '''

        # def max_translation_difference(poses):
        def nearest_neighbor_dist(poses):
            # poses: (N, 4, 4) array
            translations = np.array([pose[:3, 3] for pose in poses])  # shape (N, 3)
            # Compute pairwise distances
            dists = np.linalg.norm(translations[:, None, :] - translations[None, :, :], axis=-1) # NxN
            max_dist = np.max(dists)
            #set the diagnal of dists to max_dist
            np.fill_diagonal(dists, max_dist) 
            nn_dist = np.min(dists, axis=1)  # min distance to other poses
            return nn_dist 

        def perturb_pose(pose, trans_std=0.01, rot_std_deg=1.0):
            """
            Perturb a 6-DoF camera pose (4x4 matrix) with random noise.
            Args:
                pose: np.ndarray, shape (4,4), camera pose matrix.
                trans_std: float, stddev for translation noise (meters).
                rot_std_deg: float, stddev for rotation noise (degrees).
            Returns:
                perturbed_pose: np.ndarray, shape (4,4)
            """
            perturbed_pose = pose.copy()
            # Perturb translation
            perturbed_pose[:3, 3] += np.random.normal(0, trans_std, size=3)
            # Perturb rotation
            rot_noise = R.from_euler('xyz', np.random.normal(0, rot_std_deg, size=3)*np.array([1,1,1]), degrees=True)
            orig_rot = R.from_matrix(pose[:3, :3])
            new_rot = rot_noise * orig_rot
            perturbed_pose[:3, :3] = new_rot.as_matrix()
            return perturbed_pose

        # max_t_diff = max_translation_difference(anchor_poses)
        nn_dists = nearest_neighbor_dist(anchor_poses)
        # num_anchor_poses = len(anchor_poses)
        perturbed_poses = []

        for i, pose in enumerate(anchor_poses):

            perturbed_poses_i = [anchor_poses[i].astype(np.float32) ]  # include the original pose
            for _ in range(perturb_num):
                perturbed_pose_i = perturb_pose(pose.copy(), trans_std=nn_dists[i]*0.1, rot_std_deg=0.1) # perturb the pose with 10% of the nearest neighbor distance
                # perturbed_pose_i = perturb_pose(pose.copy(), trans_std=nn_dists[i]*0.1, rot_std_deg=0) # perturb the pose with 10% of the nearest neighbor distance
                perturbed_poses_i.append(perturbed_pose_i.astype(np.float32) )


            perturbed_poses.append(perturbed_poses_i)  # convert to float32 for consistency)
        
        return perturbed_poses

    def _perturb_and_select_interp_poses(self, anchor_poses, ref_poses, K, perturb_num=5, num_select=1): 

        def get_nn_ref_idx(input_pose, ref_poses):
            """
            Get the index of the nearest reference pose to the input pose.
            """
            input_translation = input_pose[:3, 3] # (3,)
            ref_translations = np.array([pose[:3, 3] for pose in ref_poses]) # shape (N, 3)
            dists = np.linalg.norm(ref_translations - input_translation, axis=1) # (N,)
            nn_ref_idx = np.argmin(dists)
            return nn_ref_idx


        # render images from GS for reference poses
        ref_images = []
        ref_depths = []
        for ref_pose in ref_poses:
            _, image, depth = self.render_GS(pose=ref_pose)
            ref_images.append(image)
            ref_depths.append(depth)

        

        perturbed_poses = self._perturb_interp_pose_candidates(anchor_poses, perturb_num, )

        selected_poses = []
        for perturbed_pose_group in perturbed_poses:
            # perturbed_pose_group: list of perturbed poses for each anchor pose
            # compute the distance between each perturbed pose and the reference poses
            uncertainties = []
            for i, perturbed_pose in enumerate(perturbed_pose_group):
                nn_ref_idx = get_nn_ref_idx(perturbed_pose, ref_poses)
                # import pdb; pdb.set_trace() # check the correctness of nn_ref_idx
                # print(nn_ref_idx, " is the nearest reference pose index for perturbed pose ", i)
                ref_img = torch.from_numpy(ref_images[nn_ref_idx]).cuda()
                ref_depth = torch.from_numpy(ref_depths[nn_ref_idx][None]).cuda() #(1, H, W)
                ref_pose = torch.from_numpy(ref_poses[nn_ref_idx]).cuda()

                _, img_pseudo, depth_pseudo = self.render_GS(pose=perturbed_pose)
                depth_pseudo = torch.from_numpy(depth_pseudo[None]).cuda() # (1, H, W)

                warp_dict = inverse_warp(ref_img, ref_depth, depth_pseudo, ref_pose, torch.from_numpy(perturbed_pose).cuda(), K, bg_mask=None, bandwidth=20)
                uncertainty_mask = 1-warp_dict['soft_mask_reproj']
                uncertainties.append(uncertainty_mask.mean().cpu().numpy()) # compute the uncertainty as the mean of the uncertainty mask

            sel_idx = np.argmax(uncertainties)
            # print(sel_idx, " is the selected perturbed pose index with highest uncertainty for anchor pose group")
            selected_poses.append(perturbed_pose_group[sel_idx])


        return selected_poses







    def _interpolate_between_gs_v3(self, idx1, idx2, replace=True, perturb_interp_poses=True):
        # set lamda_ts according to uncertainty masks 
        # replace the cond_image with pseudo_image if the uncertainty is high
        # perturb the interpolated poses for better interpolation views

        pose1, image1, depth1 = self.render_GS(idx1)
        pose2, image2, depth2 = self.render_GS(idx2)

        gs_height, gs_width, _ = image1.shape
        assert gs_height == self.gs_height and gs_width == self.gs_width

        interpolated_poses = self.pose_interpolation(pose1,pose2)

        if perturb_interp_poses:
            interpolated_poses = self._perturb_and_select_interp_poses(interpolated_poses, [pose1, pose2], K=torch.tensor(self.gs_intrinsics,dtype=torch.float32, device='cuda'), perturb_num=5, num_select=1,) # perturb the interpolated poses to generate more candidates for interpolation
            interpolated_poses = [pose1] + interpolated_poses[1:-1] + [pose2]  # add the original poses to the list



        dists, min_indice = self.compute_dists(interpolated_poses)
        

        # render images from GS
        pseudo_images = []
        pseudo_depths = []
        # pseudo_images_gs = []
        for pseudo_pose in interpolated_poses:
            
            _, pseudo_image, pseudo_depth = self.render_GS(pose=pseudo_pose)
            # pseudo_images_gs.append(pseudo_image.transpose[1,2,0])
            pseudo_images.append( cv2.resize(pseudo_image.transpose([1,2,0]), dsize=(self.diffusion_width, self.diffusion_height), interpolation=cv2.INTER_LINEAR ) )
            pseudo_depths.append( cv2.resize(pseudo_depth, dsize=(self.diffusion_width, self.diffusion_height), interpolation=cv2.INTER_LINEAR ) )



        save_path = os.path.join(self.save_dir, 'warp_images')
        os.makedirs(save_path, exist_ok=True)

        if self.interp_type == 'forward_warp':
            image_o, image_o2, masks, cond_image = self.warp_images(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=image2, depth_l=depth1, depth_r=depth2, save_path=save_path, save_file_name_prefix=f'{idx1}') 
        elif self.interp_type == 'backward_warp':   
            image_o, image_o2, masks, cond_image, aux = self.warp_images_bw(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=image2, depth_l=depth1, depth_r=depth2, save_path=save_path, save_file_name_prefix=f'{idx1}', return_ero_mask=True) 
            masks_ero = aux['masks_ero']
            nearby_consistency_uncertainty, nearby_inten_consistency_uncertainty = self.consistency_check_from_nearby_images_bw(self.diffusion_intrinsics, interpolated_poses, images=pseudo_images, depths=pseudo_depths,  save_path=save_path, return_ero_mask=True) 
        else:
            raise NotImplementedError

        def get_intensity_confidence(warpped_image, pseudo_image, uncertainty_mask, sigma=0.1):
            # warpped_image [uncertainty_mask.astype(bool)] == 0
            # pseudo_image[uncertainty_mask.astype(bool)] == 0

            # intensity_conf = np.exp( -( (np.linalg.norm(warpped_image-pseudo_image, axis=-1, keepdims=True))/sigma)**2 ) * (1-uncertainty_mask)
            intensity_conf = np.exp( -( (np.linalg.norm(warpped_image-pseudo_image, axis=-1, keepdims=True))/sigma)**3 ) * (1-uncertainty_mask)

            return intensity_conf
        # ''' #15
        if 'Prob' in self.diffusion_type: # 1PassProb, 2PassProb... 
            masks = aux['soft_masks_reproj']
            soft_masks_reproj_ori = aux['soft_masks_reproj_ori']    
            # cond_image = np.stack(pseudo_images[1:])
            
            #integrate the uncertainties based on intensity and geometry 
            # cond_image= np.stack(cond_image)
            # cond_image = np.stack(aux['cond_images_ori'])
            cond_image_ori = np.stack(aux['cond_images_ori'])


            gs_images= np.stack(pseudo_images[1:-1])
            intensity_conf = get_intensity_confidence(cond_image_ori, gs_images, 1-(cond_image_ori.sum(axis=-1, keepdims=True)>0), sigma=0.5)
            # intensity_conf = get_intensity_confidence(cond_image, gs_images, 1-(cond_image>0), sigma=0.5)

            geo_inten_conf = (intensity_conf) * (1-soft_masks_reproj_ori[...,None])
            # geo_inten_conf = (1-soft_masks_reproj_ori[...,None]) # 24.2
            geo_inten_uncertainty = 1-geo_inten_conf

            #reshape the mask for SVD
            mask_buf = []
            for mask_erosion_ in geo_inten_uncertainty:
                mask_erosion = np.mean(mask_erosion_,axis = -1)
                mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
                mask_erosion = np.mean(mask_erosion,axis=2)
                mask_buf.append(mask_erosion)
            masks = np.stack(mask_buf)
            masks = torch.from_numpy(masks).float()


            # cond_image = np.stack(aux['cond_images_ori'])

            cond_image = np.where(geo_inten_uncertainty>0.5, gs_images, np.stack(aux['cond_images_ori']) ) # maybe we can use optical flow to generate the target views by align gs image and input images 
            

            # The below is for debugging
            nearby_uncertainty = 1 - (1-torch.stack(nearby_consistency_uncertainty[1:-1]))* (1-torch.stack(nearby_inten_consistency_uncertainty[1:-1]))
            geo_inten_uncertainty = np.where(geo_inten_uncertainty.mean(axis=(-1,-2, -3), keepdims=True)>0.6, nearby_uncertainty[..., None].cpu().numpy(), geo_inten_uncertainty)


            # save for debugging
            for i, pseudo_image in enumerate(cond_image):
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_pseudo_image.png'), pseudo_images[i][...,::-1] *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_cond_image_ori.png'), aux['cond_images_ori'][i][...,::-1] *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_soft_mask_reproj.png'), aux['soft_masks_reproj_ori'][i] *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_geo_inten_uncertainty.png'), geo_inten_uncertainty[i] *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_inten_uncertainty.png'), (1-intensity_conf[i]) *255)

                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_nearby_consistency_uncertainty.png'), (nearby_consistency_uncertainty[i].cpu().numpy() ) *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_nearby_inten_consistency_uncertainty.png'), (nearby_inten_consistency_uncertainty[i].cpu().numpy() ) *255)
                # import pdb; pdb.set_trace()
                try:
                    cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_cond.png'), cond_image[i][...,::-1] *255)
                except:
                    pass    

        else:
            cond_image = np.stack(pseudo_images[1:-1]) * (1-masks_ero)
        # ''' 
        cond_image = [np.clip(im, 0, 1) for im in cond_image]





        lambda_ts = self.search_hypers_v2(masks, save_path, type='double_end') 

        output_path = os.path.join(self.save_dir, 'render_warp_images')
        os.makedirs(output_path, exist_ok=True)

        #for saving gpu memory
        self.gsTrainer.gaussians.to(device='cpu')
        torch.cuda.empty_cache()

        diffused_frames = self.svd_render(image_o, image_o2, masks, cond_image, output_path, lambda_ts, num_frames=25, save_prefix=f'{idx1}_') #NOTE:!!!!

        #for saving gpu memory
        self.gsTrainer.gaussians.to(device='cuda')


        if replace:
            diffused_frames[0] = PIL.Image.fromarray( (image_o*255).astype(np.uint8) ) 
            diffused_frames[-1] = PIL.Image.fromarray( (image_o2*255).astype(np.uint8) )  #image_o2


        diffused_frames = [fr.resize( (self.gs_width, self.gs_height) ) for fr in diffused_frames]  # resize to GS resolution

        diffused_frames = [torch.from_numpy(np.asarray(fr)).permute([2,0,1])/255. for fr in diffused_frames]   # HWC->CHW

        # if replace:
        #     diffused_frames[0] = torch.from_numpy(np.asarray(image_o)).permute([2,0,1]).cuda()  #HWC-CHW
        #     diffused_frames[-1] = torch.from_numpy(np.asarray(image_o2)).permute([2,0,1]).cuda() 

        return diffused_frames, interpolated_poses, pseudo_images # gs rendered images

    def _interpolate_between_gs(self, idx1, idx2, replace=True,):

        pose1, image1, depth1 = self.render_GS(idx1)
        pose2, image2, depth2 = self.render_GS(idx2)

        gs_height, gs_width, _ = image1.shape
        assert gs_height == self.gs_height and gs_width == self.gs_width

        interpolated_poses = self.pose_interpolation(pose1,pose2)

        dists, min_indice = self.compute_dists(interpolated_poses)
        

        # render images from GS
        # for pseudo_cam in nearest_pseudo_cams:
        pseudo_images = []
        for pseudo_pose in interpolated_poses:
            # render_res = self.gsTrainer.render_view(pseudo_cam)
            
            _, pseudo_image, pseudo_depth = self.render_GS(pose=pseudo_pose)
            pseudo_images.append( cv2.resize(pseudo_image.transpose([1,2,0]), dsize=(self.diffusion_width, self.diffusion_height), interpolation=cv2.INTER_LINEAR ) )


        save_path = os.path.join(self.save_dir, 'warp_images')
        os.makedirs(save_path, exist_ok=True)

        if self.interp_type == 'forward_warp':
            image_o, image_o2, masks, cond_image = self.warp_images(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=image2, depth_l=depth1, depth_r=depth2, save_path=save_path, save_file_name_prefix=f'{idx1}') 
        elif self.interp_type == 'backward_warp':   
            image_o, image_o2, masks, cond_image, aux = self.warp_images_bw(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=image2, depth_l=depth1, depth_r=depth2, save_path=save_path, save_file_name_prefix=f'{idx1}', return_ero_mask=True) 
            masks_ero = aux['masks_ero']
        else:
            raise NotImplementedError

        def get_intensity_confidence(warpped_image, pseudo_image, uncertainty_mask, sigma=0.1):
            # import pdb; pdb.set_trace()
            warpped_image [uncertainty_mask.astype(bool)] == 0
            pseudo_image[uncertainty_mask.astype(bool)] == 0

            # intensity_conf = np.exp( -( (np.linalg.norm(warpped_image-pseudo_image, axis=-1, keepdims=True))/sigma)**2 ) * (1-uncertainty_mask)
            intensity_conf = np.exp( -( (np.linalg.norm(warpped_image-pseudo_image, axis=-1, keepdims=True))/sigma)**3 ) * (1-uncertainty_mask)

            return intensity_conf
        # ''' #15
        if 'Prob' in self.diffusion_type: # 1PassProb, 2PassProb... 
            masks = aux['soft_masks_reproj']
            soft_masks_reproj_ori = aux['soft_masks_reproj_ori']    
            # cond_image = np.stack(pseudo_images[1:])
            
            #integrate the uncertainties based on intensity and geometry 
            cond_image_ori = np.stack(aux['cond_images_ori'])


            gs_images= np.stack(pseudo_images[1:-1])
            intensity_conf = get_intensity_confidence(cond_image_ori, gs_images, 1-(cond_image_ori>0), sigma=0.5)

            geo_inten_conf = (intensity_conf) * (1-soft_masks_reproj_ori[...,None])
            # geo_inten_conf = (1-soft_masks_reproj_ori[...,None]) # 24.2
            geo_inten_uncertainty = 1-geo_inten_conf

            #reshape the mask for SVD
            mask_buf = []
            for mask_erosion_ in geo_inten_uncertainty:
                mask_erosion = np.mean(mask_erosion_,axis = -1)
                mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
                mask_erosion = np.mean(mask_erosion,axis=2)
                mask_buf.append(mask_erosion)
                # mask_erosion[mask_erosion < 0.2] = 0
                # mask_erosion[mask_erosion >= 0.2] = 1
            masks = np.stack(mask_buf)
            masks = torch.from_numpy(masks).float()

            cond_image = np.where(geo_inten_uncertainty>0.5, gs_images, np.stack(aux['cond_images_ori']) ) # maybe we can use optical flow to generate the target views by align gs image and input images 


            # save for debugging
            for i, pseudo_image in enumerate(cond_image):
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_pseudo_image.png'), pseudo_image *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_soft_mask_reproj.png'), aux['soft_masks_reproj_ori'][i] *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_geo_inten_uncertainty.png'), geo_inten_uncertainty[i] *255)
                cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_inten_uncertainty.png'), (1-intensity_conf[i]) *255)
                try:
                    cv2.imwrite(os.path.join(save_path, f'{idx1}_{i:04d}_cond.png'), cond_image *255)
                except:
                    pass    

        else:
            cond_image = np.stack(pseudo_images[1:-1]) * (1-masks_ero)
        # ''' 
        cond_image = [np.clip(im, 0, 1) for im in cond_image]





        sigma_list = np.load('sigmas/sigmas_100.npy').tolist()
        lambda_ts = self.search_hypers(interpolated_poses,dists,sigma_list,save_path)

        output_path = os.path.join(self.save_dir, 'render_warp_images')
        os.makedirs(output_path, exist_ok=True)

        diffused_frames = self.svd_render(image_o, image_o2, masks, cond_image, output_path, lambda_ts, num_frames=25, save_prefix=f'{idx1}_') #NOTE:!!!!
        # diffused_frames = svd_render(image_o2, image_o, masks.flip(dims=[0]), cond_image[::-1], output_path,lambda_ts.flip(dims=[1]), num_frames=num_frames) #NOTE:!!!!

        if replace:
            diffused_frames[0] = PIL.Image.fromarray( (image_o*255).astype(np.uint8) ) 
            diffused_frames[-1] = PIL.Image.fromarray( (image_o2*255).astype(np.uint8) )  #image_o2


        diffused_frames = [fr.resize( (self.gs_width, self.gs_height) ) for fr in diffused_frames]  # resize to GS resolution

        diffused_frames = [torch.from_numpy(np.asarray(fr)).permute([2,0,1])/255. for fr in diffused_frames]   # HWC->CHW


        return diffused_frames, interpolated_poses

    def _interpolate_between(self, idx1, idx2, replace=True,):


        pose1, image1, depth1 = self.render_GS(idx1)
        pose2, image2, depth2 = self.render_GS(idx2)

        gs_height, gs_width, _ = image1.shape
        assert gs_height == self.gs_height and gs_width == self.gs_width

        interpolated_poses = self.pose_interpolation(pose1,pose2)

        dists, min_indice = self.compute_dists(interpolated_poses)

        save_path = os.path.join(self.save_dir, 'warp_images')
        os.makedirs(save_path, exist_ok=True)


        if self.interp_type == 'forward_warp':
            image_o, image_o2, masks, cond_image = self.warp_images(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=image2, depth_l=depth1, depth_r=depth2, save_path=save_path, save_file_name_prefix=f'{idx1}') 
        elif self.interp_type == 'backward_warp':   
            image_o, image_o2, masks, cond_image = self.warp_images_bw(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=image2, depth_l=depth1, depth_r=depth2, save_path=save_path, save_file_name_prefix=f'{idx1}') 
        else:
            raise NotImplementedError


        sigma_list = np.load('sigmas/sigmas_100.npy').tolist()
        lambda_ts = self.search_hypers(interpolated_poses,dists,sigma_list,save_path)

        output_path = os.path.join(self.save_dir, 'render_warp_images')
        os.makedirs(output_path, exist_ok=True)

        diffused_frames = self.svd_render(image_o, image_o2, masks, cond_image, output_path, lambda_ts, num_frames=25, save_prefix=f'{idx1}_') #NOTE:!!!!
        # diffused_frames = svd_render(image_o2, image_o, masks.flip(dims=[0]), cond_image[::-1], output_path,lambda_ts.flip(dims=[1]), num_frames=num_frames) #NOTE:!!!!

        if replace:
            diffused_frames[0] = PIL.Image.fromarray( (image_o*255).astype(np.uint8) ) 
            diffused_frames[-1] = PIL.Image.fromarray( (image_o2*255).astype(np.uint8) )  #image_o2


        diffused_frames = [fr.resize( (self.gs_width, self.gs_height) ) for fr in diffused_frames]  # resize to GS resolution

        diffused_frames = [torch.from_numpy(np.asarray(fr)).permute([2,0,1])/255. for fr in diffused_frames]   # HWC->CHW

    

        return diffused_frames, interpolated_poses
    
    def svd_render(self, image_l, image_r, masks, cond_image,output_path,lambda_ts, num_frames=25, save_prefix=''):
        pipe = self.diffusion_pipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")


        if self.densify_type in ['from_single', 'from_single_gs']:
            assert isinstance(cond_image, list)
            frames = pipe([image_l],temp_cond=cond_image,mask = masks,lambda_ts=lambda_ts,num_frames=num_frames, decode_chunk_size=8, num_inference_steps=100, weight_clamp=0.4).frames[0] 

        elif self.densify_type in ['interpolate', 'interpolate_gs', 'interpolate_loop0_gs', 'interpolate_gs_v2']:
            assert isinstance(cond_image, list)

            frames = pipe([image_l],temp_cond=cond_image+[image_r],mask = masks,lambda_ts=lambda_ts,num_frames=num_frames, decode_chunk_size=8, num_inference_steps=100, latent_num=self.latent_num).frames[0] 

   
        # output_path = save_path+'_out_dgs'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i,fr in enumerate(frames):
            fr.save(os.path.join(output_path, f"{save_prefix}{i:06d}.png"))

        export_to_video(frames, os.path.join(output_path,"generated.mp4"), fps=7)

        pipe.to("cpu")


        return frames




    
    def search_hypers_v2(self,  masks, save_path, type='double_end', diffusion_steps=100):
        '''
            masks: uncertainty masks
        '''
        assert type in ['double_end', 'single_end']
        def tau_func(uncertain, k=70, b=30):
            step_thresh = k * uncertain + b
            return step_thresh
        def quad_tau_func(uncertain, a=-0.22/1.4, b=2.4*0.22/1.4, c=0.2):
            step_thresh = (a*uncertain**2 + b*uncertain + c)*100
            return step_thresh

        # sigmas = sigmas[:-1]
        # sigmas_max = max(sigmas)

        zero_count_default = 0

        if type == 'double_end':
            index_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

            masks = torch.mean(masks, dim=(-1,-2))
            assert masks.shape[0] == 23
            # masks = torch.clamp(masks/( torch.maximum( masks.max(), torch.full_like(masks.max(), 0.6)) ), 0, 1)
            masks = torch.clamp(masks/( torch.maximum( masks.max(), torch.full_like(masks.max(), 0.5)) ), 0, 1)
            # masks = torch.clamp(masks/( torch.minimum( masks.max(), torch.full_like(masks.max(), 0.4)) ), 0, 1)
            masks = torch.cat([torch.zeros_like(masks[0:1]), masks, torch.zeros_like(masks[:1])], dim=0 ) # padding zeros for 

        elif type == 'single_end':
            index_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] # Note: generate 24 frames
            masks = torch.mean(masks, dim=(-1,-2))
            assert masks.shape[0] == 24
            # masks = torch.clamp(masks/( torch.maximum( masks.max(), torch.full_like(masks.max(), 0.6)) ), 0, 1)
            masks = torch.clamp(masks/( torch.maximum( masks.max(), torch.full_like(masks.max(), 0.5)) ), 0, 1)
            masks = torch.cat([torch.zeros_like(masks[0:1]), masks], dim=0 )



        
        for v1 in range(1):
            for v2 in range(1):
                for v3 in range(1):
                    flag = True
                    lambda_t_list = []
                
                    # for timestamp, sigma in enumerate(sigmas):
                    for timestamp in range(diffusion_steps):
                        
                        temp_cond_indices = [0]
                        for tau in range(25):
                            if tau not in index_list:
                                lambda_t_list.append(1)
                            else:
                                tau_p = 0 
                                # tau_ = tau/24
                                tau_ = masks[tau]
                            
                                # if len(sigmas)-timestamp>tau_func(tau_*0.3, k=80, b=20):

                                # if len(sigmas)-timestamp>quad_tau_func(tau_):
                                if diffusion_steps-timestamp>quad_tau_func(tau_):
                                    lambda_t_list.append(1.0)
                                else:
                                    lambda_t_list.append(0.0)

       
                    if flag == True:
                        zero_count = sum(1 for x in lambda_t_list if x > 0.5) # The more rely on the projected image, the better
                        if zero_count > zero_count_default:
                            zero_count_default = zero_count
                            v_optimized = [v1,v2,v3]
                            lambda_t_list_optimized = lambda_t_list
                    

        lambda_t_list_optimized = np.array(lambda_t_list_optimized)
        lambda_t_list_optimized = lambda_t_list_optimized.reshape([diffusion_steps,25])

        lambda_t_list_optimized = torch.tensor(lambda_t_list_optimized)
        Z= lambda_t_list_optimized

        z_upsampled = F.interpolate(Z.unsqueeze(0).unsqueeze(0), scale_factor=10, mode='bilinear', align_corners=True)

        save_path = os.path.join(save_path,'lambad_'+str(v_optimized[0])+'_'+str(v_optimized[1])+'_'+str(v_optimized[2])+'.png')
        image_numpy = z_upsampled[0].permute(1, 2, 0).numpy()

        
        return lambda_t_list_optimized


    def pose_interpolation(self, pose_start, pose_end):
        '''
            pose_start: np.array of shape (4,4)
            pose_end: np.array of shape (4,4)

        '''
        # A = pose_start[:3, 3].reshape((1,3))
        # C = pose_end[:3, 3].reshape((1,3)) 

        # #####################################
        times = np.array([0, 24])
        interp_times = np.linspace(0, 24, 25) 

        rotations = R.from_matrix([pose[:3, :3] for pose in [pose_start, pose_end]])

        slerp = Slerp(times, rotations)
        interpolated_rotations = slerp(interp_times)

        translations = np.vstack([pose[:3, 3] for pose in [pose_start, pose_end]])
        trans_spline = CubicSpline(times, translations,  bc_type='natural')
        interpolated_translations = trans_spline(interp_times)

        interpolated_poses = np.array([
            np.vstack([np.hstack([interpolated_rotations[i].as_matrix(), interpolated_translations[i][:, np.newaxis]]), [0, 0, 0, 1]])
            for i in range(len(interp_times))
        ]).astype(np.float32)
        return interpolated_poses
    

    def compute_dists(self, interpolated_poses, type='double_end'):
        assert type in ['double_end', 'single_end']

        points =[]

        for pp in interpolated_poses:
            points.append(pp[:3,3].reshape(1,3))
        points = np.array(points)


        def total_distance(index):
            index = int(index)
            if index <= 0 or index >= len(points) - 1:
                return float('inf') 
            
            left_points = points[:index]
            right_points = points[index:]

            left_most = left_points[0]
            right_most = right_points[-1]

            left_distances = np.sum(np.linalg.norm(left_points - left_most, axis=1))
            right_distances = np.sum(np.linalg.norm(right_points - right_most, axis=1))

            return left_distances + right_distances

        result = minimize_scalar(total_distance, bounds=(1, len(points)-2), method='bounded')

        min_indice = int(result.x)
        
        if type == 'single_end': # only consider the distance to the start frame
            min_indice = len(interpolated_poses) + 10000 

        print("Best index:", min_indice)

        dists = []
        pt1 = interpolated_poses[0]
        pt2 = interpolated_poses[-1]
        diff = []
        ii = 0
        for pp in interpolated_poses:
            if ii<min_indice:
                R1, t1 = pp[:3, :3], pp[:3, 3]
                R2, t2 = pt1[:3, :3], pt1[:3, 3]

                translation_diff = t1 - t2
                diff.append(translation_diff)
            else:
                R1, t1 = pp[:3, :3], pp[:3, 3]
                R2, t2 = pt2[:3, :3], pt2[:3, 3]

                translation_diff = t1 - t2
                diff.append(translation_diff)
            ii+=1
        diff = np.array(diff)

        dists = np.linalg.norm(diff, axis=1)
        dists = dists/np.max(dists) # [0-1] normalization

        return dists,min_indice

    # masks, cond_image, aux = self.consistency_check_from_nearby_images_bw(self.diffusion_intrinsics, interpolated_poses, images=pseudo_images, depths=pseudo_depths,  save_path=save_path, return_ero_mask=True) 

    def consistency_check_from_nearby_images_bw(self, intrinsics, interpolated_poses, images, depths, save_path=None, return_ero_mask=False):

        '''
            warp images in a backward way
            interpolated_poses: a list of 4x4 matrices, w2c ?
            image_l: np.array of shape (H,W,3), range [0,255]
            image_r: np.array of shape (H,W,3), range [0,255]
            depth_l: np.array of shape (H,W)
            depth_r: np.array of shape (H,W)
        '''



        # new_width, new_height = 1024,576  # New dimensions
        new_width, new_height = self.diffusion_width, self.diffusion_height #1024,576  # New dimensions
        assert new_width == 1024 and new_height == 576

        uncertainty_masks = []
        warped_images = []
        intensity_uncertainty_masks = []
        window_radius = 1
        for cur_i in range(len(interpolated_poses)):
            
            masks = []
            warps = []
            for ref_i in range(cur_i-window_radius, cur_i+window_radius+1):
                if ref_i == cur_i or ref_i < 0 or ref_i >= len(interpolated_poses):
                    continue

                pose_s = interpolated_poses[ref_i]
                pose_t = interpolated_poses[cur_i]

                depth_s = depths[ref_i]
                depth_t = depths[cur_i]

                image_s = images[ref_i]
                image_t = images[cur_i]


                warp_dict = inverse_warp(
                                    torch.tensor(image_s, device='cuda', dtype=torch.float32).permute([2,0,1]),  
                                    torch.tensor(depth_s[None], device='cuda'), 
                                    depth_pseudo=torch.tensor(depth_t[None], device='cuda'), 
                                    pose1=torch.tensor(pose_s, device='cuda', dtype=torch.float32), pose2=torch.tensor(pose_t, device='cuda', dtype=torch.float32), K=torch.tensor(intrinsics, device='cuda', dtype=torch.float32), 
                                    bandwidth=10
                                    )
                mask = warp_dict['soft_mask_reproj']
                warp = warp_dict['warped_img']
                masks.append(mask)
                warps.append(warp)

            conf_mask = torch.stack(masks).mean(dim=0) 
            uncertainty_masks.append(1-conf_mask)

            warped_image = torch.stack(warps).mean(dim=0)
            intensity_conf = torch.exp( -( (torch.norm(warped_image-torch.tensor(image_t, device='cuda', dtype=torch.float32).permute([2,0,1]), dim=0) )/0.1)**3 ) 
            warped_images.append(warped_image)
            intensity_uncertainty_masks.append(1-intensity_conf)

        

        return uncertainty_masks, intensity_uncertainty_masks 




    # def warp_images_bw(self, intrinsics, interpolated_poses, image_l, image_r, depth_l, depth_r, save_path, save_file_name_prefix=''):
    def warp_images_bw(self, intrinsics, interpolated_poses, image_l, image_r=None, depth_l=None, depth_r=None, save_path=None, save_file_name_prefix='', return_ero_mask=False):

        '''
            warp images in a backward way
            interpolated_poses: list of 4x4 matrices, w2c ?
            image_l: np.array of shape (H,W,3), range [0,255]
            image_r: np.array of shape (H,W,3), range [0,255]
            depth_l: np.array of shape (H,W)
            depth_r: np.array of shape (H,W)
        '''



        # new_width, new_height = 1024,576  # New dimensions
        new_width, new_height = self.diffusion_width, self.diffusion_height #1024,576  # New dimensions
        assert new_width == 1024 and new_height == 576


        pose_s_l = interpolated_poses[0]

        image_l = cv2.resize(image_l, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        image_o = image_l
        # image_l = np.array(image_l)


        depth_l = cv2.resize(depth_l, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        interp_num = len(interpolated_poses)-1
        
        assert interp_num == 24

        if image_r is not None:
            interp_num = interp_num-1
            pose_s_r = interpolated_poses[-1]
            # image_r = PIL.Image.open(os.path.join(img_path,'000002.jpg'))
            # image_r = image_r.resize((1024,576),PIL.Image.Resampling.NEAREST) 
            image_r = cv2.resize(image_r, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            image_o2 = image_r
            # image_r = np.array(image_r)
            depth_r = cv2.resize(depth_r, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        cond_image = []
        cond_images_ori = []
        masks = []
        masks_ero = []
        soft_masks_reproj = []
        soft_masks_reproj_ori = []
        # for i in range(23):
        for i in range(interp_num):
            if depth_r is not None:
                if i<12:
                    depth = depth_l
                    pose_s = pose_s_l
                    image = image_l
                else:
                    depth = depth_r
                    pose_s = pose_s_r
                    image = image_r
            else:
                depth = depth_l
                pose_s = pose_s_l
                image = image_l


            # pose_t = pose_s # for debugging
            pose_t = interpolated_poses[i+1]
            _, rendered_t, rendered_depth_t = self.render_GS(pose=pose_t)

            rendered_depth_t = cv2.resize(rendered_depth_t, (new_width, new_height), interpolation=cv2.INTER_NEAREST) # resize to the same size as the input images
            # import pdb; pdb.set_trace()

            warp_dict = inverse_warp(torch.tensor(image, device='cuda', dtype=torch.float32).permute([2,0,1]),  
                                    torch.tensor(depth[None], device='cuda'), 
                                    depth_pseudo=torch.tensor(rendered_depth_t[None], device='cuda'), 
                                    pose1=torch.tensor(pose_s, device='cuda', dtype=torch.float32), pose2=torch.tensor(pose_t, device='cuda', dtype=torch.float32), K=torch.tensor(intrinsics, device='cuda', dtype=torch.float32), )
            # mask2 = warp_dict['mask'].cpu().numpy()
            # mask2 = warp_dict['mask_depth_strict'].cpu().numpy()
            mask2 = warp_dict['mask_reproj'].cpu().numpy()
            warped_frame2 = warp_dict['warped_img'].cpu().numpy().transpose([1,2,0])
            
            PIL.Image.fromarray(np.uint8( rendered_depth_t)).save(os.path.join(save_path,save_file_name_prefix+'_'+str(i).zfill(4)+"_depth.png"))
            # PIL.Image.fromarray(np.uint8( warp_dict['mask_reproj'] )).save(os.path.join(save_path,save_file_name_prefix+'_'+str(i).zfill(4)+"_depth_s.png"))


            # '''
            mask = 1-mask2
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = np.repeat(mask[:,:,np.newaxis]*255.,repeats=3,axis=2)
    
            kernel = np.ones((5,5), np.uint8)
            mask_erosion = cv2.dilate(np.array(mask), kernel, iterations = 1)
            mask_erosion = PIL.Image.fromarray(np.uint8(mask_erosion))
            mask_erosion.save(os.path.join(save_path,save_file_name_prefix+'_'+str(i).zfill(4)+"_mask.png"))

            mask_erosion_ = np.array(mask_erosion)/255.
            mask_erosion_[mask_erosion_ < 0.5] = 0
            mask_erosion_[mask_erosion_ >= 0.5] = 1
            masks_ero.append(mask_erosion_)

            # warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2))
            cond_images_ori.append(warped_frame2/255.)
            warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2*(1-mask_erosion_)))

            warped_frame2.save(os.path.join(save_path,save_file_name_prefix+'_'+str(i).zfill(4)+".png"))
            # '''

            # cond_image.append(warped_frame2)
            cond_image.append(np.asarray(warped_frame2, dtype=np.float32) / 255.) #!!!!

            mask_erosion = np.mean(mask_erosion_,axis = -1)
            mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
            mask_erosion = np.mean(mask_erosion,axis=2)
            mask_erosion[mask_erosion < 0.2] = 0
            mask_erosion[mask_erosion >= 0.2] = 1
            masks.append(torch.from_numpy(mask_erosion).unsqueeze(0))

            #process soft_mask_reproj
            # soft_mask_reproj = np.mean(warp_dict['soft_mask_reproj'].cpu().numpy() ,axis = -1)
            soft_masks_reproj_ori.append(1-warp_dict['soft_mask_reproj'].cpu().numpy())
            soft_mask_reproj = 1-warp_dict['soft_mask_reproj'].cpu().numpy() # confidence-> uncertainty
            soft_mask_reproj = soft_mask_reproj.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
            soft_mask_reproj = np.mean(soft_mask_reproj, axis=2)

            soft_masks_reproj.append(torch.from_numpy(soft_mask_reproj).unsqueeze(0) )

            i+=1
        masks = torch.cat(masks)
        # import pdb; pdb.set_trace()
        soft_masks_reproj = torch.cat(soft_masks_reproj).to(dtype=torch.float32)
        masks_ero = np.stack(masks_ero).astype(np.uint8)
        soft_masks_reproj_ori = np.stack(soft_masks_reproj_ori).astype(np.float32)

        if return_ero_mask:
            if image_r is None:
                return image_o/255., None, masks, cond_image, {"masks_ero": masks_ero, "soft_masks_reproj": soft_masks_reproj, "soft_masks_reproj_ori":soft_masks_reproj_ori, "cond_images_ori": cond_images_ori} #masks_ero
            else:
                return image_o/255., image_o2/255., masks, cond_image, {"masks_ero": masks_ero, "soft_masks_reproj": soft_masks_reproj, "soft_masks_reproj_ori": soft_masks_reproj_ori, "cond_images_ori":cond_images_ori } #masks_ero    
        else:

            if image_r is None:
                return image_o/255., None, masks, cond_image    
            else:
                return image_o/255., image_o2/255., masks, cond_image

    def warp_images(self, intrinsics, interpolated_poses, image_l, image_r=None, depth_l=None, depth_r=None, save_path='', save_file_name_prefix=''):
        '''
            Warp images in a forward way
            interpolated_poses: list of 4x4 matrices, w2c ?
            image_l: np.array of shape (H,W,3), range [0,255]
            image_r: np.array of shape (H,W,3), range [0,255]
            depth_l: np.array of shape (H,W)
            depth_r: np.array of shape (H,W)
        '''
    

        # new_width, new_height = 1024,576  # New dimensions
        new_width, new_height = self.diffusion_width, self.diffusion_height #1024,576  # New dimensions
        assert new_width == 1024 and new_height == 576


        pose_s_l = interpolated_poses[0]

        # image_l = cv2.resize(image_l, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        image_l = cv2.resize(image_l, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        image_o = image_l
        # image_l = np.array(image_l)


        depth_l = cv2.resize(depth_l, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        pose_s_r = interpolated_poses[-1]
        # image_r = PIL.Image.open(os.path.join(img_path,'000002.jpg'))
        # image_r = image_r.resize((1024,576),PIL.Image.Resampling.NEAREST) 
        # image_r = cv2.resize(image_r, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        interp_num = len(interpolated_poses)-1
        assert interp_num == 24
        if image_r is not None:
            interp_num = interp_num-1  
            image_r = cv2.resize(image_r, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image_o2 = image_r
            # image_r = np.array(image_r)
            depth_r = cv2.resize(depth_r, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        cond_image = []
        masks = []
        # for i in range(23):
        for i in range(interp_num):
            if image_r is not None:
                if i<12:
                    depth = depth_l
                    pose_s = pose_s_l
                    image = image_l
                else:
                    depth = depth_r
                    pose_s = pose_s_r
                    image = image_r
            else:
                depth = depth_l
                pose_s = pose_s_l
                image = image_l

            pose_t = interpolated_poses[i+1]
            
            warped_frame2, mask2,flow12= forward_warp(image, None, depth, pose_s, pose_t, intrinsics, None)
            mask = 1-mask2
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = np.repeat(mask[:,:,np.newaxis]*255.,repeats=3,axis=2)
    
            kernel = np.ones((5,5), np.uint8)
            mask_erosion = cv2.dilate(np.array(mask), kernel, iterations = 1)
            mask_erosion = PIL.Image.fromarray(np.uint8(mask_erosion))
            mask_erosion.save(os.path.join(save_path,save_file_name_prefix+'_'+str(i).zfill(4)+"_mask.png"))

            mask_erosion_ = np.array(mask_erosion)/255.
            mask_erosion_[mask_erosion_ < 0.5] = 0
            mask_erosion_[mask_erosion_ >= 0.5] = 1
            warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2))
            warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2*(1-mask_erosion_)))
            warped_frame2.save(os.path.join(save_path,save_file_name_prefix+'_'+str(i).zfill(4)+".png"))

            # cond_image.append(warped_frame2)
            cond_image.append(np.asarray(warped_frame2, dtype=np.float32) / 255.) #!!!!

            mask_erosion = np.mean(mask_erosion_,axis = -1)
            mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
            mask_erosion = np.mean(mask_erosion,axis=2)
            mask_erosion[mask_erosion < 0.2] = 0
            mask_erosion[mask_erosion >= 0.2] = 1
            masks.append(torch.from_numpy(mask_erosion).unsqueeze(0))

            i+=1
        masks = torch.cat(masks)

        if image_r is None:
            return image_o/255., None, masks, cond_image    
        else:
            return image_o/255., image_o2/255., masks, cond_image


    

    def refine_GS(self, dense_views, dense_poses, intrinsics, cam_confidence=0.01, gs_start_iter=0, disable_densification=False, load_iteration=None, pseudo_cam_sampling_rate=1, load_ckpt=True):
        # refine GS with dense views
        refine_ckpts = glob.glob(f'{self.gsTrainer.scene.model_path}/refine_*_chkpnt*.pth') 
        # '''
        if load_ckpt:
            # import pdb; pdb.set_trace()
            if len(refine_ckpts) > 0:
                refine_ckpts = sorted(refine_ckpts, key=lambda x: int(os.path.basename(x).split('_')[1]) )
                print(f"Load the latest refined GS: {refine_ckpts[-1]}...")
                self.gsTrainer.load_checkpoint(checkpoint=refine_ckpts[-1]) # load the latest refined GS
            else:
                checkpoint_path = os.path.join(self.gsTrainer.scene.model_path, "chkpnt" + str(self.gsTrainer.checkpoint_iterations[-1]) + ".pth")
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = os.path.join(self.gsTrainer.scene.model_path, "chkpnt_latest.pth" )
                self.gsTrainer.load_checkpoint(checkpoint=checkpoint_path ) # load the last checkpoint from initial GS 
                print(f"Load the initilized GS: {checkpoint_path}...")
        # '''

        self.original_GS_train_cameras_bak = copy.deepcopy(self.gsTrainer.scene.train_cameras)

        # self.gsTrainer.update_cameras(dense_views, dense_poses, intrinsics, cam_confidences=0.3, append=True, ) # append the interpolated views to GS cameras 
        self.gsTrainer.update_cameras(dense_views, dense_poses, intrinsics, cam_confidences=cam_confidence, append=True, load_iteration=load_iteration) # append the interpolated views to GS cameras 
        # self.gsTrainer.reset_scene()

        self.gsTrainer.reset_optimizers() # reset the optimizers for the new GS cameras ????
        self.gsTrainer.reset_gs() ###for debug 

        # epoch_iteration = self.gsTrainer.opt.iterations

        # self.gsTrainer.training(gs_start_iter)
        self.gsTrainer.finetune(0, self.refine_epoch, disable_densification=disable_densification, pseudo_cam_sampling_rate=pseudo_cam_sampling_rate) # finetune the GS with the new cameras)
        self.gsTrainer.scene.train_cameras = self.original_GS_train_cameras_bak # restore the original GS cameras

        self.refine_epoch +=1
        # gs_start_iter = self.gsTrainer.opt.iterations + 1 
        # self.gsTrainer.opt.iterations = gs_start_iter+epoch_iteration # set the end iteration for next epoch
    
    def consistency_check(self, idx1, idx2):
        from solver_utils.consistency import consistency_check_with_depth

        pose1, image1, depth1 = self.render_GS(idx1)
        pose2, image2, depth2 = self.render_GS(idx2)
        
        pose1 = torch.from_numpy(pose1)
        pose2 = torch.from_numpy(pose2)
        depth1 = torch.from_numpy(depth1)
        depth2 = torch.from_numpy(depth2)
        intrinsics = torch.tensor(self.gs_intrinsics)

        reproj_error = consistency_check_with_depth(depth1=depth1, pose1=pose1, intrinsics1=intrinsics, depth2=depth2, pose2=pose2, intrinsics2=intrinsics)
        cv2.imwrite(f'{self.save_dir}/reproj_error_{idx1}_{idx2}.png', reproj_error.cpu().numpy()/100*255)


        return reproj_error




    def run(self, refine_cycles=1):
        # '''
        self.init_GS() # skip for now

        # self.refine_GS(dense_views=[], dense_poses=[], intrinsics=self.gs_intrinsics, confidence=None) # for debug
        for i in range(refine_cycles):
            
            down_sample_rate = 1 
            # if self.num_input_views > 20:
            #     down_sample_rate = 300/self.num_input_views*25 # 300 is the max number of interpolated views 

            dense_views, dense_poses, dense_pcds = self.densify_views(cycle_num=i,down_sample_rate=down_sample_rate, densify_type=self.densify_type, num_views_for_pcd_densification=self.args.num_views_for_pcd_densification)

            # self.generate_corresp_mask(gs_images, dense_views )
            # self.gsTrainer.reset_gaussians_from_pcd(dense_pcds)
            if dense_pcds is not None:
                if i==0:
                    self.gsTrainer.reset_gaussians_from_pcd(dense_pcds, append_to_old_gaussians=False)
                else:
                    self.gsTrainer.reset_gaussians_from_pcd(dense_pcds, append_to_old_gaussians=True) # reset the GS with the dense pcds, append to the old gaussians


            self.gsTrainer.opt.use_lpips_loss = True

            if i == 0:
                self.refine_GS(dense_views=dense_views, dense_poses=dense_poses, intrinsics=self.gs_intrinsics, cam_confidence=self.cam_confidence, load_iteration=None, pseudo_cam_sampling_rate=self.pseudo_cam_sampling_rate, load_ckpt=False) # not use previsous gaussians, use the densified pcds
            else:
                self.refine_GS(dense_views=dense_views, dense_poses=dense_poses, intrinsics=self.gs_intrinsics, cam_confidence=self.cam_confidence, load_iteration=None, pseudo_cam_sampling_rate=self.pseudo_cam_sampling_rate, load_ckpt=True) # use previsous gaussians

            self.gsTrainer.opt.use_lpips_loss = False


