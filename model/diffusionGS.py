
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


from PGSR.utils.trainer import init_GSTrainer, GSTrainer
from model.SVD_2pass2latent import export_to_video
from solver_utils.forward_warp import forward_warp




class DiffusionGS:
    def __init__(self, GSTrainer:GSTrainer, num_input_views=12, save_dir=None, diffusion_type='2Pass', debug=False):
        
        

        self.debug = debug

        self.gsTrainer = GSTrainer
        self.num_input_views = num_input_views 
        self.save_dir = save_dir

        self.refine_epoch = 0 
        
        self.gs_intrinsics, self.gs_extrinsics = self.gsTrainer.scene.getTrainCameras()[0].get_calib_matrix_nerf() # K, w2c 
        self.gs_extrinsics, self.gs_intrinsics = self.gs_extrinsics.cpu().numpy(), self.gs_intrinsics.cpu().numpy()

        self.gs_height, self.gs_width = self.gsTrainer.scene.getTrainCameras()[0].image_height,  self.gsTrainer.scene.getTrainCameras()[0].image_width

        self.diffusion_height, self.diffusion_width = 576, 1024
        self.diffusion_intrinsics  = self.gs_intrinsics.copy() *  np.array([self.diffusion_width/self.gs_width, self.diffusion_height/self.gs_height, 1]).reshape([3,1])
        print(f"The intrinsics of GS = {self.gs_intrinsics} \n the intrinsics of SVD={self.diffusion_intrinsics}")
        # import pdb; pdb.set_trace()

        

        assert diffusion_type in ['2Pass', '2Pass2Latent', '2Pass4Latent']

        if diffusion_type == '2Pass':
            from model.SVD_2pass import StableVideoDiffusionPipeline
            # raise NotImplementedError
            self.diffusion_pipeline = StableVideoDiffusionPipeline
            self.latent_num = 1 
        elif diffusion_type == '2Pass2Latent':
            from model.SVD_2pass import StableVideoDiffusionPipeline
            self.diffusion_pipeline = StableVideoDiffusionPipeline
            self.latent_num = 2
        elif diffusion_type == '2Pass4Latent':
            from model.SVD_2pass import StableVideoDiffusionPipeline
            self.diffusion_pipeline = StableVideoDiffusionPipeline
            self.latent_num = 4
        else:
            raise NotImplementedError


    def init_GS(self, cycle=0):
        gs_start_iter = 0
        epoch_iteration = self.gsTrainer.opt.iterations
        self.gsTrainer.training(gs_start_iter, epoch_indicator=cycle)
        pass
    
    def render_GS(self, idx):
        cam = self.gsTrainer.scene.getTrainCameras()[idx]
        pose=cam.world_view_transform.transpose(0,1).cpu().numpy() # world to cam? transpose to make right multiplication->left multiplication
        image = cam.get_image()[0].permute([1,2,0]).cpu().numpy()*255 # HWC

        render_res = self.gsTrainer.render_view(cam)
        depth = render_res['plane_depth'].detach().squeeze().cpu().numpy()
        return pose, image, depth


    def densify_views(self, cycle_num, down_sample_rate=1):
        # densify views based on sparse views

        dense_views = [] 
        dense_poses = []

        for i in range(self.num_input_views):
            saving_path = os.path.join(self.save_dir, 'dense_views' f'interpolated_dense_views_cyc{cycle_num}_view{i}.pt')
            if os.path.exists(saving_path):
                data = torch.load(saving_path)
                diffused_frames, interpolated_poses = data['views'], data['poses']
                print("Load interpolated dense views from ", saving_path)
            else:
                diffused_frames, interpolated_poses = self._interpolate_between(i, (i+1)%self.num_input_views, replace=True)
            
                # if down_sample_rate > 1:
                if down_sample_rate <1:
                    # diffused_frames = [diffused_frames[i] for i in range(0, len(diffused_frames), down_sample_rate)]
                    # interpolated_poses = [interpolated_poses[i] for i in range(0, len(interpolated_poses), down_sample_rate)]
                    num_samples = int(len(diffused_frames) * down_sample_rate)
                    indices = np.linspace(0, len(diffused_frames) - 1, num_samples, dtype=int)
                    diffused_frames = [diffused_frames[i] for i in indices ]
                    interpolated_poses = [interpolated_poses[i] for i in indices ]

            dense_views.extend(diffused_frames)
            dense_poses.extend(interpolated_poses)  

            if self.debug:
                break # only for debug 

            # TODO:save dense views
            os.makedirs(os.path.join(self.save_dir, 'dense_views'), exist_ok=True)
            torch.save( {'views':diffused_frames, 'poses':interpolated_poses}, saving_path, )
        
        # if down_sample_rate > 1:
        #     dense_views = [dense_views[i] for i in range(0, len(dense_views), down_sample_rate)]
        #     dense_poses = [dense_poses[i] for i in range(0, len(dense_poses), down_sample_rate)]
        

        return dense_views, dense_poses



    def _interpolate_between(self, idx1, idx2, replace=True):
        pose1, image1, depth1 = self.render_GS(idx1)
        pose2, image2, depth2 = self.render_GS(idx2)

        gs_height, gs_width, _ = image1.shape
        assert gs_height == self.gs_height and gs_width == self.gs_width

        interpolated_poses = self.pose_interpolation(pose1,pose2)

        dists, min_indice = self.compute_dists(interpolated_poses)

        save_path = os.path.join(self.save_dir, 'warp_images')
        os.makedirs(save_path, exist_ok=True)

        # image_o,image_o2,masks, cond_image = self.warp_images(self.gs_intrinsics, interpolated_poses, image_l=image1, image_r=image2, depth_l=depth1, depth_r=depth2, save_path=save_path, save_file_name_prefix=f'{idx1}') 
        image_o,image_o2,masks, cond_image = self.warp_images(self.diffusion_intrinsics, interpolated_poses, image_l=image1, image_r=image2, depth_l=depth1, depth_r=depth2, save_path=save_path, save_file_name_prefix=f'{idx1}') 

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

        # import pdb; pdb.set_trace() 
        # if replace:
        #     diffused_frames[0] = torch.from_numpy(np.asarray(image_o)).permute([2,0,1]).cuda()  #HWC-CHW
        #     diffused_frames[-1] = torch.from_numpy(np.asarray(image_o2)).permute([2,0,1]).cuda() 

    

        return diffused_frames, interpolated_poses
    
    def svd_render(self, image_l, image_r, masks, cond_image,output_path,lambda_ts, num_frames=25, save_prefix=''):
        # pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe = self.diffusion_pipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        # cv2.imwrite(os.path.join(output_path, "image_r.png"), image_r*255)

        frames = pipe([image_l],temp_cond=cond_image+[image_r],mask = masks,lambda_ts=lambda_ts,num_frames=num_frames, decode_chunk_size=8, num_inference_steps=100, latent_num=self.latent_num).frames[0] 
        # frames = pipe([image_l],temp_cond=cond_image+[image_r],mask = masks,lambda_ts=lambda_ts,num_frames=num_frames, decode_chunk_size=8,num_inference_steps=1).frames[0] 
   
        # output_path = save_path+'_out_dgs'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i,fr in enumerate(frames):
            fr.save(os.path.join(output_path, f"{save_prefix}{i:06d}.png"))

        export_to_video(frames, os.path.join(output_path,"generated.mp4"), fps=7)

        # pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        # pipe.to("cuda")

        # # frames = pipe([image_o],temp_cond=cond_image,mask = masks,lambda_ts=lambda_ts,weight_clamp=weight_clamp,num_frames=25, decode_chunk_size=8,num_inference_steps=100).frames[0]
        # frames = pipe([image_o],temp_cond=cond_image,mask = masks,lambda_ts=lambda_ts,weight_clamp=weight_clamp,num_frames=num_frames, decode_chunk_size=8,num_inference_steps=100).frames[0]
        # # frames = pipe([image_o],temp_cond=cond_image,mask = masks,lambda_ts=lambda_ts,weight_clamp=weight_clamp,num_frames=num_frames, decode_chunk_size=8,num_inference_steps=10).frames[0]

        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        # for i,fr in enumerate(frames):
        #     fr.save(os.path.join(output_path, f"{i:06d}.png"))
        # export_to_video(frames, os.path.join(output_path,"generated.mp4"), fps=7)

        return frames


    def search_hypers(self, pose,dists,sigmas,save_path):
        sigmas = sigmas[:-1]
        sigmas_max = max(sigmas)

        v2_list = np.arange(50, 1001, 50)
        v3_list = np.arange(10, 101, 10)
        v1_list = np.linspace(0.001, 0.009, 9)
        zero_count_default = 0
        index_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        for v1 in v1_list:
            for v2 in v2_list:
                for v3 in v3_list:
                    flag = True
                    lambda_t_list = []
                    for sigma in sigmas:
                        sigma_n = sigma/sigmas_max
                        temp_cond_indices = [0]
                        for tau in range(25):
                            if tau not in index_list:
                                lambda_t_list.append(1)
                            else:
                                dist_t = dists[tau]
                                Q = v3 * abs((dist_t)) - v2*sigma_n
                                k = 0.8
                                b = -0.2

                                lambda_t_1 = (-(2*v1 + k*Q) + ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                                lambda_t_2 = (-(2*v1 + k*Q) - ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                                v1_ = -v1
                                lambda_t_3 = (-(2*v1_ + k*Q) + ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                                lambda_t_4 = (-(2*v1_ + k*Q) - ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                                try:
                                    if np.isreal(lambda_t_1):
                                        if lambda_t_1 >1.0:
                                            lambda_t = lambda_t_1
                                            lambda_t_list.append(lambda_t/(1+lambda_t))
                                            continue
                                    if np.isreal(lambda_t_2):
                                        if lambda_t_2 >1.0:
                                            lambda_t = lambda_t_2
                                            lambda_t_list.append(lambda_t/(1+lambda_t))
                                            continue
                                    if np.isreal(lambda_t_3):
                                        if lambda_t_3 <=1.0 and lambda_t_3>0:
                                            lambda_t = lambda_t_3
                                            lambda_t_list.append(lambda_t/(1+lambda_t))
                                            continue
                                    if np.isreal(lambda_t_4):
                                        if lambda_t_4 <=1.0 and lambda_t_4>0:
                                            lambda_t = lambda_t_4
                                            lambda_t_list.append(lambda_t/(1+lambda_t))
                                            continue
                                    flag = False
                                    break
                                except:
                                    flag = False
                                    break
                                lambda_t_list.append(lambda_t/(1+lambda_t))
            
                    if flag == True:
                        zero_count = sum(1 for x in lambda_t_list if x > 0.5)
                        if zero_count > zero_count_default:
                            zero_count_default = zero_count
                            v_optimized = [v1,v2,v3]
                            lambda_t_list_optimized = lambda_t_list
                        
        X = np.array(sigmas)

        Y = np.arange(0,25,1)
        temp_i = np.array(temp_cond_indices)
        X, Y = np.meshgrid(X, Y)
        lambda_t_list_optimized = np.array(lambda_t_list_optimized)
        lambda_t_list_optimized = lambda_t_list_optimized.reshape([len(sigmas),25])

        lambda_t_list_optimized = torch.tensor(lambda_t_list_optimized)
        Z= lambda_t_list_optimized

        z_upsampled = F.interpolate(Z.unsqueeze(0).unsqueeze(0), scale_factor=10, mode='bilinear', align_corners=True)

        save_path = os.path.join(save_path,'lambad_'+str(v_optimized[0])+'_'+str(v_optimized[1])+'_'+str(v_optimized[2])+'.png')
        image_numpy = z_upsampled[0].permute(1, 2, 0).numpy()

        plt.figure()  
        plt.imshow(image_numpy)  
        plt.colorbar() 
        plt.axis('off')  

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close() 
        print(v_optimized)
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
        ])
        return interpolated_poses
    

    def compute_dists(self, interpolated_poses):

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

    def warp_images(self, intrinsics, interpolated_poses, image_l, image_r, depth_l, depth_r, save_path, save_file_name_prefix=''):
        '''
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

        pose_s_r = interpolated_poses[-1]
        # image_r = PIL.Image.open(os.path.join(img_path,'000002.jpg'))
        # image_r = image_r.resize((1024,576),PIL.Image.Resampling.NEAREST) 
        image_r = cv2.resize(image_r, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        image_o2 = image_r
        # image_r = np.array(image_r)


        depth_r = cv2.resize(depth_r, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        cond_image = []
        masks = []
        for i in range(23):
            if i<12:
                depth = depth_l
                pose_s = pose_s_l
                image = image_l
            else:
                depth = depth_r
                pose_s = pose_s_r
                image = image_r

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

        return image_o/255., image_o2/255., masks, cond_image

    

    def refine_GS(self, dense_views, dense_poses, intrinsics, cam_confidence=0.01, gs_start_iter=0, disable_densification=False, load_iteration=None):
        # refine GS with dense views
        refine_ckpts = glob.glob(f'{self.gsTrainer.scene.model_path}/refine_*_chkpnt*.pth') 
        # '''
        if len(refine_ckpts) > 0:
            refine_ckpts = sorted(refine_ckpts, key=lambda x: int(os.path.basename(x).split('_')[1]) )
            print(f"Load the latest refined GS: {refine_ckpts[-1]}...")
            self.gsTrainer.load_checkpoint(checkpoint=refine_ckpts[-1]) # load the latest refined GS
        else:
            checkpoint_path = os.path.join(self.gsTrainer.scene.model_path, "chkpnt" + str(self.gsTrainer.checkpoint_iterations[-1]) + ".pth")
            self.gsTrainer.load_checkpoint(checkpoint=checkpoint_path ) # load the last checkpoint from initial GS 
            print(f"Load the initilized GS: {checkpoint_path}...")
        # '''

        self.original_GS_train_cameras_bak = copy.deepcopy(self.gsTrainer.scene.train_cameras)

        # self.gsTrainer.update_cameras(dense_views, dense_poses, intrinsics, cam_confidences=0.3, append=True, ) # append the interpolated views to GS cameras 
        # import pdb; pdb.set_trace()
        # self.gsTrainer.update_cameras(dense_views, dense_poses, intrinsics, cam_confidences=0.01, append=True, load_iteration=-2) # append the interpolated views to GS cameras 
        self.gsTrainer.update_cameras(dense_views, dense_poses, intrinsics, cam_confidences=cam_confidence, append=True, load_iteration=load_iteration) # append the interpolated views to GS cameras 
        # self.gsTrainer.reset_scene()

        self.gsTrainer.reset_optimizers() # reset the optimizers for the new GS cameras ????
        self.gsTrainer.reset_gs() ###for debug 

        # epoch_iteration = self.gsTrainer.opt.iterations

        # self.gsTrainer.training(gs_start_iter)
        self.gsTrainer.finetune(0, self.refine_epoch, disable_densification=disable_densification) # finetune the GS with the new cameras)
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
        # import pdb; pdb.set_trace()
        cv2.imwrite(f'{self.save_dir}/reproj_error_{idx1}_{idx2}.png', reproj_error.cpu().numpy()/100*255)


        return reproj_error




    def run(self, refine_cycles=1):
        self.init_GS() # skip for now

        # self.refine_GS(dense_views=[], dense_poses=[], intrinsics=self.gs_intrinsics, confidence=None) # for debug
        for i in range(refine_cycles):
            
            down_sample_rate = 1 
            if self.num_input_views > 20:
                down_sample_rate = 300/self.num_input_views*25 # 300 is the max number of interpolated views 

            dense_views, dense_poses = self.densify_views(cycle_num=i,down_sample_rate=down_sample_rate)

            self.gsTrainer.opt.use_lpips_loss = True
            self.refine_GS(dense_views=dense_views, dense_poses=dense_poses, intrinsics=self.gs_intrinsics, cam_confidence=0.01, load_iteration=-2) # for debug: load colmap's pcd 

            # self.refine_GS(dense_views=[], dense_poses=[], intrinsics=self.gs_intrinsics, confidence=None) # for debug
            # self.refine_GS(dense_views, dense_poses, intrinsics=self.gs_intrinsics, confidence=None)
            self.gsTrainer.opt.use_lpips_loss = False

        self.refine_epoch = 1001 #1000

        self.gsTrainer.opt.densify_abs_grad_threshold = 20000#0.2
        self.gsTrainer.opt.densify_grad_threshold = 0.0008#0.0005
        self.gsTrainer.opt.use_lpips_loss = True
        # self.refine_GS(dense_views=[], dense_poses=[], intrinsics=self.gs_intrinsics, cam_confidence=0.001, disable_densification=True, load_iteration=None) # disable densification for the last refine cycle: load the last refined GS
        self.refine_GS(dense_views=dense_views, dense_poses=dense_poses, intrinsics=self.gs_intrinsics, cam_confidence=0.001, disable_densification=True, load_iteration=None) # disable densification for the last refine cycle: load the last refined GS
        # self.refine_GS(dense_views=[], dense_poses=[], intrinsics=self.gs_intrinsics, confidence=None, disable_densification=False, load_iteration=None) # disable densification for the last refine cycle
