# support feeding gs rendering to svd 
# use trainer_v3
# use trainer_v4
# perturb the interpolated poses --- use diffusionGS_v3_6


import matplotlib.pyplot as plt

import argparse

from dataclasses import dataclass

from diffusers.utils import load_image, export_to_video
try:
    from FSGS.utils.trainer_v4 import init_GSTrainer, GSTrainer
    from FSGS.arguments import ModelParams, PipelineParams, OptimizationParams
except ImportError:
    
    raise 
    pass
import sys 

from scipy.optimize import minimize_scalar



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #NVS solver's parameters
    parser.add_argument('--major_radius',type=float,default=50 )
    parser.add_argument('--minor_radius',type=float,default=40 )
    parser.add_argument('--weight_clamp',type=float,default=0.4 )
    parser.add_argument('--image_path',type=str)
    parser.add_argument('--folder_path',type=str)
    parser.add_argument('--iteration',type=str)
    parser.add_argument('--inverse',type=bool,default=False )
    parser.add_argument('--degrees_per_frame',type=float,default=0.4 ) 

    
    #GS's parameters
    # from PGSR.arguments import ModelParams, PipelineParams, OptimizationParams

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--diffusion_type", type=str, default ='2Pass')
    parser.add_argument("--interp_type", type=str, default ='forward_warp', choices=['forward_warp', 'backward_warp'])
    parser.add_argument("--near", type=int, default=0)
    parser.add_argument("--cam_confidence", type=float, default=0.1)
    parser.add_argument("--pseudo_cam_sampling_rate", type=float, default=0.04)


    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 10_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 50_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[80_00, 10_000])
    parser.add_argument("--train_bg", action="store_true")
    # parser.add_argument("--use_proximity_densify", type=bool, default=True)
    parser.add_argument("--densify_type", type=str, default ='interpolate', choices=['interpolate', 'from_single', 'from_single_gs', 'interpolate_gs', 'interpolate_loop0_gs', 'interpolate_gs_v2'])
    parser.add_argument("--dataset", type=str, default ='llff', choices=['llff', 'dtu', 'dl3dv'])
    parser.add_argument("--refine_cycle_num", type=int, default =1)

    parser.add_argument("--reorg_train_views", type=int, default =1)
    parser.add_argument("--num_views_for_pcd_densification", type=int, default=4)

    args = parser.parse_args(sys.argv[1:])
    print(args.folder_path)
    

    num_frames = 25  # Number of camera poses to generate




    try:
        # gsTrainer = init_GSTrainer(args, lp, op, pp, shuffle_cameras=False)  # intialize the GS scene 
        # gsTrainer = init_GSTrainer(args, lp, op, pp, shuffle_cameras=False, load_iteration=-1 )  # intialize the GS scene 
        gsTrainer = init_GSTrainer(args, lp, op, pp, shuffle_cameras=False, load_iteration=None)  # intialize the GS scene
        gs_start_iter = 0
        epoch_iteration = gsTrainer.opt.iterations
    except:
        raise
        print("Failed to initialize GS scene. If you are debugging, it's okay. Otherwise, check the error message above.")

    from model.diffusionGS_v3_6 import DiffusionGS
    # from model.diffusionGS_v3_5 import DiffusionGS
    runner = DiffusionGS(gsTrainer, num_input_views=args.num_train_samples, diffusion_type=args.diffusion_type, save_dir=args.model_path, interp_type=args.interp_type, input_args=args)
    # runner = DiffusionGS(gsTrainer, num_input_views=args.num_train_samples, diffusion_type='2Pass', save_dir=args.model_path, debug=True)

    runner.run(args.refine_cycle_num)

    







