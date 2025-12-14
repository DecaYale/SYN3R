#1passProb 
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/DNGaussian":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/NVS_Solver3/":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/NVS_Solver3/FSGS":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/FSGS":$PYTHONPATH
##################DGS MODE#################
########## GS ##########

#dataset_root="/home/yxumich/durable/datasets/nerf_llff_data/"
# dataset_root="/home/yxumich/durable/datasets/DTU/dtu_colmap/"
dataset_root="/home/yxumich/Works2/DATASETS/DL3DV/test_split/"

workspace=$1
# export CUDA_VISIBLE_DEVICES=$3
n_views=$2 #15
# view_dir="colmap_${n_views}view"
view_dir="colmap_15view"

pretrained_root="/home/yxumich/Projects/NVS_Solver/output/FSGS/batch_dl3dv_sppseudo0svdpseudo1frmsingle1.midasGS.v3.9view.tune"


mkdir -p ${workspace} 
idx=0
seq=("03f5c560f5725ad6ca55fd7e6c0af4c4c7a7ca94c444a584f2a9f316d3b35ea2"
"0850228cdbf7df721a10d73003db4b8d9d83e17c480b79a6d5d643eff6c8c163"
"0a78c25f77c1ba1d1a3f07c18c9735ae1254a9a71290734b8836eefbefaadbc7"
"25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b"
"51a802f3dc0268da35ad944e92cc7266fef00680816eb30d5847d5845b3e867a"
"6ed1058f96df97f1c8175739843cf0f272ce0c60c5727dbb010a3a0fac3ef10d"
"87c8b2841c276f00d10c53c32ffe628fb26fa3d2cd3ab7bb577ff25d31ee5dbd" 
"97f72cff0be96647eeb2fe17ac49752c739af5d1cda656b52e83917a4b2bc17d"
"9daa05c4182bb2ea065d280d4f510929d8e9c6d6e18a0782031c7c805cb822ec"
"9e4da70fe0be5d28ea7b375291bbf5523246345d807aa47d5208c6e6c2f5694c"
)
# ("room" "fortress" "flower" "fortress" "room" "trex" "horns" "flower")  
# for seq_name in $(ls $dataset_root); do
for seq_name in "${seq[@]}"; do
    echo "Processing $seq_name"
    
    mkdir -p ${workspace}/${seq_name}
    cp -a ${pretrained_root}/${seq_name}/*.pt ${workspace}/${seq_name}/ # cp nvs files
    cp -a $0 ${workspace}/${seq_name} # backup script

    # GS_args="-s ${dataset_root}/${seq_name} --model_path ${workspace}/${seq_name} -r 8 --eval --n_sparse 3  --rand_pcd --iterations 10000 --lambda_dssim 0.2 
    # GS_args="-s ${dataset_root}/${seq_name} --model_path ${workspace}/${seq_name} 
    #         -r 8 --eval --n_sparse 3  --mvs_pcd --iterations 10000 --lambda_dssim 0.25 
    #         --densify_grad_threshold 0.0008 --prune_threshold 0.01 --densify_until_iter 10000 --percent_dense 0.01 
    #         --position_lr_init 0.0005 --position_lr_final 0.000005 --position_lr_max_steps 9500 --position_lr_start 500 
    #         --split_opacity_thresh 0.1 --error_tolerance 0.01 
    #         --scaling_lr 0.005 
    #         --shape_pena 0.002 --opa_pena 0.001 --scale_pena 0
    #         --near 10
    #         --num_train_samples 3
    #         --checkpoint_iterations 10000
    #         "
    #---------midasd0, cam conf=0.2 lpips_weight 1----- baseline
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 100000000000000000000  
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001
    #     "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 
    

    # #---------midasd1 cam conf=0.2 lpips_weight 1-----
    GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1
        --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001 --use_lpips_finetune 0 
        "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 
    python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4_ablation.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 

    # #---------midasd1 cam conf=0.2 lpips_weight 1 randpcd 1-----
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001 --rand_pcd
    #     "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 
    
    #---------midasd1 cam conf=0.2 lpips_weight 1 randpcd 1 depthwarm-----
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001 --rand_pcd --svd_depth_warmup 1
    #     " 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 

    # #---------midasd0, cam conf=0.2 lpips_weight 5-----
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 100000000000000000000  
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001
    #     "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 5 $GS_args 
    
    #---------midasd0, cam conf=0.4 lpips_weight 1-----  
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 100000000000000000000  
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001
    #     "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.4 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 
    



    idx=$((idx+1))

    if [ $(($idx%1)) -eq 0 ]; then
        wait
    fi
    # break 


done
