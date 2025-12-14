# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/DNGaussian":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/FSGS":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/FSGS":$PYTHONPATH

#1passProb 
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/FSGS":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/FSGS":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/mast3r":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/gmflow":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/dust3r":$PYTHONPATH
##################DGS MODE#################
########## GS ##########

dataset_root="/home/yxumich/durable/datasets/nerf_llff_data/"

workspace=$1
# export CUDA_VISIBLE_DEVICES=$3




idx=0
seq=("trex")
# ("room" "fortress" "flower" "fortress" "room" "trex" "horns" "flower")  
for seq_name in $(ls $dataset_root); do
# for seq_name in "${seq[@]}"; do
    echo "Processing $seq_name"

    if [ -f "${workspace}/${seq_name}/refine_1_chkpnt10000.pth" ]; then
        echo "skip existing ${workspace}/${seq_name}"
        continue
    fi

    mkdir -p ${workspace}/${seq_name}
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
    GS_args="-s ${dataset_root}/${seq_name} --model_path ${workspace}/${seq_name}  --eval  --n_views 3 --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1  
        --num_train_samples 3 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001  --svd_depth_warmup 1 --use_dust3r 0   --start_sample_svd_frame 2000 
        "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "from_single" --refine_cycle_num 2 $GS_args 
    python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v6.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertainPost"  --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "interpolate_gs_v2" --refine_cycle_num 2 --num_views_for_pcd_densification 1 $GS_args   | tee ${workspace}/${seq_name}/log.txt 2>&1

    idx=$((idx+1))

    if [ $(($idx%1)) -eq 0 ]; then
        wait
    fi
    # python -c "print('hi')" &


done
# GS_args="-s ${dataset} --model_path ${workspace} -r 8 --eval --n_sparse 3  --rand_pcd --iterations 6000 --lambda_dssim 0.2 
#   --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 
#             --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 
#             --split_opacity_thresh 0.1 --error_tolerance 0.00025 
#             --scaling_lr 0.003 
#             --shape_pena 0.002 --opa_pena 0.001 
#             --near 10
#             --num_train_samples 3
#             --checkpoint_iterations 6000
#             "
GS_args="-s ${dataset} --model_path ${workspace} -r 8 --eval --n_sparse 3  --rand_pcd --iterations 10000 --lambda_dssim 0.2 
  --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 10000 --percent_dense 0.01 
            --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 9500 --position_lr_start 500 
            --split_opacity_thresh 0.1 --error_tolerance 0.00025 
            --scaling_lr 0.003 
            --shape_pena 0.002 --opa_pena 0.001 
            --near 10
            --num_train_samples 3
            --checkpoint_iterations 10000
            "

########## NVS ##########
###single image

