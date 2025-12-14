#1passProb 
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/DNGaussian":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/FSGS":$PYTHONPATH
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



idx=0
# seq=("25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b")
# ("room" "fortress" "flower" "fortress" "room" "trex" "horns" "flower")  
for seq_name in $(ls $dataset_root); do
# for seq_name in "${seq[@]}"; do
    echo "Processing $seq_name"
    if [ -f  "${workspace}/${seq_name}/chkpnt_latest.pth"  ]; then
        echo "File exists"
        continue
    fi
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
    GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1  
        --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001
        "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "from_single"  --dataset "dtu" $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "interpolate"  --dataset "dtu" $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v2.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "interpolate_gs"  --dataset "dl3dv" $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "interpolate_gs"  --dataset "dl3dv" $GS_args 
    python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02 --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v2.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "interpolate"  --dataset "dl3dv" $GS_args 

    idx=$((idx+1))

    if [ $(($idx%1)) -eq 0 ]; then
        wait
    fi
    # break 


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


# python scripts/GSNVS_solver_multiview_2pass2latent_dngs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass2Latent" --interp_type "forward_warp" $GS_args
# python scripts/GSNVS_solver_multiview_2pass2latent_dngs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass2Latent" --interp_type "backward_warp" --start_checkpoint /home/yxumich/Works2/PGSR/output/llff/room_nvs2.verify/chkpnt6000.pth $GS_args
# python scripts/GSNVS_solver_multiview_2pass2latent_dngs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass" --interp_type "backward_warp" $GS_args #--start_checkpoint /home/yxumich/Works2/PGSR/output/llff/room_nvs2.verify/chkpnt6000.pth 
# python scripts/GSNVS_solver_multiview_2pass2latent_dngs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass2Latent" --interp_type "backward_warp" $GS_args #--start_checkpoint /home/yxumich/Works2/PGSR/output/llff/room_nvs2.verify/chkpnt6000.pth 
# python scripts/GSNVS_solver_multiview_2pass2latent_dngs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass2Latent" --interp_type "backward_warp" --pseudo_cam_sampling_rate 0.1 $GS_args #--start_checkpoint /home/yxumich/Works2/PGSR/output/llff/room_nvs2.verify/chkpnt6000.pth 
# python scripts/GSNVS_solver_multiview_2pass2latent_dngs.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2Pass" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 $GS_args #--start_checkpoint /home/yxumich/Works2/PGSR/output/llff/room_nvs2.verify/chkpnt6000.pth 




# # python train_llff.py  -s $dataset --model_path $workspace -r 8 --eval --n_sparse 3  --rand_pcd --iterations 6000 --lambda_dssim 0.2 \
# python utils/trainer.py  -s $dataset --model_path $workspace -r 8 --eval --n_sparse 3  --rand_pcd --iterations 6000 --lambda_dssim 0.2 \
#             --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
#             --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
#             --split_opacity_thresh 0.1 --error_tolerance 0.00025 \
#             --scaling_lr 0.003 \
#             --shape_pena 0.002 --opa_pena 0.001 \
#             --near 10

# set a larger "--error_tolerance" may get more smooth results in visualization

            
# python DNGaussian/render.py -s $dataset --model_path $workspace -r 8 --near 10  
# python spiral.py -s $dataset --model_path $workspace -r 8 --near 10 


# python DNGaussian/metrics.py --model_path $workspace 