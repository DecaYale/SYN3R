#1passProb 
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/DNGaussian":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/FSGS":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/FSGS":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/mast3r":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/gmflow":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/NVS_Solver2/dust3r":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/gmflow":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/NVS_Solver/dust3r":$PYTHONPATH
##################DGS MODE#################
########## GS ##########

#dataset_root="/home/yxumich/durable/datasets/nerf_llff_data/"
# dataset_root="/home/yxumich/durable/datasets/DTU/dtu_colmap/"
# dataset_root="/home/yxumich/Works2/DATASETS/DL3DV/test_split/"
dataset_root="/home/yxumich/durable/datasets/mipnerf360/mipnerf360/"

workspace=$1
# export CUDA_VISIBLE_DEVICES=$3
n_views=$2 #15
# view_dir="colmap_${n_views}view"
# view_dir="colmap_15view"
# view_dir="24_views"

pretrained_root="/home/yxumich/Projects/NVS_Solver/output/FSGS/batch_dl3dv_sppseudo0svdpseudo1frmsingle1.midasGS.v4.3view.tune"
pretrained_root="/home/yxumich/Projects/NVS_Solver/output/FSGS/dus3r/batch_fsgs_dl3dv_sppseudo0svdpseudo1view${n_views}.v4.dust3r.interpv2post"
pretrained_root="/home/yxumich/Projects/NVS_Solver/output/FSGS/dus3r/batch_fsgs_dl3dv_sppseudo0svdpseudo1view${n_views}.v4.dust3r.interpv2post.flowmaskdenspcd.iter2"


mkdir -p ${workspace} 
idx=0
seq=(
"bicycle"
"bonsai"
"counter"
"garden"
"kitchen"
"room"
"stump"
)
# ("room" "fortress" "flower" "fortress" "room" "trex" "horns" "flower")  
# for seq_name in $(ls $dataset_root); do
# Reversed list
reversed=()

# Loop from end to start
for (( i=${#seq[@]}-1; i>=0; i-- )); do
    reversed+=("${seq[i]}")
done


for seq_name in "${seq[@]}"; do
# for seq_name in "${reversed[@]}"; do
    echo "Processing $seq_name"
    # if [ -f "${workspace}/${seq_name}/refine_0_chkpnt10000.pth" ]; then
    if [ -f "${workspace}/${seq_name}/refine_1_chkpnt10000.pth" ]; then
        echo "skip existing ${workspace}/${seq_name}"
        continue
    fi

    mkdir -p ${workspace}/${seq_name}
    # cp -a ${pretrained_root}/${seq_name}/*.pt ${workspace}/${seq_name}/ # cp nvs files
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
    # #---------midasd0, cam conf=0.2 lpips_weight 1----- baseline
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 100000000000000000000  
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001
    #     "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 
    

    # #---------midasd1 cam conf=0.2 lpips_weight 1-----
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001
    #     "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 

    # #---------midasd1 cam conf=0.2 lpips_weight 1 rand_pcd 1-----
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001 --rand_pcd
    #     "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 

    # # #---------midasd1 cam conf=0.2 lpips_weight 1 rand_pcd1  depth_warm1-----
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001 --rand_pcd --svd_depth_warmup 1
    #     "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProb" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs"  --dataset "dl3dv" --lpips_weight 1 $GS_args 

    # ---------midasd1 cam conf=0.2 lpips_weight 1 rand_pcd0 dust3r1 depth_warm1-----
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001  --svd_depth_warmup 1 --use_dust3r 1  --start_sample_svd_frame 2000 
    #     "
    # GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 1000000000000000000000 --sample_svd_pseudo_interval 1
    #     --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001  --svd_depth_warmup 1 --use_dust3r 0 --rand_pcd  --start_sample_svd_frame 2000 
    #     "
    GS_args="-s ${dataset_root}/${seq_name}/${view_dir} --model_path ${workspace}/${seq_name}  --eval  --n_views $n_views --sample_pseudo_interval 1000000000000000000000 --sample_svd_pseudo_interval 1
        --num_train_samples $n_views --images images_4 --resolution 1 --use_proximity_densify 0    --svd_depth_warmup 1 --use_dust3r 0   --start_sample_svd_frame 2000  
        "
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertain" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs_v2"  --dataset "dl3dv" --lpips_weight 1 $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v4.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertainPost" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs_v2"  --dataset "dl3dv" --lpips_weight 1 $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v5.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertainPost" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs_v2"  --dataset "dl3dv" --lpips_weight 1 --svd_l1_weight 2 $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v5.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertainPost" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs_v2"  --dataset "dl3dv" --lpips_weight 1 --svd_l1_weight 0 --refine_cycle_num 2 $GS_args 
    # python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v6.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertainPost" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs_v2"  --dataset "dl3dv" --lpips_weight 1 --svd_l1_weight 0 --refine_cycle_num 2 $GS_args | tee ${workspace}/${seq_name}/log.txt 2>&1
    python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v6.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertain" --interp_type "backward_warp" --cam_confidence 0.2 --pseudo_cam_sampling_rate 0.02  --densify_type "interpolate_gs_v2"  --dataset "dl3dv" --lpips_weight 1 --svd_l1_weight 0 --refine_cycle_num 0 $GS_args | tee ${workspace}/${seq_name}/log.txt 2>&1
    

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
