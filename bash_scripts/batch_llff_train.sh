export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/":$PYTHONPATH

export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/thirdparty/":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/thirdparty/FSGS":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/thirdparty/gmflow":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/thirdparty/dust3r":$PYTHONPATH
##################DGS MODE#################
########## GS ##########

dataset_root="/home/yxumich/durable/datasets/nerf_llff_data/"

workspace=$1


idx=0
# seq=("trex")
# ("room" "fortress" "flower" "fortress" "room" "trex" "horns" "flower")  
# for seq_name in "${seq[@]}"; do
for seq_name in $(ls $dataset_root); do
    echo "Processing $seq_name"

    if [ -f "${workspace}/${seq_name}/refine_1_chkpnt10000.pth" ]; then
        echo "skip existing ${workspace}/${seq_name}"
        continue
    fi

    mkdir -p ${workspace}/${seq_name}
    cp -a $0 ${workspace}/${seq_name} # backup script


    GS_args="-s ${dataset_root}/${seq_name} --model_path ${workspace}/${seq_name}  --eval  --n_views 3 --sample_pseudo_interval 100000000000000000000 --sample_svd_pseudo_interval 1  
        --num_train_samples 3 --resolution 1 --use_proximity_densify 0 --densify_grad_threshold 0.0002  --percent_dense 0.001  --svd_depth_warmup 1 --use_dust3r 0   --start_sample_svd_frame 2000 
        "
    python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v6.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertainPost"  --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "interpolate_gs_v2" --refine_cycle_num 2 --num_views_for_pcd_densification 1 $GS_args   | tee ${workspace}/${seq_name}/log.txt 2>&1

    idx=$((idx+1))

    if [ $(($idx%1)) -eq 0 ]; then
        wait
    fi

done
