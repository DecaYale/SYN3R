export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/":$PYTHONPATH

export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/thirdparty/":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/thirdparty/FSGS":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/thirdparty/gmflow":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS/thirdparty/dust3r":$PYTHONPATH
##################DGS MODE#################
########## GS ##########

dataset_root="/home/yxumich/durable/datasets/DTU/dtu_colmap/"

workspace=$1




idx=0
for seq_name in $(ls $dataset_root); do
# for seq_name in "${seq[@]}"; do

    echo "Processing $seq_name"
    mkdir -p ${workspace}/${seq_name}
    # cp -a ${pretrained_root}/${seq_name}/*.pt ${workspace}/${seq_name}/ # cp nvs files
    cp -a $0 ${workspace}/${seq_name} # backup script

    if [ -f "${workspace}/${seq_name}/refine_1_chkpnt10000.pth" ]; then
        echo "skip existing ${workspace}/${seq_name}"
        continue
    fi

    GS_args="-s ${dataset_root}/${seq_name} --model_path ${workspace}/${seq_name}  --eval  --n_views 3  --sample_svd_pseudo_interval 1  
        --num_train_samples 3 --images images --resolution 4 --lambda_dssim 0.5 
        "
    python scripts/GSNVS_solver_multiview_2pass2latent_fsgs_v6.py  --iteration dgs1  --weight_clamp 0.2 --diffusion_type "2PassProbUncertain" --interp_type "backward_warp" --cam_confidence 0.05 --pseudo_cam_sampling_rate 0.02 --densify_type "interpolate_loop0_gs"  --dataset "dtu" --refine_cycle_num 2 $GS_args  | tee ${workspace}/${seq_name}/log.txt 2>&1

    idx=$((idx+1))

    if [ $(($idx%1)) -eq 0 ]; then
        wait
    fi

done