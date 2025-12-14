
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS2/FSGS/":$PYTHONPATH


dataset_root="/home/yxumich/durable/datasets/nerf_llff_data/"
workspace=$1


idx=0
for seq_name in $(ls $dataset_root); do
# for seq_name in "${seq[@]}"; do
    echo "Processing $seq_name"

    for chkpt in $(ls ${workspace}/${seq_name} | grep "chkpnt"); do
        echo "Processing $seq_name $chkpt"
        python FSGS/render.py -s ${dataset_root}/${seq_name} --model_path ${workspace}/${seq_name} --checkpoint $chkpt --video #-r 8 --near 10 
        # python DNGaussian/metrics.py --model_path $workspace > ${workspace}/${seq_name}/eval_res.txt 
    done

    python FSGS/metrics.py --source_path $dataset_root/${seq_name}  --model_path  ${workspace}/${seq_name} > ${workspace}/${seq_name}/eval_res.txt 

    idx=$((idx+1))

    if [ $(($idx%2)) -eq 0 ]; then
        wait
    fi

done