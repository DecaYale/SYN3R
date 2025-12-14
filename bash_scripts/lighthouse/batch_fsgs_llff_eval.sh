
# export PYTHONPATH="/home/yxumich/Projects/FSGS/":$PYTHONPATH
export PYTHONPATH="/home/yxumich/Projects/SV2C_GS2/FSGS/":$PYTHONPATH

# export PYTHONPATH="/home/yxumich/Projects/FSGS/DNGaussian":$PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:$(pwd)
##################DGS MODE#################
########## GS ##########



dataset_root="/home/yxumich/durable/datasets/nerf_llff_data/"
# dataset="/home/yxumich/durable/datasets/nerf_llff_data/$1/"
# workspace=$2
workspace=$1


idx=0
seq=("horns" "trex")
# ("room" "fortress" "flower" "fortress" "room" "trex" "horns" "flower")  
for seq_name in $(ls $dataset_root); do
# for seq_name in "${seq[@]}"; do
    echo "Processing $seq_name"

    for chkpt in $(ls ${workspace}/${seq_name} | grep "chkpnt"); do
        echo "Processing $seq_name $chkpt"
        python FSGS/render.py -s ${dataset_root}/${seq_name} --model_path ${workspace}/${seq_name} --checkpoint $chkpt --video #-r 8 --near 10 
        # python DNGaussian/metrics.py --model_path $workspace > ${workspace}/${seq_name}/eval_res.txt 
    done

    # python train.py  --source_path $dataset_root/${seq_name} --model_path ${workspace}/${seq_name} --eval  --n_views 3 --sample_pseudo_interval 1 &

    # python render.py --source_path $dataset_root/${seq_name}  --model_path  ${workspace}/${seq_name} --iteration 10000

    python FSGS/metrics.py --source_path $dataset_root/${seq_name}  --model_path  ${workspace}/${seq_name} > ${workspace}/${seq_name}/eval_res.txt 



    idx=$((idx+1))

    if [ $(($idx%2)) -eq 0 ]; then
        wait
    fi
    # python -c "print('hi')" &


done