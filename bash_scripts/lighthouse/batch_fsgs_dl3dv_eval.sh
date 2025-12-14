
export PYTHONPATH="/home/yxumich/Projects/FSGS/":$PYTHONPATH
# export PYTHONPATH="/home/yxumich/Projects/FSGS/DNGaussian":$PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:$(pwd)
##################DGS MODE#################
########## GS ##########



dataset_root="/home/yxumich/Works2/DATASETS/DL3DV/test_split/"

# workspace=$2
workspace=$1
n_views=$2 #15
view_dir="colmap_${n_views}view"
if [ $n_views != 15 ] && [ $n_views != 24 ]; then
    echo "Invalid n_views"
    view_dir="colmap_15view"
fi

idx=0
seq=("horns" "trex")
seq=(
    25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b
)
# ("room" "fortress" "flower" "fortress" "room" "trex" "horns" "flower")  
for seq_name in $(ls $dataset_root); do
# for seq_name in "${seq[@]}"; do
    if [ -d "${workspace}/${seq_name}/test/ours_refine_1_chkpnt10000.pth" ]; then
        echo "skip existing ${workspace}/${seq_name}"
        # continue
    fi

    echo "Processing $seq_name"



    # for chkpt in $(ls ${workspace}/${seq_name} | grep "chkpnt"); do
    for chkpt in $(ls ${workspace}/${seq_name} | grep "chkpnt10000"); do
        echo "Processing $seq_name $chkpt"
        # python FSGS/render.py -s ${dataset_root}/${seq_name}/${view_dir}  --model_path ${workspace}/${seq_name} --checkpoint $chkpt --video --eval #-r 8 --near 10 
        python FSGS/render.py -s ${dataset_root}/${seq_name}/${view_dir}  --model_path ${workspace}/${seq_name} --checkpoint $chkpt  --eval #-r 8 --near 10 
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