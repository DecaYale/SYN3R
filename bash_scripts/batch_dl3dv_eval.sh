
# YOU NEED TO MODIFY THE PATHS IN THE SCRIPT BELOW BEFORE RUNNING IT
PROJECT_DIR=/home/yxumich/Projects/SYN3R/
DATA_DIR=/home/yxumich/Works2/DATASETS/

export PYTHONPATH="${PROJECT_DIR}":$PYTHONPATH

export PYTHONPATH="${PROJECT_DIR}/thirdparty/":$PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/thirdparty/FSGS":$PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/thirdparty/gmflow":$PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/thirdparty/dust3r":$PYTHONPATH


dataset_root="${DATA_DIR}/DL3DV/test_split/"

workspace=$1
n_views=$2 #9
view_dir="colmap_${n_views}view"
if [ $n_views != 15 ] && [ $n_views != 24 ]; then
    echo "Invalid n_views"
    view_dir="colmap_dense"
fi

idx=0
# seq=(
#     25f7dbc10c0e2a9a8ffa33c35660d9090b6f7df6478653e351b3cb1195f7347b
# )
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
        python thirdparty/FSGS/render.py -s ${dataset_root}/${seq_name}/${view_dir}  --model_path ${workspace}/${seq_name} --checkpoint $chkpt  --eval #-r 8 --near 10 
    done

    python thirdparty/FSGS/metrics.py --source_path $dataset_root/${seq_name}  --model_path  ${workspace}/${seq_name} > ${workspace}/${seq_name}/eval_res.txt 


    idx=$((idx+1))

    if [ $(($idx%2)) -eq 0 ]; then
        wait
    fi


done