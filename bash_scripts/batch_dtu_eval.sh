

# YOU NEED TO MODIFY THE PROJECT_DIR IN THE SCRIPT BELOW BEFORE RUNNING IT
PROJECT_DIR=/home/yxumich/Projects/SYN3R/
DATA_DIR=/home/yxumich/durable/datasets/

export PYTHONPATH="${PROJECT_DIR}":$PYTHONPATH

export PYTHONPATH="${PROJECT_DIR}/thirdparty/":$PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/thirdparty/FSGS":$PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/thirdparty/gmflow":$PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/thirdparty/dust3r":$PYTHONPATH
dataset_root="${DATA_DIR}/DTU/dtu_colmap/"


workspace=$1


idx=0
for seq_name in $(ls $dataset_root); do
    echo "Processing $seq_name"

    if [ -d "${workspace}/${seq_name}/test/" ]; then
        echo "${workspace}/${seq_name}/test/ exists, skip"
        # continue
    fi

    for chkpt in $(ls ${workspace}/${seq_name} | grep "chkpnt"); do
        echo "Processing $seq_name $chkpt"
        python FSGS/render.py --eval -s ${dataset_root}/${seq_name} --model_path ${workspace}/${seq_name} --checkpoint $chkpt  --video #-r 8 --near 10 
        # break 
    done

    python FSGS/metrics_dtu.py   --model_path  ${workspace}/${seq_name} > ${workspace}/${seq_name}/eval_res.txt 



    idx=$((idx+1))

    if [ $(($idx%2)) -eq 0 ]; then
        wait
    fi


done