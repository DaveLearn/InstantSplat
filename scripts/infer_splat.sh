#!/bin/bash

# Change the absolute path first!
DATA_ROOT_DIR="/home/david/projects/embodied_gaussians/datasets"
OUTPUT_DIR="output_infer"


# ensure the scene is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0  <scene> where scene is folder in $DATA_ROOT_DIR/ eg: real/multiple1_aruco" 
    exit 1
fi

SCENE=$1

# if Scene starts with "real" then use 5 views, otherwise use 6 views
if [[ $SCENE == real* ]]; then
    N_VIEWS=5
else
    N_VIEWS=6
fi

gs_train_iter=1500


# Function to get the id of an available GPU
get_available_gpu() {
    local mem_threshold=500
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1; exit }
    '
}

# Function: Run task on specified GPU
run_on_gpu() {
    local GPU_ID=$1
    local SCENE=$2
    local N_VIEW=$3
    local gs_train_iter=$4
    SOURCE_PATH=${DATA_ROOT_DIR}/${SCENE}/modelling/static/
    IMAGE_PATH=${SOURCE_PATH}color
    MODEL_PATH=./${OUTPUT_DIR}/${SCENE}/${N_VIEW}_views

    # Create necessary directories
    mkdir -p ${MODEL_PATH}

    echo "======================================================="
    echo "Starting process: ${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="

    # (1) Co-visible Global Geometry Initialization
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
    -s ${IMAGE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW} \
    --co_vis_dsp \
    --conf_aware_ranking \
    --infer_video \
    --niter 500 \
    --lr 0.03 \
    --ckpt_path naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    2>&1  | tee ${MODEL_PATH}/01_init_geo.log
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"


    # (2) Train: jointly optimize pose
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --images color \
    --n_views ${N_VIEW} \
    --iterations ${gs_train_iter} \
    --depth_ratio 0 \
    --lambda_dist 10 \
     2>&1  | tee ${MODEL_PATH}/02_train.log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"

    # (3) Render-Training_View
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEW} \
    --iterations ${gs_train_iter} \
    --depth_ratio 0 \
    --num_cluster 50 \
    --mesh_res 2048 \
    --skip_mesh \
    --depth_trunc 4.0 \
    2>&1  | tee ${MODEL_PATH}/03_render_train.log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/03_render_train.log"
    # --voxel_size 0.004 \
    # --sdf_trunc 0.016 \


    echo "======================================================="
    echo "Task completed: ${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="
}

# Main loop

run_on_gpu 0 "$SCENE" "$N_VIEWS" "$gs_train_iter"


# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="
