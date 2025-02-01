#!/bin/bash
# ds_root is $1 or default to my folder
DS_ROOT=${1:-/home/david/projects/embodied_gaussians/datasets}

for scene in $(ls $DS_ROOT/real); do
    echo "Splatting $scene"
    ./scripts/infer_splat.sh real/$scene
done

for scene in $(ls $DS_ROOT/simulated); do
    echo "Splatting $scene"
    ./scripts/infer_splat.sh simulated/$scene
done