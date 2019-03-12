#!/bin/bash

SOURCE_MODEL_NAME=$1   # e.g. 10
MODES=(1 1 4);
DIFFS=(0 1 0);
START_SEQ=$2
GPU_IDX=$3

seeds=(1763218 9977709 902073 4526168 8768338 6410600 1017197 7024514 5583941 8925548 8655125 6662210 9296437 964726 3634365 1260494 1203408 3841672 3769693 6092647 1755977 399558 3410388 7655408 6193727 72983 391298 7597 5611762 198814);

for outer_i in $(seq 0 2);
do
    MODE=${MODES[$outer_i]};
    DIFF=${DIFFS[$outer_i]};
    for i in $(seq 0 3);
    do
        let idx=$i+$START_SEQ;
        let seed=${seeds[$idx]};
        model_name=FREEWAY_M0D0_"$SOURCE_MODEL_NAME"_TRAIN_"$idx";
        echo "CUDA_VISIBLE_DEVICES=$GPU_IDX python main.py --rom ../../roms/freeway.bin --model_name $model_name --mode $MODE --difficulty $DIFF --evaluate True --max_episode_count 100 --random_seed $seed"
        CUDA_VISIBLE_DEVICES=$GPU_IDX python main.py --rom ../../roms/freeway.bin --model_name $model_name --mode $MODE --difficulty $DIFF --evaluate True --max_episode_count 100 --random_seed $seed 2> $model_name/out_M"$MODE"D"$DIFF".log && echo "RUN $i finished"
        echo "";
        sleep 5;
    done
done
