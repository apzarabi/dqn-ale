#!/bin/bash

SOURCE_MODEL_NAME=$1   # e.g. 10
MODE=$2    # e.g. 1
DIFF=$3
START_SEQ=$4
GPU_IDX=$5

seeds=(5010403 2210265 6441506 2677247 956850 742235 8751009 4601299 8134312 1866342 1991829 7030309);

for i in $(seq 0 3);
do
    let idx=$i+$START_SEQ;
    seed=${seeds[$idx]};
    model_name=FREEWAY_M0D0_"$SOURCE_MODEL_NAME"_TRAIN_"$idx";
    echo "CUDA_VISIBLE_DEVICES=$GPU_IDX python main.py --rom ../../roms/freeway.bin --model_name $model_name --mode $MODE --difficulty $DIFF --evaluate True --max_episode_count 100 --random_seed ${seed[$i]}"
    CUDA_VISIBLE_DEVICES=$GPU_IDX python main.py --rom ../../roms/freeway.bin --model_name $model_name --mode $MODE --difficulty $DIFF --evaluate True --max_episode_count 100 --random_seed ${seed[$i]} 2> $model_name/out_M"$MODE"D"$DIFF".log && echo "RUN $i finished"
    echo "";
    sleep 5;
done
