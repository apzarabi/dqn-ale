#!/bin/bash

model=05;
RUNS=$(seq 4 11);
MODES=(1 1 4);
DIFFS=(0 1 0);

let modes_len=${#MODES[@]}-1

for run in $RUNS;
do
    model_name=FREEWAY_M0D0_"$model"_TRAIN_"$run"
    echo $model_name;
    for j in $(seq 0 $modes_len);
    do
        mode=${MODES[$j]}
        diff=${DIFFS[$j]}
        game=M"$mode"D"$diff";
        scp eureka:/remote/eureka1/pourzara/jesse/dqn-ale/"$model_name"/eval_"$game"/episodeResults.csv "$model"/"$game"/"$run".csv
    done
done
