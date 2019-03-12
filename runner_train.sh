#!/bin/bash

MODEL_NAME=$1   # e.g. 10
START_SEQ=$2;

seed=($3 $4 $5 $6);

if [ $MODEL_NAME = "MF" ]; then
    model_freq=10000000;    # ten million
else
    model_freq=$MODEL_NAME;
fi


for i in $(seq 0 3);
do
    let idx=$i+$START_SEQ;
    model_name=FREEWAY_M0D0_"$MODEL_NAME"_TRAIN_"$idx";
    if [ -e $model_name ]
    then
        echo "FILE EXISTS";
        echo $model_name;
    else
        mkdir $model_name;
        
        touch $model_name/in_training.stat;
        touch $model_name/train_stat.txt;
        hostname > $model_name/train_stat.txt;
        echo $i >> $model_name/train_stat.txt;

        CUDA_VISIBLE_DEVICES=$i python main.py --rom ../../roms/freeway.bin --model_name $model_name --model_freq $model_freq --random_seed ${seed[$i]} 2>> $model_name/train_stat.txt && echo "RUN $i finished" && rm $model_name/in_training.stat &
        echo "";
        sleep 2;
    fi
done
