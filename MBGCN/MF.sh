#!/bin/bash
# shell file for Tmall
model='MF'
dataset_name='ubc'
gpu_id='1'
create_embeddings='True'
es_patience="40"
embedding_size='32'
epoch="20"
output="/home/tuannd/MBGCN/output/mf"
# lr_list=('1e-2' '3e-3' '1e-3' '3e-4' '1e-4')
# L2_list=('1e-2' '1e-3' '1e-4' '1e-5' '1e-6')
lr_list=('1e-2' '3e-3')
L2_list=('1e-2' '1e-3' '1e-4' '1e-5' '1e-6')
for lr in ${lr_list[@]}
do
    for L2 in ${L2_list[@]}
    do
        name=${dataset_name}-${model}_lr${lr}-L${L2}-size${embedding_size}@tuannd
        python main.py \
            --name ${name} \
            --model ${model} \
            --gpu_id ${gpu_id} \
            --dataset_name ${dataset_name} \
            --L2_norm ${L2} \
            --lr ${lr} \
            --create_embeddings ${create_embeddings} \
            --es_patience ${es_patience} \
            --output ${output} \
            --epoch ${epoch} \
            --embedding_size ${embedding_size}
    done
done