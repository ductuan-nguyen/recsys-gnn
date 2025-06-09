#!/bin/bash

# shell file for Tmall
model='MBGCN'
dataset_name='ubc'
gpu_id='0'
create_embeddings='False'
es_patience="50"
embedding_size='64'
lamb='1'

mgnn_weight1='1'
mgnn_weight2='1'
mgnn_weight3='1'
# mgnn_weight4='1'

# relation='buy,cart,collect,click'
relation='buy,cart,remove'
# relation='buy,cart'
epoch="10"
# pretrain_path='output/mf/ubc/ubc-MF_lr1e-2-L1e-6-size32@tuannd/model_epoch_19.pkl'
pretrain_path='output/ubc/ubc-MF_lr1e-2-L1e-6-size64@tuannd/model_epoch_19.pkl'

mode='test'
output_test='submits/3-rel-mf64-freeze-emb-uu-add-item-user-feat'
trained_mbgcn_model='/home/tuannd/MBGCN/output/ubc/ubc-MBGCN_lr3e-4-L1e-4-size64-lamb1-md0.1-nd0.1@3rel-mf64-freeze-emb-uu-add-item-user-feat-concat/model_epoch_9.pkl'
# lr_list=('3e-5')
# L2_list=('1e-5')
# lr_list=('1e-4' '3e-5' '1e-5' '3e-6')
# L2_list=('1e-2' '1e-3' '1e-4' '1e-5' '1e-6')

lr='3e-4'
L2='1e-4'
message_dropout=('0.1')
node_dropout=('0.1')
# message_dropout=('0' '0.1' '0.2' '0.3' '0.4' '0.5')
# node_dropout=('0' '0.1' '0.2' '0.3' '0.4' '0.5')

for md in ${message_dropout[@]}
do
    for nd in ${node_dropout[@]}
    do
        name=${dataset_name}-${model}_lr${lr}-L${L2}-size${embedding_size}-lamb${lamb}-md${md}-nd${nd}@3rel-mf64
        python main.py \
            --name ${name} \
            --model ${model} \
            --gpu_id ${gpu_id} \
            --dataset_name ${dataset_name} \
            --epoch ${epoch} \
            --L2_norm ${L2} \
            --lr ${lr} \
            --create_embeddings ${create_embeddings} \
            --es_patience ${es_patience} \
            --embedding_size ${embedding_size}\
            --mgnn_weight ${mgnn_weight1}\
            --mgnn_weight ${mgnn_weight2}\
            --mgnn_weight ${mgnn_weight3}\
            --lamb ${lamb} \
            --trained_mbgcn_model ${trained_mbgcn_model} \
            --mode ${mode} \
            --message_dropout ${md} \
            --node_dropout ${nd} \
            --output_test ${output_test} \
            --relation ${relation} \
            --pretrain_path ${pretrain_path}
    done
done
