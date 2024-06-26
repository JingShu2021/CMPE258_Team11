# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang University-model. All Rights Reserved.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

# # pretraining on COCOTextV2
PRETRAIN=/data/cmpe258-sp24/jingshu/Data/pretrained_path/pretrain_COCOText_checkpoint.pth
# PRETRAIN=exps/e2e_TransVTS_r50_COCOTextV2/checkpoint.pth
# PRETRAIN=exps/e2e_TransVTS_r50_SynthText/checkpointMOTA17.3IDF142.9.pth
# EXP_DIR=exps/e2e_TransVTS_r50_COCOTextV2
EXP_DIR=/data/cmpe258-sp24/jingshu/exps/e2e_TransDETR_r50_COCOTextV2
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
    # --use_env main.py \
CUDA_VISIBLE_DEVICES=0 
python3 main.py \
    --meta_arch TransDETR_ignored \
    --dataset_file VideoText \
    --epochs 10 \
    --with_box_refine \
    --lr_drop 30 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 2 \
    --sampler_steps 20 40 50 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --rec \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path /data/cmpe258-sp24/jingshu/Data/Dataset\
    --data_txt_path_train ./tools/gen_labels/datasets/data_path/COCOTextV2.train \
    --data_txt_path_val ./tools/gen_labels/datasets/data_path/COCOTextV2.train \
    --pretrained ${PRETRAIN} 
    
