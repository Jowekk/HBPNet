 DATASET_DIR=../data/dataset/
 TRAIN_DIR=../data/MSCNN/log_HBPNet
 CHECKPOINT_PATH=../data/checkpoints/vgg_16.ckpt
 python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=aircraft \
    --dataset_split_name=train \
    --model_name=vgg_16 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=vgg_16/hbp_part/ \
    --batch_size=16 \
    --learning_rate=1e-1 \
    --trainable_scopes=vgg_16/hbp_part \
    --optimizer=momentum \
    --momentum=0.9 \
    --learning_rate_decay_factor=0.5 \
    --weight_decay=5e-6 \
    --num_epochs_per_decay=24 \
    --max_number_of_steps=60000
 #    --learning_rate_decay_type=fixed \

