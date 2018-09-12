 python train_image_classifier.py \
    --max_number_of_steps=30000 \
    --train_dir=./train_log \
    --dataset_name=fgvc \
    --dataset_split_name=train \
    --dataset_dir=datasets \
    --model_name=vgg16 \
    --batch_size=16 \
    --checkpoint_path=checkpoint/vgg_16.ckpt \
    --checkpoint_exclude_scopes=vgg_16/fc6,vgg_16/fc7,vgg_16/fc8
