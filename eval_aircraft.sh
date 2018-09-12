DATASET_DIR=../data/dataset/
CHECKPOINT_FILE=../data/MSCNN/log_HBPNet/model.ckpt-60000  
python eval_image_classifier.py \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=aircraft \
    --dataset_split_name=validation \
    --model_name=vgg_16 \
    --batch_size=16
