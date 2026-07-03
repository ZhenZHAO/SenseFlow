# - - - - -  single GPU 

rseed=2024
ROOT=.
# configname="GPSv1_1"
configname=$1
gpuindex=$2

# CUDA_VISIBLE_DEVICES=0 \
CUDA_VISIBLE_DEVICES=${gpuindex} \
python $ROOT/main.py \
    --config configs/${configname}.yaml \
    --seed=${rseed} 

# sh train_single.sh GPSv1_1 0