
expname="test_senseflow"
timestr="2024_09_24_20_14_20"
path_test="xxx/2w_case39_n_2.json"

CUDA_VISIBLE_DEVICES=1 \
python infer.py \
  --model_path ./results/${expname}/${timestr}/ckpt_best.pt \
  --val_txt=${path_test} \
  --batch_size 2000 \
  --n_loops_test 8
