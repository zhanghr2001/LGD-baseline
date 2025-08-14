source /root/miniconda3/bin/activate lgd

python evaluate.py \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything/ \
    --iou-eval \
    --vis \
    --use-rgb 1 \
    --use-depth 0 \
    --split test_seen \
    --train-ratio 0 \
    --network logs/250718_0215_train_lgrconvnet3_grasp_anything/epoch_01_step_002000_iou_0.63
