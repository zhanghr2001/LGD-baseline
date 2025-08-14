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
    --network logs/250719_2003_train_lggcnn_match_data_16batch/epoch_02_step_006000_iou_0.55
