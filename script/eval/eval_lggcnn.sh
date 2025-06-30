source /root/miniconda3/bin/activate lgd

python evaluate.py \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything/ \
    --iou-eval \
    --use-rgb 1 \
    --use-depth 0 \
    --split test_unseen \
    --train-ratio 0 \
    --network logs/250619_1626_train_lggcnn_grasp_anything_large_data/epoch_00_step_090000_iou_0.45
