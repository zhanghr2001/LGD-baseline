source /root/miniconda3/bin/activate lgd

python evaluate.py \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything/ \
    --iou-eval \
    --use-rgb 1 \
    --use-depth 0 \
    --split test_unseen \
    --train-ratio 0 \
    --network logs/250619_1625_train_lgrconvnet3_grasp_anything/epoch_00_step_130000_iou_0.60
