source /root/miniconda3/bin/activate lgd

python evaluate_diffusion.py \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything/ \
    --iou-eval \
    --use-rgb 1 \
    --use-depth 0 \
    --split test_seen \
    --train-ratio 0.9 \
    --ds-shuffle \
    --network logs/250619_1806_train_lgdm_grasp_anything_large_data/epoch_00_step_130000_iou_0.38

python evaluate_diffusion.py \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything/ \
    --iou-eval \
    --use-rgb 1 \
    --use-depth 0 \
    --split test_unseen \
    --train-ratio 0.9 \
    --ds-shuffle \
    --network logs/250619_1806_train_lgdm_grasp_anything_large_data/epoch_00_step_130000_iou_0.38
