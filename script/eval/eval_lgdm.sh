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
    --network logs/250729_1522_train_lgdm_resume_e2/epoch_00_step_005000_iou_0.53

python evaluate_diffusion.py \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything/ \
    --iou-eval \
    --use-rgb 1 \
    --use-depth 0 \
    --split test_unseen \
    --train-ratio 0.9 \
    --ds-shuffle \
    --network logs/250729_1522_train_lgdm_resume_e2/epoch_00_step_005000_iou_0.53
