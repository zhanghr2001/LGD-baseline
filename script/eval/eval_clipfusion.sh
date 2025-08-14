source /root/miniconda3/bin/activate lgd

python evaluate_clipfusion.py \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything/ \
    --iou-eval \
    --use-rgb 1 \
    --use-depth 0 \
    --split test_unseen \
    --train-ratio 0 \
    --network logs/250718_0316_train_clipfusion_grasp_anything_match_data2/epoch_00_step_022000_iou_0.41
