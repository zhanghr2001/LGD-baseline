source /root/miniconda3/bin/activate lgd

# python train_network.py \
#     --network clipfusion \
#     --description train_clipfusion_grasp_anything \
#     --use-rgb 1 \
#     --use-depth 0 \
#     --dataset my-grasp-anything \
#     --dataset-path /baishuanghao/mllm_data/grasp_anything \
#     --seen 1 \
#     --train-ratio 0.9 \
#     --epochs 50 \
#     --batch-size 8 \
#     --batches-per-epoch 1685

python train_network_clipfusion_data.py \
    --network clipfusion \
    --description train_clipfusion_grasp_anything_large_data \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.999 \
    --epochs 2 \
    --batch-size 8 \
    --batches-per-epoch 145000 \
    --optim adam \
    --lr 2e-3 \
    --eval-every-n-steps 20000 \
    --resume True \
    --checkpoint_path /baishuanghao/LGD/logs/250619_1740_train_clipfusion_grasp_anything_large_data/epoch_00_step_130000_iou_0.13
