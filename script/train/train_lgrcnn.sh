source /root/miniconda3/bin/activate lgd

# python train_network.py \
#     --network lgrconvnet3 \
#     --description train_lgrconvnet3_grasp_anything \
#     --use-rgb 1 \
#     --use-depth 0 \
#     --dataset my-grasp-anything \
#     --dataset-path /baishuanghao/mllm_data/grasp_anything \
#     --seen 1 \
#     --train-ratio 0.9 \
#     --epochs 50 \
#     --batch-size 80 \
#     --batches-per-epoch 15000 \
#     --optim sgd \
#     --lr 1e-3

# python train_network_data.py \
#     --network lgrconvnet3 \
#     --description train_lgrconvnet3_grasp_anything \
#     --use-rgb 1 \
#     --use-depth 0 \
#     --dataset my-grasp-anything \
#     --dataset-path /baishuanghao/mllm_data/grasp_anything \
#     --split train \
#     --train-ratio 0.999 \
#     --epochs 2 \
#     --batch-size 8 \
#     --batches-per-epoch 145000 \
#     --optim adam \
#     --lr 1e-3 \
#     --eval-every-n-steps 10000 \
#     --resume True \
#     --checkpoint-path /baishuanghao/LGD/logs/250619_1625_train_lgrconvnet3_grasp_anything/epoch_00_step_130000_iou_0.60

python train_network_data.py \
    --network lgrconvnet3 \
    --description train_lgrconvnet3_grasp_anything \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.99 \
    --epochs 3 \
    --batch-size 8 \
    --batches-per-epoch 22500 \
    --optim adam \
    --lr 1e-3 \
    --eval-every-n-steps 2000 \
