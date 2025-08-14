source /root/miniconda3/bin/activate lgd

# python train_network_diffusion.py \
#     --network lgdm \
#     --description train_lgdm_grasp_anything \
#     --use-rgb 1 \
#     --use-depth 0 \
#     --dataset my-grasp-anything \
#     --dataset-path /baishuanghao/mllm_data/grasp_anything \
#     --seen 1 \
#     --train-ratio 0.9 \
#     --epochs 1000 \
#     --batch-size 8 \
#     --batches-per-epoch 1000 \

# python train_network_diffusion_data.py \
#     --network lgdm \
#     --description train_lgdm_grasp_anything_large_data \
#     --use-rgb 1 \
#     --use-depth 0 \
#     --dataset my-grasp-anything \
#     --dataset-path /baishuanghao/mllm_data/grasp_anything \
#     --split train \
#     --train-ratio 0.9999 \
#     --epochs 3 \
#     --batch-size 8 \
#     --batches-per-epoch 145000 \
#     --optim adam \
#     --lr 1e-3 \
#     --eval-every-n-steps 40000 \
#     --resume True \
#     --checkpoint-path /baishuanghao/LGD/logs/250619_1806_train_lgdm_grasp_anything_large_data/epoch_00_step_130000_iou_0.38

python train_network_diffusion_data.py \
    --network lgdm \
    --description train_lgdm_grasp_anything_match_data \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.999 \
    --epochs 3 \
    --batch-size 8 \
    --batches-per-epoch 22500 \
    --optim adam \
    --lr 1e-3 \
    --eval-every-n-steps 5000 \
