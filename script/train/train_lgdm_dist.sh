source /root/miniconda3/bin/activate lgd

# torchrun --nproc_per_node=2 train_network_diffusion_dist.py \
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
#     --batches-per-epoch 1685

torchrun --nproc_per_node=2 train_network_diffusion_dist.py \
    --network lgdm \
    --description train_lgdm_grasp_anything \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --seen 1 \
    --train-ratio 0.9 \
    --epochs 1000 \
    --batch-size 80 \
    --batches-per-epoch 180 \
    --lr 1e-2