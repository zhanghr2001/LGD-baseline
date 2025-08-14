source /root/miniconda3/bin/activate lgd

# python train_network.py \
#     --network lggcnn \
#     --description train_lggcnn_grasp_anything \
#     --use-rgb 1 \
#     --use-depth 0 \
#     --dataset my-grasp-anything \
#     --dataset-path /baishuanghao/mllm_data/grasp_anything \
#     --seen 1 \
#     --train-ratio 0.9 \
#     --epochs 50 \
#     --batch-size 8 \
#     --batches-per-epoch 1685

# python train_network_data.py \
#     --network lggcnn \
#     --description train_lggcnn_grasp_anything_large_data \
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
#     --checkpoint-path /baishuanghao/LGD/logs/250619_1626_train_lggcnn_grasp_anything_large_data/epoch_00_step_090000_iou_0.45

python train_network_data.py \
    --network lggcnn \
    --description train_lggcnn_match_data_16batch_5e-4 \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.99 \
    --epochs 3 \
    --batch-size 16 \
    --batches-per-epoch 11250 \
    --optim adam \
    --lr 5e-4 \
    --eval-every-n-steps 1000 \
