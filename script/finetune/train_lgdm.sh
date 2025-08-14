source /root/miniconda3/bin/activate lgd

python train_network_diffusion_data.py \
    --network lgdm \
    --description train_lgdm_grasp_anything_match_data \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-real-data \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.999 \
    --epochs 3 \
    --batch-size 8 \
    --batches-per-epoch 22500 \
    --optim adam \
    --lr 1e-3 \
    --eval-every-n-steps 5000 \
    --resume True \
    --checkpoint-path logs/250718_0215_train_lgrconvnet3_grasp_anything/epoch_02_step_006000_iou_0.71
