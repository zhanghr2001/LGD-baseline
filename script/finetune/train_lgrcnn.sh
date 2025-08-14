source /root/miniconda3/bin/activate lgd

# 1353 train, 151 eval
python train_network_data.py \
    --network lgrconvnet3 \
    --description finetune_lgrcnn_filter0.4_b4_e1 \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-real-data \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.9 \
    --epochs 1 \
    --batch-size 4 \
    --batches-per-epoch 230 \
    --optim adam \
    --lr 1e-3 \
    --eval-every-n-steps 20 \
    --resume True \
    --checkpoint-path logs/250718_0215_train_lgrconvnet3_grasp_anything/epoch_02_step_006000_iou_0.71
