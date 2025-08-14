source /root/miniconda3/bin/activate lgd

python train_network_clipfusion_data.py \
    --network clipfusion \
    --description train_clipfusion_grasp_anything_match_data2 \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-real-data \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.99 \
    --epochs 3 \
    --batch-size 8 \
    --batches-per-epoch 22500 \
    --optim adam \
    --lr 3e-4 \
    --eval-every-n-steps 2000 \
    --resume True \
    --checkpoint-path logs/250718_0215_train_lgrconvnet3_grasp_anything/epoch_02_step_006000_iou_0.71
 