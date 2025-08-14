source /root/miniconda3/bin/activate lgd

python train_network_data.py \
    --network lggcnn \
    --description train_lggcnn_match_data_16batch \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-real-data \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.99 \
    --epochs 3 \
    --batch-size 16 \
    --batches-per-epoch 11250 \
    --optim adam \
    --lr 1e-3 \
    --eval-every-n-steps 1000 \
    --resume True \
    --checkpoint-path logs/250718_0215_train_lgrconvnet3_grasp_anything/epoch_02_step_006000_iou_0.71
