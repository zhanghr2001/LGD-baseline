source /root/miniconda3/bin/activate lgd

python train_network_grasp_det_seg.py \
    --description train_det_seg_grasp_anything \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --seen 1 \
    --train-ratio 0.9 \
    --epochs 50 \
    --batch-size 8 \
    --batches-per-epoch 1685