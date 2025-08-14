source /root/miniconda3/bin/activate lgd

python train_network_diffusion_data.py \
    --network lgdm \
    --description train_lgdm_resume_e2 \
    --use-rgb 1 \
    --use-depth 0 \
    --dataset my-grasp-anything \
    --dataset-path /baishuanghao/mllm_data/grasp_anything \
    --split train \
    --train-ratio 0.9999 \
    --epochs 1 \
    --batch-size 8 \
    --batches-per-epoch 22500 \
    --optim adam \
    --lr 1e-3 \
    --eval-every-n-steps 5000 \
    --resume True \
    --checkpoint-path logs/250719_2003_train_lgdm_grasp_anything_match_data/epoch_01_step_015000_iou_0.43