#!/bin/bash

folder_path="/baishuanghao/LGD/logs/250506_0317_train_lgrconvnet3_grasp_anything"  # Replace '/path/to/your/folder' with the actual folder path

if [ ! -d "$folder_path" ]; then
    echo "Folder $folder_path not found."
    exit 1
fi

pattern="epoch_*"
for file in "$folder_path"/$pattern; do
    if [ -f "$file" ]; then
        echo "Running command with file: $file"
        python evaluate.py --dataset my-grasp-anything --dataset-path /baishuanghao/mllm_data/grasp_anything/ --iou-eval --use-depth 0 --seen 0 --train-ratio 0.01 --network "$file"  # Execute the command with the file as a parameter
    fi
done

echo "All files processed."
