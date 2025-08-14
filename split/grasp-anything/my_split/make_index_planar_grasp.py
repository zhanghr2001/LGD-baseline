import os
import shutil
import csv
import torch
import pickle
from tqdm import tqdm

ds_root = "/baishuanghao/mllm_data/grasp_anything"

csv_path = "planar_grasp_index.csv"
# csv_path = "/baishuanghao/mllm_data/grasp_anything/planar_grasp_index.csv"

image_root = os.path.join(ds_root, "image")
label_root = os.path.join(ds_root, "grasp_label_positive")
description_root = os.path.join(ds_root, "scene_description")


def make(label_root, suffix, csv_path):
    files = os.listdir(label_root)
    files = [f.removesuffix(suffix) for f in files if f.endswith(suffix)]
    files.sort()

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for label_id in tqdm(files):
            scene_id, obj_index = label_id.split("_")
            obj_index = int(obj_index)
            description_path = os.path.join(description_root, scene_id + ".pkl")
            with open(description_path, "rb") as f:
                data = pickle.load(f)
            # label_path = os.path.join(label_root, f"{label_id}.pt")
            # label = torch.load(label_path)
            # if len(label) == 0:
            #     print("empty label")
            #     continue
            if obj_index >= len(data[1]):
                print("cant get obj name")
                continue
            description = data[0]
            obj_name = data[1][obj_index]
            writer.writerow([label_id, obj_name, description])


    print(len(files))
    print(files[0:100])


make(label_root, ".pt", csv_path)