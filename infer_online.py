import os
from math import degrees

from PIL import Image
import numpy as np
import torch
import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import grasp, image, mask, evaluation
from utils.visualisation.plot import save_results

raw_shape = [1280, 720]
input_size = 720
output_size = 224
prompt_template = "a {}"
data_root = "real_data"
# checkpoint_path = "logs/250807_2357_finetune_lgrcnn_filter0.4_b4_e1/epoch_00_step_000120_iou_0.09"
checkpoint_path = "logs/250718_0215_train_lgrconvnet3_grasp_anything/epoch_02_step_006000_iou_0.71"

model_name = ["lgrcnn", "lggcnn", "clipfusion", "lgd"]
model = torch.load(checkpoint_path)
model.eval()

while True:
    s = input("continue? (y/n): ")
    if s != "y":
        exit()
    
    image_id = int(input("image id: "))
    obj_name = input("obj name: ")
    prompt = prompt_template.format(obj_name)
    image_path = os.path.join(data_root, f"{image_id:04d}.jpg")
    rgb_img = image.Image.from_file(image_path)
    height, width, channel = rgb_img.img.shape
    if width == raw_shape[0] and height == raw_shape[1]:
        # crop center 720*720 region
        rgb_img.crop([0, 280],[720, 1000])
    elif width == 720 and height == 720:
        pass
    else:
        raise ValueError("not supported image shape")

    rgb_img.resize((output_size, output_size))
    rgb_img.normalise()
    rgb_img.img = rgb_img.img.transpose((2, 0, 1))
    img = torch.from_numpy(rgb_img.img)
    img = img.unsqueeze(0).to("cuda")

    pos_output, cos_output, sin_output, width_output = model(img, prompt, obj_name)
    pos_output, cos_output, sin_output, width_output = pos_output.detach(), cos_output.detach(), sin_output.detach(), width_output.detach()
    q_img, ang_img, width_img = post_process_output(pos_output, cos_output, sin_output, width_output)
    gs = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=5)

    scale = input_size / output_size
    if len(gs) == 0:
        print("no result")
    for g in gs:
        u, v, theta = int(g.center[1] * scale), int(g.center[0] * scale), g.angle
        print(repr([u, v, theta]))
