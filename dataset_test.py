import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow

from tqdm import tqdm

Dataset = get_dataset("my-grasp-anything")
dataset = Dataset("/baishuanghao/mllm_data/grasp_anything",
                    output_size=224,
                    ds_rotate=0,
                    random_rotate=False,
                    random_zoom=False,
                    include_depth=0,
                    include_rgb=1,
                    split="train",
                    add_file_path=None)
logging.info('Dataset size is {}'.format(dataset.length))


# Dataset = get_dataset("my-real-data")
# dataset = Dataset("/baishuanghao/mllm_data/grasp_anything",
#                     output_size=224,
#                     ds_rotate=0,
#                     random_rotate=False,
#                     random_zoom=False,
#                     include_depth=0,
#                     include_rgb=1,
#                     split="train",
#                     add_file_path=None)
# logging.info('Dataset size is {}'.format(dataset.length))
# print(set(dataset.obj_names))
# x, (pos, cos, sin, width), idx, rot, zoom_factor, prompt, query, bboxes, bbox_positions, bbox_mask = dataset[0]

# print(cos)
# print(sin)
# print(x, (pos, cos, sin, width), idx, rot, zoom_factor, prompt, query, bboxes, bbox_positions, bbox_mask)

for i in tqdm(range(len(dataset))):
    x, (pos, cos, sin, width), idx, rot, zoom_factor, prompt, query, bboxes, bbox_positions, bbox_mask = dataset[i]
#     try:
#         x, (pos, cos, sin, width), idx, rot, zoom_factor, prompt, query, bboxes, bbox_positions, bbox_mask = dataset[i]
#     except ValueError:
#         print(i)
        
