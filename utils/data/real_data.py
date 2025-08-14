import glob
import os
import re
import random

import pickle
import pandas as pd
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

from utils.dataset_processing import grasp, image, mask
from utils.dataset_processing.grasp import Grasp, GraspRectangle, GraspRectangles
from .language_grasp_data import LanguageGraspDatasetBase

image_root = "/baishuanghao/mllm_data/grasp_real-world-v1/planar_data_process"
grasp_xml_path = "/baishuanghao/mllm_data/grasp_real-world-v1/grasp_real-world-v1_annotations_with_grasp_labels.xml"
bbox_xml_path = "/baishuanghao/mllm_data/grasp_real-world-v1/grasp_real-world-v1_annotations_with_object_detection.xml"


class MyRealDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """
    def __init__(self, _, ds_rotate=0, requires_bbox=False, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(MyRealDataset, self).__init__(**kwargs)

        self.disable_augment = True
        self.input_size = 720
        self.requires_bbox = requires_bbox

        self.grasp_data = parse_grasp_data(grasp_xml_path)
        self.bbox_data = parse_bbox_data(bbox_xml_path)
        self.image_root = image_root
        self.input_image_size = 720
        self.output_image_size = 224
        self.unseen_objs = real_unseen_objs

        if kwargs["split"] != "train":
            raise ValueError("only train supported")
        self.grasp_data = list(filter(lambda x:x["obj_name"] not in self.unseen_objs, self.grasp_data))
        
        new_grasp_data = []
        for d in self.grasp_data:
            if d["grasp_label"][4] == 0:
                if random.random() < 0.4:
                    new_grasp_data.append(d)
            else:
                new_grasp_data.append(d)

        self.grasp_data = new_grasp_data

        self.image_ids = [x["image_id"] for x in self.grasp_data]
        self.obj_names = [x["obj_name"] for x in self.grasp_data]
        self.rgb_files = list(map(lambda x: os.path.join(self.image_root, f"{x:04d}.jpg"), self.image_ids))
        self.scene_descriptions = [f"a {x}" for x in self.obj_names]

        self.length = len(self.grasp_data)

        if ds_rotate:
            self.grasp_data = self.grasp_data[int(self.length * ds_rotate):] + self.grasp_data[
                                                                                 :int(self.length * ds_rotate)]

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        # Jacquard try
        def _my_grasp_anything_format(grasp: list):
            x, y, w, h, theta = grasp
            # index based on row, column (y,x), and the Grasp-Anything dataset's angles are flipped around an axis.
            return Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr
        gtbbs = self.grasp_data[idx]["grasp_label"]
        gtbbs = [_my_grasp_anything_format(gtbbs)]
        gtbbs = GraspRectangles(gtbbs)
        gtbbs.scale(self.output_image_size / self.input_image_size)
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 720.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError("No depth data.")

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_file = self.rgb_files[idx]
        rgb_img = image.Image.from_file(rgb_file)
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img

    def get_prompts(self, idx):
        return self.scene_descriptions[idx], self.obj_names[idx]

    def get_mask(self, idx):
        raise NotImplementedError

    def get_bbox_positions(self, idx):
        image_id = self.image_ids[idx]
        bbox_positions = self.bbox_data[idx].values()
        bbox_mask = [True] * len(bbox_positions)
        return bbox_positions, bbox_mask

    def get_bbox_images(self, idx, bbox_positions, bbox_mask):
        n = len(bbox_positions)
        rgb_file = self.rgb_files[idx]
        rgb_img = image.Image.from_file(rgb_file)
        bboxes = torch.zeros([len(bbox_positions), 3, self.output_size, self.output_size])
        for i in range(n):
            if bbox_mask[i]:
                x_min, y_min, x_max, y_max = bbox_positions[i]
                if x_max - x_min < 5 or y_max - y_min < 5:
                    # bbox too small
                    bbox_mask[i] = False
                    continue
                bbox = rgb_img.cropped((y_min, x_min), (y_max, x_max), resize=(self.output_size, self.output_size))
                bbox.normalise()
                bbox.img = bbox.img.transpose((2, 0, 1))
                bboxes[i] = torch.from_numpy(bbox.img)

        return bboxes

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        # Load the prompts
        prompt, query = self.get_prompts(idx)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                    rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        if self.requires_bbox:
            bbox_positions, bbox_mask = self.get_bbox_positions(idx)
            bboxes = self.get_bbox_images(idx, bbox_positions, bbox_mask)

            bbox_positions = torch.tensor(bbox_positions) / self.input_size
            bbox_mask = torch.tensor(bbox_mask)
        else:
            bboxes = bbox_positions = bbox_mask = 0

        return x, (pos, cos, sin, width), idx, rot, zoom_factor, prompt, query, bboxes, bbox_positions, bbox_mask


def parse_grasp_data(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []
    for image in root.findall("image"):
        image_id = int(image.get("id"))

        for obj in image.findall("*"):
            if obj.tag in ["box", "polygon"]:
                obj_name = obj.get("label")
                grasp_label = obj.get("grasp_label")
                grasp_label = [float(i) for i in grasp_label.split()]
                data.append(
                    {
                        "image_id": image_id,
                        "obj_name": obj_name,
                        "grasp_label": grasp_label,
                    }
                )
    return data

def parse_bbox_data(xml_path):
    """return [{obj_name: bbox}], bbox in cv format"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []
    for image in root.findall("image"):
        image_id = int(image.get("id"))

        image_data = {}

        for box in image.findall("box"):
            obj_name = box.get("label")

            bbox = [
                float(box.get("xtl")),  # x_min (xtl)
                float(box.get("ytl")),  # y_min (ytl)
                float(box.get("xbr")),  # x_max (xbr)
                float(box.get("ybr")),  # y_max (ybr)
            ]

            image_data[obj_name] = bbox
        data.append(image_data)

    return data

real_unseen_objs = [
    "spirit",
    "banana",
    "mango",
    "corn",
    "potato",
    "red potato",
    "green pepper",
    "basketball",
    "tennis ball",
    "green cube",
    "yellow cube",
    "white bowl",
    "blue bowl",
    "green cup",
    "red cup",
]