import glob
import os
import re
import random

import pickle
import pandas as pd
import torch
import numpy as np
from PIL import Image

from utils.dataset_processing import grasp, image, mask
from .language_grasp_data import LanguageGraspDatasetBase


class GraspAnythingDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, 'positive_grasp', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, 'prompt', '*.pkl'))
        self.rgb_files = glob.glob(os.path.join(file_path, 'image', '*.jpg'))
        self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))

        if kwargs["seen"]:
            with open(os.path.join('split/grasp-anything/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))
        else:
            with open(os.path.join('split/grasp-anything/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))

        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()
        self.mask_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

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
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        mask_file = self.grasp_files[idx].replace("positive_grasp", "mask").replace(".pt", ".npy")
        mask_img = mask.Mask.from_file(mask_file)
        rgb_file = re.sub(r"_\d{1}\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.replace("positive_grasp", "image")
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
        prompt_file, obj_id = self.grasp_files[idx].replace("positive_grasp", "prompt").split('_')
        prompt_file += '.pkl'
        obj_id = int(obj_id.split('.')[0])

        with open(prompt_file, 'rb') as f:
            x = pickle.load(f)
            prompt, queries = x

        return prompt, queries[obj_id]


class MyGraspAnythingDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, requires_bbox=False, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(MyGraspAnythingDataset, self).__init__(**kwargs)

        self.disable_augment = True
        self.input_size = 416
        self.requires_bbox = requires_bbox

        self.rgb_dir = os.path.join(file_path, "image")
        self.grasp_dir = os.path.join(file_path, "grasp_label_positive")
        self.scene_description_dir = os.path.join(file_path, "scene_description")
        self.mask_dir = os.path.join(file_path, "mask")

        # split_dir = "split/grasp-anything-filter"
        split_dir = "split/grasp-anything-match"
        if kwargs["split"] == "train":
            # TRAIN_SIZE = 1200000
            self.data = pd.read_csv(os.path.join(split_dir, "train.csv"), names=["id", "obj_name", "scene_description"])
            # self.data = self.data.sample(n=TRAIN_SIZE, random_state=42)
        elif kwargs["split"] == "test_seen":
            self.data = pd.read_csv(os.path.join(split_dir, "test_seen.csv"), names=["id", "obj_name", "scene_description"])
        elif kwargs["split"] == "test_unseen":
            self.data = pd.read_csv(os.path.join(split_dir, "test_unseen.csv"), names=["id", "obj_name", "scene_description"])
        else:
            raise ValueError

        self.ids = self.data["id"].tolist()
        self.obj_names = self.data["obj_name"].tolist()
        self.scene_descriptions = self.data["scene_description"].tolist()
        self.grasp_files = list(map(lambda x: os.path.join(self.grasp_dir, f"{x}.pt"), self.ids))
        self.rgb_files = list(map(lambda x: os.path.join(self.rgb_dir, f"{x.split('_')[0]}.jpg"), self.ids))
        # self.mask_files = list(map(lambda x: os.path.join(self.rgb_dir, f"{x.split('_')[0]}.pt"), self.ids))

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

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
        grasp_id = self.ids[idx]
        image_id = grasp_id.split("_")[0]

        bbox_positions = []
        bbox_mask = []
        for obj_id in range(4):
            mask_path = os.path.join(self.mask_dir, f"{image_id}_{obj_id}.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                bbox_pos = mask_to_bbox_position(mask)
                if bbox_pos is None:
                    bbox_positions.append([0,0,0,0])
                    bbox_mask.append(False)
                else:
                    bbox_positions.append(bbox_pos)
                    bbox_mask.append(True)
            else:
                bbox_positions.append([0,0,0,0])
                bbox_mask.append(False)

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
        # GraspRectangles, angle 0 to -pi
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        # Load the prompts
        prompt, query = self.get_prompts(idx)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)
        print("angle max: ", np.max(ang_img))
        print("angle min:", np.min(ang_img))

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


def mask_to_bbox_position(mask):
    """Get bbox position from boolean mask array."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (x_min, y_min, x_max, y_max)  # (left, top, right, bottom)
