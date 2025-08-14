import os
import pickle

import numpy as np
from PIL import Image, ImageDraw

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow


# with open("/baishuanghao/LGD/split/grasp-anything/unseen.obj", "rb") as f:
#     d = pickle.load(f)
# print(len(d))

# Dataset = get_dataset("my-grasp-anything")

# ds = Dataset(
#     "/baishuanghao/mllm_data/grasp_anything",
#     output_size=224,
#     ds_rotate=0,
#     random_rotate=1,
#     random_zoom=1,
#     include_depth=0,
#     include_rgb=1,
#     seen=1,
#     add_file_path="",
# )
# for i in range(10):
#     print(ds[i])

def mask_to_bbox(mask_array):
    """从二值mask numpy数组中提取边界框坐标"""
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return (x_min, y_min, x_max, y_max)  # (left, top, right, bottom)

def draw_bbox_on_image(image, bbox, output_path, bbox_color="red", bbox_width=3):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=bbox_color, width=bbox_width)
    
    # 保存结果
    image.save(output_path)
    print(f"结果已保存到: {output_path}")

grasp_id = "00188d7df27191a3e897e251575f2ebff92bc58a65a62fe70dac110deed2960f_1"
image_dir = "/baishuanghao/mllm_data/grasp_anything/image"
mask_dir = "/baishuanghao/mllm_data/grasp_anything/mask"
save_dir = "./vis/test"
os.makedirs(save_dir, exist_ok=True)

image_id = grasp_id.split("_")[0]
image_path = os.path.join(image_dir, f"{image_id}.jpg")
mask_path = os.path.join(mask_dir, f"{grasp_id}.npy")

image = Image.open(image_path)
mask = np.load(mask_path)

bbox = mask_to_bbox(mask)
save_path = os.path.join(save_dir, "t.jpg")
print(bbox)


draw_bbox_on_image(image, bbox, save_path)



