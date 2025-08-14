import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # 进度条（可选，安装：pip install tqdm）

def compute_pixel_count(args):
    """计算单个mask的像素数量（供多进程调用）"""
    mask_id, mask_dir = args
    mask_path = os.path.join(mask_dir, f"{mask_id}.npy")
    try:
        mask = np.load(mask_path)
        return (mask_id, np.count_nonzero(mask))
    except Exception as e:
        print(f"Error processing {mask_id}: {str(e)}")
        return (mask_id, 0)  # 出错时返回0，后续会被过滤

def filter_masks_parallel(
    input_csv, 
    output_csv, 
    mask_dir, 
    threshold=100,
    num_workers=None
):
    """
    多进程统计mask像素数量并过滤数据
    
    参数:
        input_csv (str): 输入CSV路径
        id_column (str): ID列名
        mask_dir (str): 存放.npy文件的目录
        output_csv (str): 输出CSV路径
        threshold (int): 像素数量阈值
        num_workers (int): 进程数（默认使用全部CPU核心）
    """
    # 读取CSV
    df = pd.read_csv(input_csv, names=["id", "obj_name", "scene_description"])
    mask_ids = df["id"].tolist()

    # 准备多进程参数
    tasks = [(mask_id, mask_dir) for mask_id in mask_ids]
    num_workers = num_workers or cpu_count()

    # 多进程计算像素数量
    print(f"启动 {num_workers} 个进程统计像素数量...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(compute_pixel_count, tasks),
            total=len(tasks),
            desc="Processing masks"
        ))

    # 将结果转为字典 {mask_id: pixel_count}
    pixel_counts = dict(results)

    # 过滤数据：保留像素数量 >= threshold 的行
    df['pixel_count'] = df["id"].map(pixel_counts)
    filtered_df = df[df['pixel_count'] >= threshold].copy()

    # 保存结果（可选：是否保留pixel_count列）
    filtered_df.drop(columns=['pixel_count'], inplace=True)
    filtered_df.to_csv(output_csv, index=False)

    print(
        f"完成！原始数据 {len(df)} 条，"
        f"保留 {len(filtered_df)} 条 (阈值={threshold})"
    )

if __name__ == "__main__":
    # 示例参数（根据需求修改）
    input_csv = "/baishuanghao/LGD/split/grasp-anything/my_split/seen.csv"
    mask_dir = "/baishuanghao/mllm_data/grasp_anything/mask"
    output_csv = "/baishuanghao/LGD/split/grasp-anything/my_split/seen_filter.csv"
    threshold = 200                  # 像素阈值
    num_workers = 6                  # 进程数（None=自动选择）

    # 运行
    filter_masks_parallel(input_csv, output_csv, mask_dir, threshold, num_workers)
