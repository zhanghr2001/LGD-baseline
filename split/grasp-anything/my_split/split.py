import random
import pickle
import pandas as pd
from collections import Counter
from typing import List, Union

def split_base_new_classes(index_csv_path, base_ratio=0.7):
    """
    读取索引CSV文件，按物品出现频率划分base和new类别
    :param index_csv_path: 索引CSV文件路径（格式：id, item_name）
    :param base_ratio: base类别的比例（默认70%）
    :return: (base_ids, new_ids), (base_items, new_items)
    """
    # 读取索引文件
    df = pd.read_csv(index_csv_path, header=None, names=['id', 'item_name', "scene_description"])
    
    # 统计每个物品的出现频率
    item_counts = Counter(df['item_name'])
    
    most_common = item_counts.most_common()
    # 按出现频率排序（从高到低）
    candidate_items = [item for item, count in most_common if count > 40]
    least_items = [item for item, count in most_common if count <= 40]
    # 划分base和new类别
    split_idx = int(len(candidate_items) * base_ratio)
    base_items = candidate_items[:split_idx] + least_items
    new_items = candidate_items[split_idx:]
    
    # 获取base和new对应的ID集合
    # base_ids = set(df[df['item_name'].isin(base_items)]['id'])
    # new_ids = set(df[df['item_name'].isin(new_items)]['id'])
    base_df = df[df['item_name'].isin(base_items)]
    new_df = df[df['item_name'].isin(new_items)]

    base_df.to_csv("/baishuanghao/GraspLLM/split/grasp_anything/train.csv", index=False, header=False)
    new_df.to_csv("/baishuanghao/GraspLLM/split/grasp_anything/test.csv", index=False, header=False)
    
    # return (base_ids, new_ids), (base_items, new_items)
    return

def check_ids_in_base_or_new(
    test_ids: List[Union[int, str]],
    base_ids: set,
    new_ids: set
) -> dict:
    """
    检查给定的ID列表有多少比例属于base或new
    :param test_ids: 待检查的ID列表（可以是int或str）
    :param base_ids: base类别的ID集合
    :param new_ids: new类别的ID集合
    :return: dict {
        'base_count': base类别的ID数量,
        'new_count': new类别的ID数量,
        'base_ratio': base类别的比例（0~1）,
        'new_ratio': new类别的比例（0~1）,
        'total': 总ID数量,
        'status': 'base'/'new'/'mixed'（如果全部属于base或new，否则mixed）
    }
    """
    test_ids = set(test_ids)  # 转换为集合去重（可选）
    total = len(test_ids)
    
    base_count = len(test_ids & base_ids)
    new_count = len(test_ids & new_ids)
    
    base_ratio = base_count / total if total > 0 else 0
    new_ratio = new_count / total if total > 0 else 0
    
    if base_count == total:
        status = 'base'
    elif new_count == total:
        status = 'new'
    else:
        status = 'mixed'
    
    return {
        'base_count': base_count,
        'new_count': new_count,
        'base_ratio': base_ratio,
        'new_ratio': new_ratio,
        'total': total,
        'status': status
    }


def split_csv(input_file, output_selected, output_remaining, 
                       select_num, random_state=None):
    """
    随机选取指定数量的行并分割CSV
    
    参数:
        input_file: 输入CSV文件路径
        output_selected: 选中的行保存路径
        output_remaining: 剩余的行保存路径
        select_num: 要选取的固定行数
        random_state: 随机种子(保证可重复性)
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    if len(df) < select_num:
        raise ValueError(f"文件只有 {len(df)} 行，不足要求的 {select_num} 行")
    
    # 随机选择指定数量的行
    selected = df.sample(n=select_num, random_state=random_state)
    
    # 剩余的行
    remaining = df.drop(selected.index)
    
    # 保存结果
    selected.to_csv(output_selected, index=False, header=False)
    remaining.to_csv(output_remaining, index=False, header=False)

# 示例用法
if __name__ == "__main__":
    # 1. 划分base和new类别
    # index_csv = "/baishuanghao/mllm_data/grasp_anything/planar_grasp_index.csv"  # 替换为你的索引文件路径
    # split_base_new_classes(index_csv)

    split_csv("/baishuanghao/GraspLLM/split/grasp_anything/seen.csv", "/baishuanghao/GraspLLM/split/grasp_anything/test_seen.csv", "/baishuanghao/GraspLLM/split/grasp_anything/train.csv", 10000)

    # (base_ids, new_ids), (base_items, new_items) = split_base_new_classes(index_csv)
    
    # print(f"Base类别数量: {len(base_items)}")
    # print(f"New类别数量: {len(new_items)}")
    # print(f"Base grasp数量: {len(base_ids)}")
    # print(f"New grasp数量: {len(new_ids)}")
    # print(f"示例Base类别: {list(base_items)[:5]}...")
    # print(f"示例New类别: {list(new_items)[:5]}...")
    
    # # 2. 检查另一个CSV文件的ID属于哪一类
    # with open("/baishuanghao/GraspLLM/split/grasp_anything/seen.obj", "rb") as f:
    #     seen_ids = pickle.load(f)
    # with open("/baishuanghao/GraspLLM/split/grasp_anything/unseen.obj", "rb") as f:
    #     unseen_ids = pickle.load(f)

    # result = check_ids_in_base_or_new(seen_ids, base_ids, new_ids)
    # print("seen result: ", result)
    # result = check_ids_in_base_or_new(unseen_ids, base_ids, new_ids)
    # print("unseen result: ", result)
