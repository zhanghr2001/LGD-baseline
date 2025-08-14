import pandas as pd

data = pd.read_csv("/baishuanghao/GraspLLM/split/grasp_anything/train.csv", names=["id", "obj_name", "scene_description"])

# 物品名为NaN
clean_data = data[~data['obj_name'].apply(lambda x: isinstance(x, float))]
error_data = data[data['obj_name'].apply(lambda x: isinstance(x, float))]

print(error_data)

clean_data.to_csv("/baishuanghao/GraspLLM/split/grasp_anything/train_filter.csv", index=False, header=False)