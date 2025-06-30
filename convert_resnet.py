import torch
from collections import OrderedDict

def convert_resnet_weights(original_weights):
    # 创建一个新的有序字典来存储转换后的权重
    new_weights = OrderedDict()
    
    # 转换第一层 (mod1)
    new_weights['mod1.conv1.weight'] = original_weights['conv1.weight']
    new_weights['mod1.bn1.weight'] = original_weights['bn1.weight']
    new_weights['mod1.bn1.bias'] = original_weights['bn1.bias']
    new_weights['mod1.bn1.running_mean'] = original_weights['bn1.running_mean']
    new_weights['mod1.bn1.running_var'] = original_weights['bn1.running_var']
    
    # 定义层名映射关系
    layer_mapping = {
        'layer1': 'mod2',
        'layer2': 'mod3',
        'layer3': 'mod4',
        'layer4': 'mod5'
    }
    
    block_counts = {
        'layer1': 3,
        'layer2': 4,
        'layer3': 23,  # 根据你的错误信息，layer3有23个block
        'layer4': 3
    }
    
    # 遍历每一层
    for orig_layer, new_mod in layer_mapping.items():
        num_blocks = block_counts[orig_layer]
        
        for block_idx in range(num_blocks):
            # 处理第一个block可能有downsample的情况
            has_downsample = (block_idx == 0) and (orig_layer in ['layer1', 'layer2', 'layer3', 'layer4'])
            
            # 处理每个block中的conv和bn层
            for conv_num in [1, 2, 3]:
                # conv权重
                orig_conv_key = f"{orig_layer}.{block_idx}.conv{conv_num}.weight"
                new_conv_key = f"{new_mod}.block{block_idx+1}.convs.conv{conv_num}.weight"
                if orig_conv_key in original_weights:
                    new_weights[new_conv_key] = original_weights[orig_conv_key]
                
                # bn参数
                for bn_param in ['weight', 'bias', 'running_mean', 'running_var']:
                    orig_bn_key = f"{orig_layer}.{block_idx}.bn{conv_num}.{bn_param}"
                    new_bn_key = f"{new_mod}.block{block_idx+1}.convs.bn{conv_num}.{bn_param}"
                    if orig_bn_key in original_weights:
                        new_weights[new_bn_key] = original_weights[orig_bn_key]
            
            # 处理downsample (如果有)
            if has_downsample:
                # 投影卷积
                orig_down_conv_key = f"{orig_layer}.{block_idx}.downsample.0.weight"
                new_proj_conv_key = f"{new_mod}.block{block_idx+1}.proj_conv.weight"
                if orig_down_conv_key in original_weights:
                    new_weights[new_proj_conv_key] = original_weights[orig_down_conv_key]
                
                # 投影BN
                for bn_param in ['weight', 'bias', 'running_mean', 'running_var']:
                    orig_down_bn_key = f"{orig_layer}.{block_idx}.downsample.1.{bn_param}"
                    new_proj_bn_key = f"{new_mod}.block{block_idx+1}.proj_bn.{bn_param}"
                    if orig_down_bn_key in original_weights:
                        new_weights[new_proj_bn_key] = original_weights[orig_down_bn_key]
    
    return new_weights

# 使用示例
if __name__ == "__main__":
    # 加载原始权重
    original_weights = torch.load("/baishuanghao/LGD/trained-models/grasp_det_seg/resnet101-5d3b4d8f.pth")
    
    # 转换权重
    converted_weights = convert_resnet_weights(original_weights)
    
    # 保存转换后的权重
    torch.save(converted_weights, "/baishuanghao/LGD/trained-models/grasp_det_seg/resnet101")
    print("权重转换完成并已保存")