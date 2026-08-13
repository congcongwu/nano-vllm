import os  # 操作系统接口，用于拼接模型权重文件的路径
from glob import glob  # 通配符匹配，用来查找目录下所有 .safetensors 文件
import torch  # PyTorch 关量库
from torch import nn  # PyTorch 神经网络模块
from safetensors import safe_open  # 安全加载 safetensors 格式的权重文件


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):  # 默认权重加载函数：直接把加载到的权重拷贝到参数
    param.data.copy_(loaded_weight)  # 原地拷贝权重数据到 param


def load_model(model: nn.Module, path: str):  # 从磁盘把权重加载进模型，处理打包（packed）参数的重映射
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})  # 获取模型的“参数打包映射表”（如 qkv 合并、gate_up 合并），可能为空字典
    for file in glob(os.path.join(path, "*.safetensors")):  # 遍历模型目录下所有的 safetensors 权重分片文件
        with safe_open(file, "pt", "cpu") as f:  # 以 PyTorch 格式、CPU 设备打开当前分片文件
            for weight_name in f.keys():  # 遍历该分片内的每一组权重（key 是权重的全名）
                for k in packed_modules_mapping:  # 遍历所有打包映射键（如 "q_proj"/"k_proj"/"v_proj"）
                    if k in weight_name:  # 若当前权重名包含该打包键前缀
                        v, shard_id = packed_modules_mapping[k]  # 取出它映射到的目标模块名和分片标识
                        param_name = weight_name.replace(k, v)  # 把权重名中的打包键替换为目标名，得到真实参数名
                        param = model.get_parameter(param_name)  # 通过参数名从模型中取出对应的 Parameter
                        weight_loader = getattr(param, "weight_loader")  # 取得参数自带的权重加载器（用于分片切分）
                        weight_loader(param, f.get_tensor(weight_name), shard_id)  # 用分片加载器把原权重切分后写入
                        break  # 命中打包映射后跳出内部循环
                else:  # 若循环结束都没遇到打包键（普通权重）
                    param = model.get_parameter(weight_name)  # 直接按原名取参数
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 读取参数的加载器，若没有则用默认拷贝加载器
                    weight_loader(param, f.get_tensor(weight_name))  # 加载权重（普通权重直接整体拷贝）
