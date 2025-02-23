import os
import torch

# 直接定义变量
LMD = 0.8  # 设置 lmd 的值
GAMMA = 0.8  # 设置 gamma
STEP_SIZE = 5  # 设置 step_size
LR_CNN = 0.001  # 设置卷积层的 lr
LR_FC = 0.01  # 设置全连接层的 lr
PATH = "Emotion6"  # 选择训练需要调用的数据集
LOG_FILE = 'test.log'  # 保存输出的 log 文件
CUDA_NUM = '0'  # 选择需要调用的 cuda

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_NUM

# 根据数据集设置类别数
if PATH == "Emotion6":
    NUM_CLASSES = 7
else:
    NUM_CLASSES = 8

# 其他常量
BATCH_SIZE = 32
EPOCH = 10
USE_GPU = torch.cuda.is_available()
SAVE_FILE = r'model_save/parameter.pkl'

