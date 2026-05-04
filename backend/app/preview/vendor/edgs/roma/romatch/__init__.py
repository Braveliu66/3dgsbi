# 本文件为 3DGS 预览系统内置算法代码，裁剪自对应上游仓库的关键运行路径；保留原许可证，避免运行时依赖 GitHub 克隆目录。
import os
from .models import roma_outdoor, tiny_roma_v1_outdoor, roma_indoor

DEBUG_MODE = False
RANK = int(os.environ.get('RANK', default = 0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1