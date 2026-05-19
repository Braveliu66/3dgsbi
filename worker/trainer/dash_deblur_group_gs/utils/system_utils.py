#
# 版权所有 (C) 2023, Inria
# GRAPHDECO 研究组, https://team.inria.fr/graphdeco
# 初始化输出目录
#
# 本软件仅可在 LICENSE.md 文件条款下用于
# 非商业、研究和评估用途。
#
# 咨询请联系：george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # 创建目录，等价于命令行中的 mkdir -p。
    try:
        makedirs(folder_path)
    except OSError as exc: # 初始化输出目录 Python 2.5 ????
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
