// 本文件为 3DGS 预览系统内置 C++ 扩展代码，来自上游关键运行依赖；保留原许可证，随 worker 镜像编译。
#include <torch/extension.h>
#include "ssim.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
}
