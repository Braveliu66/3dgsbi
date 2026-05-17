# DashDeblurGroupGS Embedded Trainer Sources

This directory contains the runtime trainer used by the fine reconstruction worker.

Source boundaries:
- Deblurring-3D-Gaussian-Splatting core Python trainer files are selectively imported from `benhenryL/Deblurring-3D-Gaussian-Splatting` at commit `e63366b8581c0fde2fda0ab1aea99518da2e2f10`.
- DashGaussian scheduling is represented only by `utils/schedule_utils.py`, based on the resolution and Gaussian growth scheduling role from `YouyuChen0207/DashGaussian` at commit `4e3b5606e593b5e58b90a5c3ea8c421bedc308a1`.
- Group Training integration is represented by the `gaussians_grouping` interface and non-destructive cache contract from `Chengbo-Wang/3DGS-with-Group-Training` at commit `eae77c105b3bd2bef8cdc8d70f3b2e6ed4e7f0bf`.
- CUDA rasterizer extensions live under `submodules/` and are built into the worker Python environment by `worker/Dockerfile`.

Speedy-Splat, FastGS renderer replacement, renderer-level pruning, SparseAdam, and Dash antialiasing are intentionally not included.
