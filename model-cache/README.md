# Model Cache

Put third-party model weights here. Large weights are intentionally ignored by Git.

Expected first-batch paths:

- `litevggt/te_dict.pt`
- `lingbot-map/lingbot-map-long.pt`
- `roma/roma_indoor.pth`
- `roma/dinov2_vitl14_pretrain.pth`

`roma/*` is used by EDGS correspondence initialization. Keep these files in the
mounted cache so the worker never downloads weights at runtime.
