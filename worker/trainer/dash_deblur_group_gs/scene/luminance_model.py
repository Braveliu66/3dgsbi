import torch
from torch import nn


class PerImageExposureModel(nn.Module):
    def __init__(self, num_images):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(num_images, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_images, 1, 1, 1))

    def forward(self, image, image_id):
        if not torch.is_tensor(image_id):
            image_id = torch.tensor(image_id, device=image.device, dtype=torch.long)
        else:
            image_id = image_id.to(device=image.device, dtype=torch.long)
        image_id = image_id.view(-1)[0]

        gain = torch.exp(torch.clamp(self.log_gain[image_id], min=-0.30, max=0.30))
        bias = torch.clamp(self.bias[image_id], min=-0.05, max=0.05)
        return torch.clamp(image * gain + bias, 0.0, 1.0)

    def regularization_loss(self, lambda_gain=5e-3, lambda_bias=1e-2):
        return (
            lambda_gain * (self.log_gain ** 2).mean()
            + lambda_bias * (self.bias ** 2).mean()
        )
