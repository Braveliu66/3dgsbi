import torch


SOURCE_SFM = 0
SOURCE_EAP = 1
SOURCE_NEXUS = 2
SOURCE_CLONE = 3
SOURCE_SPLIT = 4


class DeblurAwareGDAGS:
    def __init__(self, num_gaussians, device="cuda", source_type=SOURCE_SFM):
        self.blur_grad_dir_sum = torch.zeros(num_gaussians, 3, device=device)
        self.blur_grad_count = torch.zeros(num_gaussians, device=device)
        self.canonical_grad_dir_sum = torch.zeros(num_gaussians, 3, device=device)
        self.canonical_grad_count = torch.zeros(num_gaussians, device=device)
        self.protect_until_iter = torch.zeros(num_gaussians, dtype=torch.long, device=device)
        self.gaussian_age = torch.zeros(num_gaussians, dtype=torch.long, device=device)
        self.source_type = torch.full((num_gaussians,), source_type, dtype=torch.uint8, device=device)

    def assert_shape(self, n):
        assert self.blur_grad_dir_sum.shape[0] == n
        assert self.blur_grad_count.shape[0] == n
        assert self.canonical_grad_dir_sum.shape[0] == n
        assert self.canonical_grad_count.shape[0] == n
        assert self.protect_until_iter.shape[0] == n
        assert self.gaussian_age.shape[0] == n
        assert self.source_type.shape[0] == n

    def update_blur_stats(self, viewspace_points, visibility, blur_type=None, grad=None):
        self._update(self.blur_grad_dir_sum, self.blur_grad_count, viewspace_points, visibility, grad)

    def update_canonical_stats(self, viewspace_points, visibility, grad=None):
        self._update(self.canonical_grad_dir_sum, self.canonical_grad_count, viewspace_points, visibility, grad)

    def _update(self, grad_dir_sum, grad_count, viewspace_points, visibility, grad=None):
        if isinstance(viewspace_points, list):
            for idx, points in enumerate(viewspace_points):
                vis = visibility[idx] if isinstance(visibility, list) else visibility
                item_grad = grad[idx] if isinstance(grad, list) else None
                self._update_one(grad_dir_sum, grad_count, points, vis, item_grad)
            return
        self._update_one(grad_dir_sum, grad_count, viewspace_points, visibility, grad)

    def _update_one(self, grad_dir_sum, grad_count, viewspace_points, visibility, grad=None):
        if grad is None:
            grad = getattr(viewspace_points, "grad", None)
        if grad is None:
            return
        grad_visible = grad[visibility]
        if grad_visible.numel() == 0:
            return
        grad3 = torch.zeros(grad_visible.shape[0], 3, device=grad_visible.device, dtype=grad_visible.dtype)
        grad3[:, : min(3, grad_visible.shape[-1])] = grad_visible[:, : min(3, grad_visible.shape[-1])]
        grad_dir = grad3 / (grad3.norm(dim=-1, keepdim=True) + 1e-8)
        grad_dir_sum[visibility] += grad_dir.detach()
        grad_count[visibility] += 1

    def on_clone(self, parent_idx, current_iter, protect_iters):
        parent_idx = parent_idx.to(device=self.blur_grad_count.device, dtype=torch.long)
        self._append(
            0.5 * self.blur_grad_dir_sum[parent_idx].clone(),
            0.5 * self.blur_grad_count[parent_idx].clone(),
            0.5 * self.canonical_grad_dir_sum[parent_idx].clone(),
            0.5 * self.canonical_grad_count[parent_idx].clone(),
            self._new_protect(parent_idx.numel(), current_iter, protect_iters),
            self._new_age(parent_idx.numel()),
            self._new_source(parent_idx.numel(), SOURCE_CLONE),
        )

    def on_split(self, parent_idx_repeated, current_iter, protect_iters):
        parent_idx_repeated = parent_idx_repeated.to(device=self.blur_grad_count.device, dtype=torch.long)
        self._append(
            0.25 * self.blur_grad_dir_sum[parent_idx_repeated].clone(),
            0.25 * self.blur_grad_count[parent_idx_repeated].clone(),
            0.25 * self.canonical_grad_dir_sum[parent_idx_repeated].clone(),
            0.25 * self.canonical_grad_count[parent_idx_repeated].clone(),
            self._new_protect(parent_idx_repeated.numel(), current_iter, protect_iters),
            self._new_age(parent_idx_repeated.numel()),
            self._new_source(parent_idx_repeated.numel(), SOURCE_SPLIT),
        )

    def on_external_add(self, num_new, source_type, current_iter, protect_iters):
        device = self.blur_grad_count.device
        self._append(
            torch.zeros(num_new, 3, device=device),
            torch.zeros(num_new, device=device),
            torch.zeros(num_new, 3, device=device),
            torch.zeros(num_new, device=device),
            self._new_protect(num_new, current_iter, protect_iters),
            self._new_age(num_new),
            self._new_source(num_new, source_type),
        )

    def on_prune(self, keep_mask):
        keep_mask = keep_mask.to(device=self.blur_grad_count.device, dtype=torch.bool)
        self.blur_grad_dir_sum = self.blur_grad_dir_sum[keep_mask]
        self.blur_grad_count = self.blur_grad_count[keep_mask]
        self.canonical_grad_dir_sum = self.canonical_grad_dir_sum[keep_mask]
        self.canonical_grad_count = self.canonical_grad_count[keep_mask]
        self.protect_until_iter = self.protect_until_iter[keep_mask]
        self.gaussian_age = self.gaussian_age[keep_mask]
        self.source_type = self.source_type[keep_mask]

    def step_age(self):
        self.gaussian_age += 1

    def _append(self, blur_dir, blur_count, can_dir, can_count, protect, age, source):
        self.blur_grad_dir_sum = torch.cat((self.blur_grad_dir_sum, blur_dir), dim=0)
        self.blur_grad_count = torch.cat((self.blur_grad_count, blur_count), dim=0)
        self.canonical_grad_dir_sum = torch.cat((self.canonical_grad_dir_sum, can_dir), dim=0)
        self.canonical_grad_count = torch.cat((self.canonical_grad_count, can_count), dim=0)
        self.protect_until_iter = torch.cat((self.protect_until_iter, protect), dim=0)
        self.gaussian_age = torch.cat((self.gaussian_age, age), dim=0)
        self.source_type = torch.cat((self.source_type, source), dim=0)

    def _new_protect(self, n, current_iter, protect_iters):
        return torch.full((n,), current_iter + protect_iters, dtype=torch.long, device=self.blur_grad_count.device)

    def _new_age(self, n):
        return torch.zeros(n, dtype=torch.long, device=self.blur_grad_count.device)

    def _new_source(self, n, source_type):
        return torch.full((n,), source_type, dtype=torch.uint8, device=self.blur_grad_count.device)
