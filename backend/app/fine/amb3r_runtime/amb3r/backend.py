
import torch
import torch.nn as nn

from torch_scatter import scatter_mean

from .blocks import ZeroConvBlock, DownBlock
from .tools.voxel_utils import get_vox_indices
from ..ptv3.point_transformer import PointTransformerV3




class BackEnd(nn.Module):
    def __init__(self, hash_base=1024, in_dim=1024, out_dim=256, 
                 k_neighbors=16, depth=48, interp_v2=False):
        super(BackEnd, self).__init__()
        self.register_buffer('interp_v2', torch.tensor(interp_v2, dtype=torch.bool))
        self.base = hash_base
        self.aligner = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.GELU(),
            nn.Linear(in_dim//2, out_dim),
            nn.GELU()
        )

        self.point_transformer = PointTransformerV3()
        self.k_neighbors = k_neighbors
        self.downsample = DownBlock(in_channels=1024, mid_channels=1024, out_channels=1024)
        self.zero_conv = ZeroConvBlock()
        self.gate_scale = nn.Parameter(torch.ones(1))

        self.zero_conv_layers = nn.ModuleList(
            [ZeroConvBlock() for _ in range(depth)]
        )
        self.gate_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(depth)]
        )


    @torch.no_grad()
    def hash_fn(self, coords):
        '''
        A simple hash function for voxel coordinates
        '''
        b, x, y, z = coords.unbind(dim=1)
        return ((b.long() << 48)
               | (x.long() << 32)
               | (y.long() << 16)
               |  z.long())
    
    
    def mean_by_voxel(self, points, feats, batch_ids, voxel_size, bounding_boxes):
        '''Compute mean features for each voxel.
        
        Params:
            - points: (N, 3) tensor of point coordinates
            - feats: (N, C) tensor of point features
            - batch_ids: (N,) tensor of batch indices for each point
            - voxel_size: scalar or (3,) tensor defining the size of each voxel
            - bounding_boxes: (B, 2, 3) tensor of min and max coordinates for each batch
        
        Returns:
            - voxel_feats: (M, C) tensor of mean features for each voxel
            - info: dict containing 'unique_indices' which are the voxel indices corresponding to the mean features
        
        '''
        
        voxel_indices = get_vox_indices(points, batch_ids, voxel_size, bounding_boxes, shift=False, cat_batch_ids=True)
        voxel_hash = self.hash_fn(voxel_indices)
        unique_hash, inverse_id = torch.unique(voxel_hash, return_inverse=True)

        voxel_feats = scatter_mean(feats, inverse_id, dim=0) 
        original_indices = torch.arange(voxel_hash.shape[0], device=voxel_hash.device)
        min_original_indices_per_unique_id = torch.full((unique_hash.shape[0],),
                                                voxel_hash.shape[0],
                                                dtype=torch.long,
                                                device=voxel_hash.device)
        
        first_occurrence_original_indices = torch.scatter_reduce(
            min_original_indices_per_unique_id,
            0,
            inverse_id,
            original_indices,
            reduce="amin",
            include_self=False
        )

        unique_voxel_indices = voxel_indices[first_occurrence_original_indices]

        info = {
            'unique_indices': unique_voxel_indices,
        }    

        return voxel_feats, info

    
    def voxel_to_point_interpolation(self, point_out, pts, chunk_size=50000):
        """Interpolate point/voxel features back to exact continuous points via batched chunked KNN."""
        Bs = pts.shape[0] if len(pts.shape) == 3 else (pts.shape[0] // pts.shape[1] if hasattr(pts, 'shape') and len(pts.shape) == 2 else 1)
        # Using .view so we figure out N dynamically
        if len(pts.shape) == 2:
            Bs = point_out.batch.max().item() + 1
            N = pts.shape[0] // Bs
        else:
            Bs, N, _ = pts.shape
            
        pts_feat_from_voxel = point_out.feat      # (V, C_out)
        pts_coord_from_voxel = point_out.coord    # (V, 3)
        pts_batch_from_voxel = point_out.batch    # (V,)

        original_pts = pts.view(Bs, N, 3)         # (Bs, N, 3)

        batch_outputs = []
        for batch_id in range(Bs):
            voxel_mask = pts_batch_from_voxel == batch_id
            voxel_coords = pts_coord_from_voxel[voxel_mask]
            voxel_feats = pts_feat_from_voxel[voxel_mask]
            if voxel_coords.numel() == 0:
                batch_outputs.append(torch.zeros((N, pts_feat_from_voxel.shape[-1]), device=pts.device, dtype=pts_feat_from_voxel.dtype))
                continue

            k_interp = min(self.k_neighbors, voxel_coords.shape[0])
            chunk_outputs = []
            for start_idx in range(0, N, chunk_size):
                end_idx = min(start_idx + chunk_size, N)
                query = original_pts[batch_id, start_idx:end_idx]
                dists = torch.cdist(query.float(), voxel_coords.float())
                dists, indices = torch.topk(dists, k=k_interp, dim=-1, largest=False)
                gathered = voxel_feats.index_select(0, indices.reshape(-1)).view(indices.shape[0], k_interp, -1)
                weights = 1.0 / (dists.to(gathered.dtype) + 1e-8)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                chunk_outputs.append((gathered * weights.unsqueeze(-1)).sum(dim=-2))
            batch_outputs.append(torch.cat(chunk_outputs, dim=0))

        return torch.stack(batch_outputs, dim=0)


    def forward(self, pts, feats, voxel_sizes, chunk_size=50000):
        '''
        Forward pass for the back-end processing.
        
        Params:
            - pts: (Bs, N, 3) tensor of point coordinates
            - feats: (Bs, N, C) tensor of point features
            - voxel_sizes: list of voxel sizes
            - chunk_size: int, number of points to process in each chunk for interpolation
        '''

        Bs, C = feats.shape[0], feats.shape[-1]

        if len(feats.shape) != 3:
            feats = feats.reshape(Bs, -1, C)
            pts = pts.reshape(Bs, -1, 3) 
            
        
        bounding_boxes = torch.zeros((Bs, 2, 3), device=pts.device)  # Dummy bounding boxes
        bounding_boxes[:, 0, :] = pts.view(Bs, -1, 3).min(dim=1).values
        bounding_boxes[:, 1, :] = pts.view(Bs, -1, 3).max(dim=1).values # Bs, 2, 3

        Bs, N, C = feats.shape
        pts = pts.reshape(-1, 3)
        feats = feats.reshape(-1, C)
        batch_ids = torch.arange(Bs).repeat_interleave(N).to(pts.device)

        level_feats = []

        for i, voxel_size in enumerate(voxel_sizes):
            feat, info = self.mean_by_voxel(pts, feats, batch_ids, voxel_size, bounding_boxes)
            vox_id = info['unique_indices']

            if isinstance(voxel_size, torch.Tensor):
                coord = voxel_size[vox_id[:, 0]] * vox_id[:, 1:]
            else:
                coord = voxel_size * vox_id[:, 1:]

            if self.interp_v2:
                coord = coord + bounding_boxes[vox_id[:, 0], 0]

            data_dict = {
                'feat': feat,
                'grid_coord': vox_id[:, 1:],
                'coord': coord,
                'batch': vox_id[:, 0],
            }

            point_out = self.point_transformer(data_dict)
            interpolated_feats = self.voxel_to_point_interpolation(
                point_out, pts, chunk_size
            )
            level_feats.append(interpolated_feats)
        
        return level_feats
