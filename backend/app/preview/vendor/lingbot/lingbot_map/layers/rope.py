# 本文件为 3DGS 预览系统内置算法代码，裁剪自对应上游仓库的关键运行路径；保留原许可证，避免运行时依赖 GitHub 克隆目录。
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from typing import List, Optional, Tuple, Union


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components.
            # .detach().clone() ensures the cached tensors are plain CUDA tensors
            # (not CUDA-graph-owned memory), so they can safely be reused as inputs
            # to subsequent torch.compile / CUDA graph captures.
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype).detach().clone()
            sin_components = angles.sin().to(dtype).detach().clone()
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components.
        # Use positions.shape[1] (token count) as the frequency table size instead of
        # int(positions.max()) + 1.  Both are valid upper bounds, but shape[1] is a
        # static integer known at trace time, so it is CUDA-graph-compatible.
        # (positions.max() requires a device鈫抙ost sync / aten._local_scalar_dense,
        # which prevents CUDA graph capture in torch.compile.)
        max_position = positions.shape[1]
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)
    


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    璁＄畻1D鏃嬭浆浣嶇疆缂栫爜锛圧oPE锛夌殑棰戠巼寮犻噺銆?
    
    RoPE鐨勬牳蹇冩€濇兂锛氫娇鐢ㄦ棆杞煩闃垫潵缂栫爜浣嶇疆淇℃伅锛屼娇寰楃浉瀵逛綅缃叧绯讳繚鎸佷笉鍙樸€?
    鍏紡锛氬浜庝綅缃甿鍜岀淮搴锛岄鐜囦负 胃_i = 胃^(-2i/d)锛屽叾涓告槸鍩虹棰戠巼锛堥粯璁?0000锛?
    
    Args:
        dim: 鐗瑰緛缁村害锛屽繀椤绘槸鍋舵暟锛堝洜涓鸿鎴愬澶勭悊锛?
        pos: 浣嶇疆绱㈠紩锛屽彲浠ユ槸鏁存暟锛堣嚜鍔ㄧ敓鎴?鍒皃os-1鐨勫簭鍒楋級鎴栦綅缃暟缁?[S]
        theta: 鍩虹棰戠巼锛屾帶鍒朵綅缃紪鐮佺殑鍛ㄦ湡鎬э紙榛樿10000锛?
        use_real: 鏄惁杩斿洖瀹炴暟褰㈠紡锛坈os鍜宻in鍒嗗紑锛夎繕鏄鏁板舰寮?
        linear_factor: 绾挎€х缉鏀惧洜瀛愶紝鐢ㄤ簬涓婁笅鏂囨墿灞?
        ntk_factor: NTK-Aware缂╂斁鍥犲瓙锛岀敤浜庡鐞嗘洿闀跨殑搴忓垪
        repeat_interleave_real: 褰搖se_real=True鏃讹紝鏄惁浜ら敊閲嶅锛堢敤浜庢煇浜涙ā鍨嬫灦鏋勶級
        freqs_dtype: 棰戠巼寮犻噺鐨勬暟鎹被鍨?
        
    Returns:
        澶嶆暟褰㈠紡锛歔S, D/2] 鐨勫鏁板紶閲忥紝琛ㄧず e^(i*m*胃_j)
        瀹炴暟褰㈠紡锛氫袱涓?[S, D] 鐨勫紶閲忥紙cos鍜宻in锛?
    """
    # 纭繚缁村害鏄伓鏁帮紙RoPE闇€瑕佹垚瀵瑰鐞嗙淮搴︼級
    assert dim % 2 == 0

    # 灏嗕綅缃浆鎹负torch寮犻噺
    if isinstance(pos, int):
        pos = torch.arange(pos)  # 鐢熸垚 [0, 1, 2, ..., pos-1]
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # [S]

    # 搴旂敤NTK缂╂斁锛圢eural Tangent Kernel锛岀敤浜庡鐞嗚缁冩椂鏈杩囩殑闀垮簭鍒楋級
    theta = theta * ntk_factor
    
    # 姝ラ1锛氳绠楅鐜?胃_i = 1 / (胃^(2i/d))
    # 鍏朵腑 i 鈭?{0, 2, 4, ..., dim-2}锛堝彧鍙栧伓鏁扮储寮曪紝鍥犱负鎴愬澶勭悊锛?
    # 鍏紡锛歠req_i = 1 / (theta^(2i/d) * linear_factor)
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]锛屾瘡涓鐜囧搴斾竴涓淮搴﹀
    
    # 姝ラ2锛氳绠椾綅缃?棰戠巼鐭╅樀
    # 浣跨敤澶栫Н锛歱os[m] * freqs[i] = m * 胃_i
    # 缁撴灉锛氭瘡涓綅缃甿鍜屾瘡涓鐜噄鐨勭粍鍚?
    freqs = torch.outer(pos, freqs)  # [S, D/2]
    
    # 姝ラ3锛氭牴鎹繑鍥炴牸寮忚浆鎹?
    if use_real and repeat_interleave_real:
        # 鏂瑰紡1锛氫氦閿欓噸澶嶏紙鐢ㄤ簬flux, hunyuan-dit, cogvideox绛夋ā鍨嬶級
        # 灏嗘瘡涓鐜囩殑cos鍜宻in浜ら敊鎺掑垪锛歔cos_0, cos_0, cos_1, cos_1, ...]
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # 鏂瑰紡2锛氭嫾鎺ラ噸澶嶏紙鐢ㄤ簬stable audio, allegro绛夋ā鍨嬶級
        # 灏嗘墍鏈塩os鎷兼帴锛岀劧鍚庢槸鎵€鏈塻in锛歔cos_0, cos_1, ..., cos_n, cos_0, cos_1, ..., cos_n]
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # 鏂瑰紡3锛氬鏁板舰寮忥紙鐢ㄤ簬lumina绛夋ā鍨嬶級
        # 浣跨敤娆ф媺鍏紡锛歟^(i胃) = cos(胃) + i*sin(胃)
        # torch.polar(r, 胃) 杩斿洖 r * e^(i胃)锛岃繖閲宺=1锛屾墍浠ュ氨鏄?e^(i*freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64: [S, D/2]
        return freqs_cis


class WanRotaryPosEmbed(nn.Module):
    """
    3D鏃嬭浆浣嶇疆缂栫爜锛?D RoPE锛夋ā鍧?
    
    鏍稿績鎬濇兂锛氬皢RoPE鎵╁睍鍒?D绌洪棿锛堟椂闂淬€侀珮搴︺€佸搴︼級锛屼负瑙嗛鎴?D鏁版嵁鎻愪緵浣嶇疆缂栫爜銆?
    姣忎釜缁村害锛坱, h, w锛夌嫭绔嬩娇鐢≧oPE锛岀劧鍚庢嫾鎺ヨ捣鏉ャ€?
    
    鍏紡锛?
    瀵逛簬3D浣嶇疆 (f, h, w)锛堝抚銆侀珮搴︺€佸搴︼級锛?
    - 甯х淮搴︿娇鐢?dim_f 涓壒寰佺淮搴?
    - 楂樺害缁村害浣跨敤 dim_h 涓壒寰佺淮搴? 
    - 瀹藉害缁村害浣跨敤 dim_w 涓壒寰佺淮搴?
    鍏朵腑 dim_f + dim_h + dim_w = attention_head_dim
    """
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        fhw_dim: Optional[Tuple[int, int, int]] = [20, 22, 22],
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim  # 娉ㄦ剰鍔涘ご鐨勬€荤淮搴?
        self.patch_size = patch_size  # patch澶у皬 (patch_f, patch_h, patch_w)
        self.max_seq_len = max_seq_len  # 鏈€澶у簭鍒楅暱搴︼紙鐢ㄤ簬棰勮绠楅鐜囷級

        # 姝ラ1锛氬垎閰嶇淮搴︾粰涓変釜绌洪棿缁村害
        if fhw_dim is not None:
            # 濡傛灉鎸囧畾浜嗙淮搴﹀垎閰嶏紝浣跨敤鎸囧畾鐨?
            assert attention_head_dim == sum(
                fhw_dim
            ), f"attention_head_dim {attention_head_dim} must match sum(fhw_dim) {sum(fhw_dim)}"
            t_dim, h_dim, w_dim = fhw_dim
        else:
            # 鍚﹀垯鑷姩鍒嗛厤锛歨鍜寃鍚勫崰1/3锛宼鍗犲墿浣?
            # 渚嬪锛氬鏋渁ttention_head_dim=64锛屽垯 h_dim=w_dim=21锛宼_dim=22
            h_dim = w_dim = 2 * (attention_head_dim // 6)
            t_dim = attention_head_dim - h_dim - w_dim
        
        # 淇濆瓨缁村害鍒嗛厤浠ヤ究鍦╢orward涓娇鐢?
        self.fhw_dim = (t_dim, h_dim, w_dim)

        # 姝ラ2锛氫负姣忎釜缁村害棰勮绠楅鐜?
        # 鍒嗗埆璁＄畻鏃堕棿銆侀珮搴︺€佸搴︿笁涓淮搴︾殑RoPE棰戠巼
        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            # 姣忎釜缁村害鐙珛璋冪敤1D RoPE
            # 杩斿洖澶嶆暟褰㈠紡鐨勯鐜? [max_seq_len, dim//2]
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        # 灏嗕笁涓淮搴︾殑棰戠巼鍦ㄦ渶鍚庝竴缁存嫾鎺? [max_seq_len, (t_dim + h_dim + w_dim)//2]
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, ppf, pph, ppw, patch_start_idx, device: torch.device, f_start: int = 0, f_end: Optional[int] = None) -> torch.Tensor:
        """
        鍓嶅悜浼犳挱锛氫负3D杈撳叆锛堣棰戝抚+patch锛夌敓鎴愭棆杞綅缃紪鐮?
        
        鍙傛暟锛?
        - ppf (int): 甯ф暟锛坧atches per frame锛夛紝褰揻_end涓篘one鏃朵娇鐢?
        - pph (int): 姣忓抚鐨刾atch楂樺害鏁伴噺
        - ppw (int): 姣忓抚鐨刾atch瀹藉害鏁伴噺  
        - patch_start_idx (int): 姣忓抚鐨勭壒娈妕oken鏁伴噺锛堝湪patches涔嬪墠锛?
        - device: 璁＄畻璁惧锛圕PU/GPU锛?
        - f_start (int): 璧峰甯х储寮曪紙鐢ㄤ簬causal妯″紡锛夛紝榛樿涓?
        - f_end (Optional[int]): 缁撴潫甯х储寮曪紙鐢ㄤ簬causal妯″紡锛夛紝濡傛灉涓篘one鍒欎娇鐢╬pf浣滀负甯ф暟
        
        杩斿洖锛?
        - freqs: [1, 1, ppf * (patch_start_idx + pph * ppw), head_dim//2] 澶嶆暟棰戠巼tensor
        
        Token鎺掑垪椤哄簭锛?
        [frame0_special_token_0, ..., frame0_special_token_N,
         frame0_patch_0, ..., frame0_patch_M,
         frame1_special_token_0, ..., frame1_special_token_N,
         frame1_patch_0, ..., frame1_patch_M,
         ...]
        
        妯″紡锛?
        - 闈瀋ausal妯″紡锛歠_end=None锛屼娇鐢╬pf浣滀负甯ф暟锛屼粠浣嶇疆0寮€濮?
        - Causal妯″紡锛歠_end涓嶄负None锛屼娇鐢╗f_start, f_end)鑼冨洿鐨勫抚锛宲pf浼氳閲嶆柊璁＄畻
        """

        # 姝ラ1锛氬皢棰勮绠楃殑棰戠巼绉诲埌鐩爣璁惧锛屽苟鍒嗗壊鎴愪笁涓淮搴?
        self.freqs = self.freqs.to(device)
        # 鑾峰彇瀹為檯鐨勭淮搴﹀垎閰?
        if hasattr(self, 'fhw_dim') and self.fhw_dim is not None:
            t_dim, h_dim, w_dim = self.fhw_dim
        else:
            # 鑷姩鍒嗛厤鐨勬儏鍐?
            h_dim = w_dim = 2 * (self.attention_head_dim // 6)
            t_dim = self.attention_head_dim - h_dim - w_dim
        
        # 浣跨敤姝ｇ‘鐨剆plit sizes锛堟瘡涓淮搴︾殑涓€鍗婏級
        freqs = self.freqs.split_with_sizes(
            [
                t_dim // 2,  # 鏃堕棿缁村害
                h_dim // 2,  # 楂樺害缁村害
                w_dim // 2,  # 瀹藉害缁村害
            ],
            dim=1,
        )
        
        # 澶勭悊causal妯″紡锛氬鏋滄寚瀹氫簡f_end锛岄噸鏂拌绠梡pf鍜屽抚鑼冨洿
        if f_end is not None:
            ppf = f_end - f_start
            frame_slice = slice(f_start, f_end)
        else:
            # 闈瀋ausal妯″紡锛氫娇鐢ㄤ粠0寮€濮嬬殑ppf涓抚
            frame_slice = slice(0, ppf)
        
        # 姝ラ2锛氬鐞嗙壒娈妕oken锛堝鏋滃瓨鍦級
        ## For other tokens
        if patch_start_idx > 0:
            # 2.1 涓虹壒娈妕oken鐢熸垚浣嶇疆缂栫爜
            # 鐗规畩token浣嶄簬瀵硅绾夸綅缃?(f, i, i)锛屾瘡涓壒娈妕oken鏈夊敮涓€浣嶇疆
            # camera: (f, 0, 0), register_0: (f, 1, 1), ..., scale: (f, 5, 5)
            # Shape: (ppf, patch_start_idx, dim)
            freqs_special_f = freqs[0][frame_slice].reshape(ppf, 1, -1).expand(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim_f) 甯х淮搴﹀彉鍖?
            freqs_special_h = freqs[1][:patch_start_idx].reshape(1, patch_start_idx, -1).expand(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim_h) 楂樺害=0,1,2,...
            freqs_special_w = freqs[2][:patch_start_idx].reshape(1, patch_start_idx, -1).expand(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim_w) 瀹藉害=0,1,2,...
            freqs_special = torch.cat([freqs_special_f, freqs_special_h, freqs_special_w], dim=-1)  # (ppf, patch_start_idx, dim) 鎷兼帴涓夌淮
            freqs_special = freqs_special.reshape(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim)

            # 2.2 涓哄浘鍍弍atch鐢熸垚浣嶇疆缂栫爜
            # Patch浣嶄簬 (f, patch_start_idx+h, patch_start_idx+w)锛宧,w 鏁翠綋鍋忕Щ patch_start_idx
            # 杩欐牱 patches 涓?special tokens 浣嶇疆涓嶅啿绐侊紝涓?h,w 瀵圭О澶勭悊
            # Shape: (ppf, pph, ppw, dim)
            freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_f) 甯х淮搴?
            freqs_h = freqs[1][patch_start_idx : patch_start_idx + pph].reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_h) 楂樺害浠巔atch_start_idx寮€濮?
            freqs_w = freqs[2][patch_start_idx : patch_start_idx + ppw].reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_w) 瀹藉害浠巔atch_start_idx寮€濮?
            freqs_patches = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)  # (ppf, pph, ppw, dim) 鎷兼帴涓夌淮
            freqs_patches = freqs_patches.reshape(ppf, pph * ppw, -1)  # (ppf, pph * ppw, dim) 灞曞钩绌洪棿缁村害
            
            # 姝ラ3锛氭寜鐓ф纭殑椤哄簭缁勫悎鐗规畩token鍜宲atches
            # 姣忓抚鍐呴儴椤哄簭锛歔鐗规畩tokens, patches]
            # Concatenate special tokens and patches for each frame along the second dimension
            # Shape: (ppf, patch_start_idx + pph * ppw, dim)
            freqs = torch.cat([freqs_special, freqs_patches], dim=1)  # (ppf, patch_start_idx + pph * ppw, dim)
            
            # 姝ラ4锛氬睍骞充负鏈€缁堝舰鐘跺苟娣诲姞batch鍜宧ead缁村害
            # Flatten to get final shape: (ppf * (patch_start_idx + pph * ppw), dim)
            freqs = freqs.reshape(ppf * (patch_start_idx + pph * ppw), -1)
            freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, ppf * (patch_start_idx + pph * ppw), dim) 娣诲姞batch鍜宧ead缁村害
            return freqs
        
        # 濡傛灉娌℃湁鐗规畩token锛坧atch_start_idx == 0锛夛紝鍙鐞嗗浘鍍弍atches
        # 鎵€鏈塸atches浣嶄簬 (f, 0:pph, 0:ppw)
        freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_f) 甯х淮搴?
        freqs_h = freqs[1][:pph].reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_h) 楂樺害浠?寮€濮?
        freqs_w = freqs[2][:ppw].reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_w) 瀹藉害浠?寮€濮?
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)  # (1, 1, ppf * pph * ppw, dim)
        return freqs
    
def apply_rotary_emb(x, freqs):
    """Apply 3D rotary position embedding using real arithmetic (torch.compile-safe).

    Equivalent to complex multiplication but avoids torch.view_as_complex /
    view_as_real, which are not supported by torchinductor and break CUDA graphs.

    Args:
        x: [B, H, N, D] real tensor (bfloat16 or float32).
        freqs: [1, 1, N, D//2] complex tensor (cos + i*sin per frequency).

    Returns:
        [B, H, N, D] tensor of same dtype as x.
    """
    # Real-arithmetic implementation: equivalent to (x1+i*x2)*(cos+i*sin) but avoids
    # torch.view_as_complex / view_as_real which break torch.compile CUDA graphs.
    cos = freqs.real.to(x.dtype)  # [1, 1, N, D//2]
    sin = freqs.imag.to(x.dtype)  # [1, 1, N, D//2]

    # Interleaved pairs: even indices = "real", odd indices = "imag"
    x1 = x[..., 0::2]  # [B, H, N, D//2]
    x2 = x[..., 1::2]  # [B, H, N, D//2]

    # (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.stack([out1, out2], dim=-1).reshape(x.shape)
