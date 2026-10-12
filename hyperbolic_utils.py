"""
双曲空间工具函数 (Poincaré Ball Model)
ref: https://arxiv.org/abs/1705.08039 (Poincaré embeddings)
ref: https://arxiv.org/abs/2310.18249 (Hyperbolic VICReg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def expmap0(v, c=1.0, clip_eps=1e-5):
    """
    从切线空间 (原点处) 指数映射到 Poincaré 球
    
    关键改进：使用 tanh 软压缩替代硬裁剪
    - 硬裁剪 (clamp)：范数超过 max_norm 时梯度=0，阻断反传
    - tanh 软压缩：梯度处处非零，范数渐进趋近 max_norm 但永不截断
    
    v: (B, d) 切线空间向量
    c: 曲率
    clip_eps: 安全边界
    return: (B, d) Poincaré 球上的点
    """
    v_norm_sq = v.pow(2).sum(dim=-1, keepdim=True)
    v_norm = torch.sqrt(v_norm_sq + EPS)

    sqrt_c = c ** 0.5
    max_norm = 1.0 / sqrt_c - clip_eps

    # tanh 软压缩：v_norm 从 0→∞ 时，scaled_norm 从 0→max_norm
    # 梯度处处非零，不会像 clamp 那样在裁剪区截断
    scaled_norm = torch.tanh(v_norm / max_norm) * max_norm
    v_scaled = v / (v_norm + EPS) * scaled_norm

    # 指数映射
    scale = torch.tanh(sqrt_c * scaled_norm) / (sqrt_c * scaled_norm + EPS)
    return scale * v_scaled


def logmap0(x, c=1.0):
    """
    从 Poincaré 球对数映射到切线空间 (原点处)
    x: (B, d) Poincaré 球上的点
    c: 曲率
    return: (B, d) 切线空间向量
    """
    x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
    x_norm = torch.sqrt(x_norm_sq + EPS)
    sqrt_c = c ** 0.5
    scale = torch.arctanh(torch.clamp(sqrt_c * x_norm, max=1.0 - EPS)) / (sqrt_c * x_norm + EPS)
    return scale * x


def mobius_add(x, y, c=1.0):
    """
    Poincaré 球上的 Möbius 加法
    x, y: (B, d) Poincaré 球上的点
    """
    x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm_sq = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    numerator = (1 + 2 * c * xy + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
    denominator = 1 + 2 * c * xy + c * c * x_norm_sq * y_norm_sq
    return numerator / (denominator + EPS)


def hyp_dist(x, y, c=1.0):
    """
    双曲距离 (Poincaré 球)
    dist(x, y) = (2 / sqrt(c)) * arctanh(sqrt(c) * || -x ⊕ y ||)
    """
    neg_x = -x
    mobius = mobius_add(neg_x, y, c)
    mobius_norm = mobius.pow(2).sum(dim=-1, keepdim=True).sqrt()
    sqrt_c = c ** 0.5
    dist = (2.0 / (sqrt_c + EPS)) * torch.arctanh(torch.clamp(sqrt_c * mobius_norm, max=1.0 - EPS))
    return dist.squeeze(-1)


class HyperbolicSimilarityLoss(nn.Module):
    """
    双曲空间中的相似度损失
    使用双曲距离作为损失，与欧氏版本的 -cos_sim 语义一致

    原始 Similarity_Loss 返回: -cos_sim（最小化 → cos_sim 增大 → 视图靠近）
    双曲版本返回: +distance（最小化 → 距离减小 → 视图靠近）
    注意：距离与相似度成反比，因此符号与欧氏版相反才能保持同样的优化方向
    """
    def __init__(self, curvature=1.0):
        super().__init__()
        self.curvature = curvature

    def forward(self, z_list, z_avg):
        """
        z_list: list of (B, d) tensors, each is a patch group in Poincaré ball
        z_avg: (B, d) average of all patches in Poincaré ball (via Möbius mean)
        """
        num_patch = len(z_list)
        total_dist = 0.0
        for i in range(num_patch):
            total_dist += hyp_dist(z_list[i], z_avg, c=self.curvature).mean()
        avg_dist = total_dist / num_patch

        # 返回平均双曲距离：最小化损失 = 拉近各视图与样本均值的距离（视图不变性）
        return avg_dist, avg_dist.clone().detach()


def mobius_mean(z_list, c=1.0, max_iter=10):
    """
    计算 Poincaré 球上点的 Fréchet 均值 (Möbius 平均)
    使用迭代加权 Möbius 加法
    
    z_list: (num_patches, B, d) 或 list/tuple of (B, d)
    """
    # 检查是否是 list 或 tuple（chunk() 返回 tuple）
    if isinstance(z_list, (list, tuple)):
        z_stack = torch.stack(z_list, dim=0)  # (P, B, d)
    else:
        z_stack = z_list
    
    P = z_stack.shape[0]
    mean = z_stack.mean(dim=0)
    mean_norm = mean.pow(2).sum(dim=-1, keepdim=True).sqrt()
    sqrt_c = c ** 0.5
    scale = torch.clamp(1.0 / (sqrt_c * mean_norm + EPS), max=1.0 - EPS)
    mean = scale * mean

    for _ in range(max_iter):
        logs = []
        for i in range(P):
            neg_mean = -mean
            transported = mobius_add(neg_mean, z_stack[i], c)
            log_i = logmap0(transported, c)
            logs.append(log_i)
        log_mean = torch.stack(logs, dim=0).mean(dim=0)
        mean = expmap0(log_mean, c)
    
    return mean


class HyperbolicTCRLoss(nn.Module):
    """
    双曲空间中的 TCR 损失（通过切空间映射）
    策略：将双曲点通过 logmap0 映射到切线空间，在切线空间中计算标准 TCR
    """
    def __init__(self, eps=0.01, curvature=1.0):
        super().__init__()
        self.eps = eps
        self.curvature = curvature

    def compute_discrimn_loss(self, W):
        """标准 TCR 判别损失（eigvalsh 实现, 避免 rocBLAS trsm/logdet 内核启动越界）"""
        p, m = W.shape  # [d, B]
        scalar = p / (m * self.eps)
        M = scalar * W.matmul(W.T)  # (d, d)
        # 对称矩阵特征值分解 (走 syevd 核, 不经过 rocBLAS trsm)
        eigvals = torch.linalg.eigvalsh(M)
        # logdet(I + M) = sum(log(1 + eigenvalue_i))
        logdet = torch.log1p(eigvals.clamp(min=-1.0 + 1e-6)).sum()
        return logdet / 2.0

    def forward(self, X):
        """
        X: (B, d) Poincaré 球上的点
        """
        tangent_X = logmap0(X, c=self.curvature)
        return -self.compute_discrimn_loss(tangent_X.T)


class HyperbolicCodingRateLoss(nn.Module):
    """
    【真正的双曲编码率损失】直接在双曲空间计算
    
    核心思想：最大化双曲空间中点集的"分散度"
    1. 计算点集的 Fréchet 均值（双曲均值）
    2. 计算所有点到均值的平均双曲距离
    3. 最大化这个平均距离（类似于 TCR 最大化编码率）
    
    这比映射到切空间再计算更符合双曲几何直觉
    """
    def __init__(self, curvature=1.0, mean_max_iter=10):
        super().__init__()
        self.curvature = curvature
        self.mean_max_iter = mean_max_iter

    def forward(self, X):
        """
        X: (B, d) Poincaré 球上的点
        
        返回：负的平均距离（因为我们要最小化损失 = 最大化分散度）
        """
        # 计算双曲均值（Fréchet mean）
        batch_mean = mobius_mean(X.unsqueeze(0), c=self.curvature, max_iter=self.mean_max_iter)
        
        # 计算所有点到均值的双曲距离
        total_dist = 0.0
        for i in range(X.shape[0]):
            total_dist += hyp_dist(X[i:i+1], batch_mean, c=self.curvature).mean()
        
        avg_dist = total_dist / X.shape[0]
        
        # 返回负的平均距离：最小化损失 = 最大化分散度
        return -avg_dist