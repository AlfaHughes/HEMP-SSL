import argparse
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE, MDS
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader

from model.TosT.tost2 import vit_cab_tiny_12_p16
import model.crate as crate

matplotlib.use('Agg')

# ==================== 数据集常量 ====================
CIFAR10_LABELS = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
CIFAR10_LABELS_EN = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 人工统计语义相似度矩阵（行=列=类别，对角线=1）
CIFAR10_SEMANTIC_SIMILARITY = np.array([
    [1.0,    0.625,  0.425,  0.15,   0.16,   0.155,  0.125,  0.1325, 0.55,   0.5525],
    [0.625,  1.0,    0.1875, 0.225,  0.225,  0.225,  0.175,  0.3,    0.5,    0.8   ],
    [0.425,  0.1875, 1.0,    0.4,    0.4,    0.4,    0.4,    0.35,   0.15,   0.175 ],
    [0.15,   0.225,  0.4,    1.0,    0.475,  0.675,  0.4,    0.4,    0.1,    0.125 ],
    [0.16,   0.225,  0.4,    0.475,  1.0,    0.525,  0.325,  0.65,   0.125,  0.175 ],
    [0.155,  0.225,  0.4,    0.675,  0.525,  1.0,    0.325,  0.5,    0.1,    0.175 ],
    [0.125,  0.175,  0.4,    0.4,    0.325,  0.325,  1.0,    0.325,  0.1,    0.15  ],
    [0.1325, 0.3,    0.35,   0.4,    0.65,   0.5,    0.325,  1.0,    0.225,  0.3   ],
    [0.55,   0.5,    0.15,   0.1,    0.125,  0.1,    0.1,    0.225,  1.0,    0.475 ],
    [0.5525, 0.8,    0.175,  0.125,  0.175,  0.175,  0.15,   0.3,    0.475,  1.0   ],
])

# CIFAR10-H 人类混淆矩阵 (Peterson et al., ICCV 2019)
CIFAR10H_HUMAN_CONFUSION = np.array([
    [0.91, 0.01, 0.02, 0.01, 0.00, 0.00, 0.00, 0.01, 0.03, 0.01],
    [0.01, 0.95, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.03],
    [0.02, 0.00, 0.90, 0.01, 0.01, 0.01, 0.04, 0.01, 0.00, 0.00],
    [0.00, 0.00, 0.01, 0.84, 0.02, 0.10, 0.01, 0.01, 0.00, 0.01],
    [0.00, 0.00, 0.01, 0.01, 0.90, 0.01, 0.01, 0.06, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.10, 0.01, 0.87, 0.01, 0.01, 0.00, 0.00],
    [0.00, 0.00, 0.04, 0.01, 0.01, 0.00, 0.93, 0.01, 0.00, 0.00],
    [0.00, 0.00, 0.01, 0.01, 0.06, 0.01, 0.00, 0.91, 0.00, 0.00],
    [0.03, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.93, 0.03],
    [0.01, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.93],
])

# 由人类混淆矩阵对称化构造语义相似度矩阵
_confusion_sym = (CIFAR10H_HUMAN_CONFUSION + CIFAR10H_HUMAN_CONFUSION.T) / 2.0
_confusion_diag = np.diag(_confusion_sym).copy()
CIFAR10H_SEMANTIC_SIMILARITY = _confusion_sym / np.sqrt(np.outer(_confusion_diag, _confusion_diag))
np.fill_diagonal(CIFAR10H_SEMANTIC_SIMILARITY, 1.0)


# ==================== 权重加载 ====================
def load_pretrained_weights(model, pretrained_path, device='cuda'):
    """加载预训练/微调权重，自动处理 key 前缀、分类头跳过、形状不匹配"""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location=device)

    # 支持多种 checkpoint 格式
    # 1. 自监督预训练: state_dict / model_state_dict / model / encoder_state_dict
    # 2. 微调保存: {'net': ..., 'classifier': ...}  (evaluate_finetune.py 格式)
    state_dict = None
    for key in ['model_state_dict', 'state_dict', 'model', 'encoder_state_dict', 'net']:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            print(f"   从 checkpoint['{key}'] 提取权重")
            break
    if state_dict is None:
        # 可能直接就是 state_dict
        state_dict = checkpoint if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()) else None
    if state_dict is None:
        raise ValueError(f"无法从 checkpoint 中识别权重字典，顶层键: {list(checkpoint.keys())}")

    # 跳过分类头/投影头（特征提取只需要 backbone）
    # 注意: 'projection' 必须精确匹配投影头(key 以 'projection.' 开头),
    #       不能用子串匹配, 否则会误跳 pos_embeder.token_projection
    def _should_skip(key):
        k = key.lower()
        if k.startswith('projection.') or k == 'projection':
            return True
        if 'mlp_head' in k or 'classifier' in k or 'pre_feature' in k:
            return True
        return False

    new_state_dict = {}
    for k, v in state_dict.items():
        if _should_skip(k):
            continue
        new_k = k.replace("module.", "", 1).replace("backbone.", "").replace("encoder.backbone.", "")
        new_state_dict[new_k] = v

    model_state = model.state_dict()
    filtered = {}
    skip_keys = []
    for k, v in new_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skip_keys.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"✅ 加载完成: 总键 {len(new_state_dict)} | 成功 {len(filtered)} | 跳过 {len(skip_keys)} | "
          f"缺失 {len(missing)} | 多余 {len(unexpected)}")
    if missing:
        print(f"   ⚠️ 缺失前 5 个: {list(missing)[:5]}")
    return model


# ==================== 特征提取（统一函数，消除重复）====================
@torch.no_grad()
def extract_features_batch(model, images):
    """提取一个 batch 的 CLS token 特征，自动适配 forward_features / 手动前向"""
    if hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features')):
        features = model.forward_features(images)
        if features.dim() == 3:
            features = features[:, 0, :]
        return features.cpu().numpy()

    # 手动前向路径 B
    if hasattr(model, 'patch_embed'):
        patch_out = model.patch_embed(images)
        if isinstance(patch_out, (tuple, list)):
            x, (Hp, Wp) = patch_out
            has_hw = True
        else:
            x = patch_out
            Hp = Wp = int(x.shape[1] ** 0.5)
            has_hw = False
    elif hasattr(model, 'to_patch_embedding'):
        x = model.to_patch_embedding(images)
        Hp = Wp = int(x.shape[1] ** 0.5)
        has_hw = False
    else:
        raise AttributeError("Model has no patch_embed or to_patch_embedding")

    b, n, _ = x.shape
    if hasattr(model, 'pos_embeder'):
        pos = model.pos_embeder(b, Hp, Wp).reshape(b, -1, n).permute(0, 2, 1)
        x = x + pos
    if hasattr(model, 'pos_drop'):
        x = model.pos_drop(x)

    if hasattr(model, 'blocks'):
        for blk in model.blocks:
            if has_hw and 'H' in blk.forward.__code__.co_varnames:
                x = blk(x, Hp, Wp)
            else:
                x = blk(x)

    # CLS + Class Attention (CaiT 风格)
    if hasattr(model, 'cls_attn_blocks'):
        cls_tok = model.cls_token.expand(b, -1, -1) if hasattr(model, 'cls_token') else \
            torch.zeros(b, 1, x.shape[-1], device=x.device)
        x = torch.cat((cls_tok, x), dim=1)
        for blk in model.cls_attn_blocks:
            if has_hw and 'H' in blk.forward.__code__.co_varnames:
                x = blk(x, Hp, Wp)
            else:
                x = blk(x)
        if hasattr(model, 'norm'):
            x = model.norm(x)
        features = x[:, 0, :].cpu().numpy()

    # 标准 ViT 风格
    else:
        if hasattr(model, 'cls_token'):
            x = torch.cat((model.cls_token.expand(b, -1, -1), x), dim=1)
        if hasattr(model, 'pos_embedding'):
            x += model.pos_embedding[:, :(n + 1)]
        if hasattr(model, 'dropout'):
            x = model.dropout(x)
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            for attn, ff in model.transformer.layers:
                x = ff(attn(x) + x)
        elif hasattr(model, 'blocks'):
            for blk in model.blocks:
                x = blk(x)
        features = x[:, 0, :].cpu().numpy()

    return features


def compute_class_centers(model, dataloader, num_classes=10, device='cuda'):
    model.eval()
    class_features = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for images, labels in dataloader:
            features = extract_features_batch(model, images.to(device))
            for i, label in enumerate(labels):
                lbl = label.item() if torch.is_tensor(label) else int(label)
                class_features[lbl].append(features[i])
    embed_dim = features.shape[-1]
    centers = [np.mean(class_features[i], axis=0) if class_features[i] else np.zeros(embed_dim)
               for i in range(num_classes)]
    return np.array(centers)


def extract_all_class_features(model, dataloader, num_classes=10, samples_per_class=100, device='cuda'):
    model.eval()
    all_features, all_labels = [], []
    class_counts = [0] * num_classes
    with torch.no_grad():
        for images, labels in dataloader:
            features = extract_features_batch(model, images.to(device))
            for i, label in enumerate(labels):
                lbl = label.item() if torch.is_tensor(label) else int(label)
                if class_counts[lbl] < samples_per_class:
                    all_features.append(features[i])
                    all_labels.append(lbl)
                    class_counts[lbl] += 1
            if all(c >= samples_per_class for c in class_counts):
                break
    return np.array(all_features), np.array(all_labels)


# ==================== 距离与 SCI ====================
def compute_distance_matrix(features):
    return cdist(features, features, metric='euclidean')


def compute_sci(semantic_similarity, feature_distances, class_indices):
    n = len(class_indices)
    S = semantic_similarity[np.ix_(class_indices, class_indices)]
    D = feature_distances[np.ix_(class_indices, class_indices)]
    mask = np.triu_indices(n, k=1)
    corr, p_value = pearsonr(S[mask], -D[mask])
    return corr, p_value


# ==================== 可视化 ====================
def plot_tsne(all_features, all_labels, checkpoint_name="model", semantic_matrix=None,
              dataset_labels=None, num_classes=10):
    if semantic_matrix is None:
        semantic_matrix = CIFAR10_SEMANTIC_SIMILARITY
    # 标签缺失或数量不足时，回退为数字标签
    if dataset_labels is None or len(dataset_labels) < num_classes:
        dataset_labels = [f'C{i}' for i in range(num_classes)]

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    n = num_classes
    if n <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n))
    elif n <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n))
    else:
        # 类别很多时用连续 colormap，避免颜色重复
        colors = plt.cm.hsv(np.linspace(0, 1, n))

    # 图1: 语义空间 MDS
    ax1 = axes[0]
    semantic_distance = 1 - semantic_matrix
    np.fill_diagonal(semantic_distance, 0)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=1000)
    coords = mds.fit_transform(semantic_distance)

    # 只画相似度高的连线 (top 10%)
    triu = np.triu_indices(n, k=1)
    sims = semantic_matrix[triu]
    threshold = np.percentile(sims, 90) if len(sims) > 0 else 0.3
    for idx in range(len(triu[0])):
        i, j = triu[0][idx], triu[1][idx]
        s = semantic_matrix[i, j]
        if s > threshold:
            c = 'green' if s > 0.5 else ('blue' if s > 0.35 else 'orange')
            alpha = min(0.6, s)
            ax1.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                     c, alpha=alpha, linewidth=1.5, zorder=1)
    for i in range(n):
        c = colors[i % len(colors)]
        ax1.scatter(coords[i, 0], coords[i, 1], c=[c], s=200 if n <= 20 else 80, marker='o',
                    edgecolors='black', linewidths=1.5, zorder=10)
        if n <= 20:
            ax1.annotate(dataset_labels[i], coords[i], fontsize=8, ha='center', va='bottom',
                         xytext=(0, 6), textcoords='offset points')
    ax1.set_title(f'Semantic Space (MDS, {n} classes)', fontsize=14)
    ax1.set_xlabel('Dimension 1'); ax1.set_ylabel('Dimension 2')

    # 图2: 特征空间 t-SNE
    ax2 = axes[1]
    n_samples = min(1000, len(all_features))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_features), n_samples, replace=False)
    f_sample, l_sample = all_features[indices], all_labels[indices]
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples - 1))
    samples_2d = tsne.fit_transform(f_sample)
    samples_2d[:, 1] = -samples_2d[:, 1]
    # 图例只画前 20 类，避免图例爆炸
    legend_n = min(n, 20)
    for label in range(n):
        mask = l_sample == label
        c = colors[label % len(colors)]
        ax2.scatter(samples_2d[mask, 0], samples_2d[mask, 1],
                    c=[c], label=dataset_labels[label] if label < legend_n else None,
                    alpha=0.6, s=30, zorder=5)
    if legend_n <= 10:
        ax2.legend(loc='best', fontsize=9)
    ax2.set_title(f'Feature Space (t-SNE, {n} classes, n={n_samples})', fontsize=14)
    ax2.set_xlabel('t-SNE dimension 1'); ax2.set_ylabel('t-SNE dimension 2')

    plt.tight_layout()
    save_path = f'tsne_{checkpoint_name}_{n}cls.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ t-SNE 可视化已保存至: {save_path}")
    plt.close()


# ==================== 报告打印 ====================
def print_distance_matrix(distance_matrix, labels=None, n=None):
    if n is None:
        n = len(labels) if labels is not None else distance_matrix.shape[0]
    if labels is None or len(labels) < n:
        labels = [f'C{i}' for i in range(n)]
    print(f"\n{'=' * 80}")
    print(f"📊 类中心特征空间距离矩阵 ({n}×{n})")
    print("=" * 80)
    # 类名缩写为前 6 字符，避免类别多时排版爆炸
    short_labels = [str(l)[:6] for l in labels]
    if n <= 20:
        print(f"{'':^8}", end="")
        for label in short_labels:
            print(f"{label:^8}", end="")
        print()
        for i in range(n):
            print(f"{short_labels[i]:^8}", end="")
            for j in range(n):
                print(f"{'--':^8}" if i == j else f"{distance_matrix[i, j]:^8.1f}", end="")
            print()
    else:
        # 大矩阵只打印统计信息
        off_diag = distance_matrix[~np.eye(n, dtype=bool)]
        print(f"  非对角线: min={off_diag.min():.2f}, max={off_diag.max():.2f}, "
              f"mean={off_diag.mean():.2f}, std={off_diag.std():.2f}")
        print(f"  对角线: min={np.diag(distance_matrix).min():.2f}, max={np.diag(distance_matrix).max():.2f}")


def print_sci_analysis(sci, p_value, label="SCI"):
    print(f"\n📉 {label}: {sci:.4f}")
    print(f"🎯 p 值: {p_value:.4f}")
    if sci > 0.7:
        verdict = "✅ 语义连续性优秀"
    elif sci > 0.3:
        verdict = "🟡 语义连续性良好"
    elif sci > 0:
        verdict = "⚠️ 语义连续性较弱"
    elif sci == 0:
        verdict = "⚪ 语义连续性为零"
    else:
        verdict = "❌ 语义不连续性严重"
    print(f"   {verdict}")


def print_cifar10h_confusion():
    print("\n" + "=" * 80)
    print("🧠 CIFAR10-H 人类混淆矩阵 (Peterson et al., ICCV 2019)")
    print("=" * 80)
    header = "true\\pred"
    print(f"{header:>10}", end="")
    for label in CIFAR10_LABELS_EN:
        print(f"{label:>8}", end="")
    print()
    for i, li in enumerate(CIFAR10_LABELS_EN):
        print(f"{li:>10}", end="")
        for j in range(10):
            v = CIFAR10H_HUMAN_CONFUSION[i, j]
            if i == j:
                print(f"\033[1;32m{v:>8.2f}\033[0m", end="")
            elif v > 0.05:
                print(f"\033[1;31m{v:>8.2f}\033[0m", end="")
            elif v > 0.01:
                print(f"\033[1;33m{v:>8.2f}\033[0m", end="")
            else:
                print(f"{v:>8.2f}", end="")
        print()

    print("\n🔍 主要人类混淆对 (P > 0.05):")
    pairs = sorted([(CIFAR10H_HUMAN_CONFUSION[i, j], i, j)
                    for i in range(10) for j in range(10) if i != j and CIFAR10H_HUMAN_CONFUSION[i, j] > 0.05],
                   reverse=True)
    for prob, i, j in pairs:
        print(f"   {CIFAR10_LABELS[i]} → {CIFAR10_LABELS[j]}: {prob:.2f}")


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description='Feature distance & semantic continuity verification')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'],
                        help='dataset to evaluate (default: cifar10)')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to pretrained checkpoint (若不指定则用下方默认)')
    parser.add_argument('--model', type=str, default='crate', choices=['vit_cab', 'crate'],
                        help='model architecture (default: crate)')
    parser.add_argument('--data_path', type=str, default='E:/DATA', help='dataset root',choices=['E:/DATA', 'E:/DATA/TinyImgenet'])
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--samples_per_class', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./verify_results', help='directory to save results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    args.dataset = args.dataset.strip()

    # ==================== 常用权重路径（取消注释切换，也可用 --checkpoint 命令行覆盖）====================
    # CRATE 自监督预训练
    CHECKPOINTS = {
        'crate_sl':      r"D:\myproject\python\8.pt.darts-learn-ma\augments\cifar10-crate-test\65.11best.pth.tar",
        'crate_sl_Timg':     r"D:\myproject\python\8.pt.darts-learn-ma\augments\tinyimagenet-crate-test\43.07best.pth.tar",
        'crate_emp_hyper':  r"D:\myproject\python\第四个点\EMP-SSL-main\crate_finetune_logs\models\CRATE-EMP-双曲\86.35best.pt",
        'crate_emp_hyper_100':   r"D:\myproject\python\第四个点\EMP-SSL-main\crate_finetune_logs_cifar100\models\59.17best.pt",
        'crate_emp_eud':    r"D:\myproject\python\第四个点\EMP-SSL-main\crate_finetune_logs\models\CRATE-EMP\84.72best.pt",
        'crate_emp_hyper_timg':    r"D:\myproject\python\第四个点\EMP-SSL-main\crate_finetune_logs_tinyimagenet\models\best.pt",
        # 'crate_hyp_proj':   r"D:\myproject\python\第四个点\EMP-SSL-main\logs\emp_hyp_proj\patchsim200_numpatch20_bs100_lr0.001_NONE\save_models\best.pt",
        # ViT-CAB 监督训练
        # 'vit_cab_67':       r"D:\tianyuan\模型结果\VIT\ViT\ViT-tiny-cifar10\best67.6.tar",
        'vit_cab_72':     r"D:\myproject\python\第四个点\EMP-SSL-main2\weight\tost-vit\72.5checkpoint.pth.tar",
        'vit_cab_43':       r"D:\myproject\python\第四个点\EMP-SSL-main2\weight\tost-vit\tiny_img\43.32best.pth.tar",
        # 对比学习基线
        'simclr_85':        r"D:\myproject\python\第四个点\EMP-SSL-main\finetune_results\SIMCLR\85.89best.pt",
        'byol_82':          r"D:\myproject\python\第四个点\EMP-SSL-main\finetune_results\BYOL\82.79best.pt",
        # 'swav_69':          r"D:\myproject\python\第四个点\EMP-SSL-main\finetune_results\SWAV\acc69.05.pth",
        # 'vicreg_12':        r"D:\myproject\python\第四个点\EMP-SSL-main\finetune_results\VICREG\acc12.70.pth",
    }
    if args.checkpoint is None:
        args.checkpoint = CHECKPOINTS['crate_sl']  # ← 改这里切换默认权重
    # ====================================================================================================

    # ==================== 数据集配置 ====================
    image_size = 32  # CIFAR 默认
    if args.dataset == 'cifar10':
        num_classes = 10
        dataset_labels = CIFAR10_LABELS
        semantic_matrix = CIFAR10_SEMANTIC_SIMILARITY
        dataset_cls = CIFAR10
        normalize_mean, normalize_std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'tinyimagenet':
        num_classes = 200
        dataset_labels = None
        dataset_cls = None  # Tiny-ImageNet 用 ImageFolder
        normalize_mean, normalize_std = [0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]
        image_size = 64  # Tiny-ImageNet 是 64×64
        # 优先加载 SCI/TinyImageNet/tinyimagenet_similarity_matrix.npy
        npy_path = os.path.join('SCI', 'TinyImageNet', 'tinyimagenet_similarity_matrix.npy')
        if os.path.exists(npy_path):
            semantic_matrix = np.load(npy_path)
            print(f"  📂 加载 Tiny-ImageNet 语义矩阵: {npy_path}")
            # 尝试加载类名 (tinyimagenet_semantic.py 生成)
            labels_npy = os.path.join('SCI', 'TinyImageNet', 'tinyimagenet_classes.npy')
            if os.path.exists(labels_npy):
                dataset_labels = list(np.load(labels_npy, allow_pickle=True))
        else:
            print(f"  ⚠️ 未找到 {npy_path}，将用 WordNet Wu-Palmer 在线计算...")
            semantic_matrix = None  # 后续动态计算
    else:  # cifar100
        num_classes = 100
        # 优先加载 SCI/CIFAR100/cifar100_similarity_matrix.npy
        npy_path = os.path.join('SCI', 'CIFAR100', 'cifar100_similarity_matrix.npy')
        if os.path.exists(npy_path):
            semantic_matrix = np.load(npy_path)
            dataset_labels = None
        else:
            from cifar100_semantic import CIFAR100_LABELS, CIFAR100_SEMANTIC_SIMILARITY
            dataset_labels = CIFAR100_LABELS
            semantic_matrix = CIFAR100_SEMANTIC_SIMILARITY
        dataset_cls = CIFAR100
        normalize_mean, normalize_std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]

    # Tiny-ImageNet 语义矩阵动态计算回退 (WordNet Wu-Palmer)
    if args.dataset == 'tinyimagenet' and semantic_matrix is None:
        print("  🔧 在线计算 Tiny-ImageNet Wu-Palmer 语义矩阵...")
        from tinyimagenet_semantic import read_tinyimagenet_wnids, compute_wup_similarity_matrix
        wnid_list, _ = read_tinyimagenet_wnids(args.data_path)
        semantic_matrix = compute_wup_similarity_matrix(wnid_list, verbose=False)
        os.makedirs(os.path.join('SCI', 'TinyImageNet'), exist_ok=True)
        np.save(os.path.join('SCI', 'TinyImageNet', 'tinyimagenet_similarity_matrix.npy'), semantic_matrix)
        print(f"  💾 语义矩阵已缓存到 SCI/TinyImageNet/tinyimagenet_similarity_matrix.npy")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # 模型
    print("🔧 加载模型...")
    if args.model == 'crate':
        model = crate.CRATE_test(num_classes=num_classes, image_size=image_size)
    else:
        model = vit_cab_tiny_12_p16()
    model = load_pretrained_weights(model, args.checkpoint, device=args.device)
    model = model.to(args.device).eval()

    # 数据
    print(f"🔧 加载 {args.dataset} 数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    if args.dataset == 'tinyimagenet':
        # Tiny-ImageNet: ImageFolder 读取 train/<wnid>/images/*.JPEG
        train_dir = os.path.join(args.data_path, 'train')
        print(f"  ImageFolder root: {train_dir}")
        dataset = ImageFolder(root=train_dir, transform=transform)
        # 按 wnids.txt 顺序重排样本，使类索引与语义矩阵对齐
        wnids_file = os.path.join(args.data_path, 'wnids.txt')
        if os.path.exists(wnids_file):
            with open(wnids_file) as f:
                wnids = [l.strip() for l in f if l.strip()]
            wnid_to_idx = {w: i for i, w in enumerate(wnids)}
            dataset.samples.sort(key=lambda s: wnid_to_idx.get(s[0].split(os.sep)[-2], -1))
            dataset.targets = [s[1] for s in dataset.samples]
            print(f"  按 wnids.txt 重排样本顺序 (共 {len(wnids)} 个 WNID)")
    else:  # cifar10 / cifar100
        dataset = dataset_cls(root=args.data_path, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"✅ 数据集加载完成，共 {len(dataset)} 张图片, {num_classes} 个类别")

    # 类中心 & 距离矩阵
    print("🔧 计算类中心...")
    class_centers = compute_class_centers(model, dataloader, num_classes=num_classes, device=args.device)
    distance_matrix = compute_distance_matrix(class_centers)
    print(f"✅ 类中心 {class_centers.shape}, 距离矩阵 {distance_matrix.shape}")

    # 样本特征用于 t-SNE
    print("🔧 提取样本特征...")
    all_features, all_labels = extract_all_class_features(
        model, dataloader, num_classes=num_classes, samples_per_class=args.samples_per_class, device=args.device)
    print(f"✅ 共 {len(all_features)} 个样本")

    # 可视化
    # plot_tsne(all_features, all_labels, checkpoint_name, semantic_matrix, dataset_labels, num_classes)

    # 距离矩阵
    # print_distance_matrix(distance_matrix, dataset_labels, num_classes)

    # 全局 SCI
    all_idx = list(range(num_classes))
    sci, p = compute_sci(semantic_matrix, distance_matrix, all_idx)
    print("\n" + "=" * 50)
    print(f"📈 {args.dataset} 语义连续性指数 (SCI)")
    print("=" * 50)
    print_sci_analysis(sci, p, f"{args.dataset} SCI")

    # CIFAR-10 额外分析
    if args.dataset == 'cifar10':
        # 局部分析: 猫-狗-鸟
        cat, dog, bird = 3, 5, 2
        cd, cb, db = distance_matrix[cat, dog], distance_matrix[cat, bird], distance_matrix[dog, bird]
        print("\n" + "=" * 50)
        print("🔍 重点分析：猫 vs 狗 vs 鸟")
        print("=" * 50)
        print(f"特征距离: 猫-狗={cd:.4f}, 猫-鸟={cb:.4f}, 狗-鸟={db:.4f}")
        local_sci, _ = compute_sci(CIFAR10_SEMANTIC_SIMILARITY, distance_matrix, [cat, dog, bird])
        print(f"📍 局部 SCI (猫-狗-鸟): {local_sci:.4f}")

        # CIFAR10H SCI
        sci_h, p_h = compute_sci(CIFAR10H_SEMANTIC_SIMILARITY, distance_matrix, all_idx)
        print_sci_analysis(sci_h, p_h, "CIFAR10H 矩阵 SCI")
        print(f"\n📊 对比: 主观={sci:.4f} | CIFAR10H={sci_h:.4f}")

        # CIFAR10-H 混淆矩阵
        print_cifar10h_confusion()
    else:
        local_sci = 0.0
        sci_h = 0.0
        p_h = 0.0

    # 保存结果
    np.savez(os.path.join(args.output_dir, f'{args.dataset}_{checkpoint_name}_results.npz'),
             class_centers=class_centers, distance_matrix=distance_matrix,
             sci=sci, sci_p=p, sci_h=sci_h, sci_h_p=p_h, local_sci=local_sci)
    print(f"\n💾 结果已保存至 {args.output_dir}/{args.dataset}_{checkpoint_name}_results.npz")


if __name__ == "__main__":
    main()
