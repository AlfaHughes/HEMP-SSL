############
## Import ##
############
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from model.model import encoder
from dataset.datasets import load_dataset
import numpy as np
import math
import torch.nn.functional as F
from tqdm import tqdm
import torch
from loss import TotalCodingRate
from func import chunk_avg
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast

######################
## Parsing Argument ##
######################
parser = argparse.ArgumentParser(description='EMP-SSL Original Training (Euclidean loss only, optional hyperbolic projection head)')

parser.add_argument('--patch_sim', type=float, default=200,
                    help='coefficient of similarity loss (default: 200)')
parser.add_argument('--tcr', type=int, default=1,
                    help='coefficient of tcr (default: 1)')
parser.add_argument('--num_patches', type=int, default=100,
                    help='number of patches used in EMP-SSL (default: 100)')
parser.add_argument('--arch', type=str, default="crate-test",
                    choices=['resnet18-cifar', 'resnet18-tinyimagenet', 'resnet18-imagenet', 'crate-test'],
                    help='network architecture (default: crate-test)')
parser.add_argument('--bs', type=int, default=100,
                    help='batch size (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for AdamW (default: 1e-3)')
parser.add_argument('--eps', type=float, default=0.2,
                    help='eps for TCR (default: 0.2)')
parser.add_argument('--msg', type=str, default="NONE",
                    help='additional message for description (default: NONE)')
parser.add_argument('--dir', type=str, default="EMP-SSL-Training",
                    help='directory name (default: EMP-SSL-Training)')
parser.add_argument('--data', type=str, default="cifar10",
                    choices=['cifar10', 'cifar100', 'tinyimagenet', 'imagenet100', 'imagenet'],
                    help='data (default: cifar10)')
parser.add_argument('--epoch', type=int, default=30,
                    help='max number of epochs to finish (default: 30)')
parser.add_argument('--warmup_epoch', type=int, default=5,
                    help='linear warmup epochs before cosine decay (default: 5)')
parser.add_argument('--use_hyperbolic_proj', action='store_true', default=False,
                    help='use hyperbolic projection head in encoder (loss stays Euclidean, default: False)')
parser.add_argument('--hyp_curvature', type=float, default=0.3,
                    help='curvature for hyperbolic projection head (default: 0.3)')
parser.add_argument('--hyp_hidden_dim', type=int, default=512,
                    help='hidden dim of projection head (default: 512)')
parser.add_argument('--hyp_out_dim', type=int, default=128,
                    help='output dim of projection head (default: 128)')
parser.add_argument('--data_path', type=str, default="./DATA",
                    help='base path for datasets (default: ./DATA)')
parser.add_argument('--use_class_pair', type=str, default='False',
                    choices=['True', 'False'],
                    help='use class pair sampling: each batch contains pairs of same-class samples')

args = parser.parse_args()

# 将字符串转换为布尔值
args.use_class_pair = args.use_class_pair == 'True'

print(args)

num_patches = args.num_patches
dir_name = f"./logs/{args.dir}/patchsim{args.patch_sim}_numpatch{args.num_patches}_bs{args.bs}_lr{args.lr}_{args.msg}"


#####################
## Helper Function ##
#####################

def chunk_avg(x, n_chunks=2, normalize=False):
    x_list = x.chunk(n_chunks, dim=0)
    x = torch.stack(x_list, dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0), dim=1)


class Similarity_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)

        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()

        z_sim = z_sim / num_patch
        z_sim_out = z_sim.clone().detach()

        return -z_sim, z_sim_out


def cal_TCR(z, criterion, num_patches):
    z_list = z.chunk(num_patches, dim=0)
    loss = 0
    # TCR 的 logdet 对 fp16 敏感，强制 fp32 计算
    with torch.autocast('cuda', enabled=False):
        for i in range(num_patches):
            loss += criterion(z_list[i].float())
    loss = loss / num_patches
    return loss


######################
## Prepare Training ##
######################
torch.multiprocessing.set_sharing_strategy('file_system')

# 根据数据集设置图像尺寸
if args.data in ["cifar10", "cifar100"]:
    image_size = 32
elif args.data == "tinyimagenet":
    image_size = 64
elif args.data == "imagenet":
    image_size = 224
else:
    image_size = 32

# 根据数据集自动选择合适的模型架构（如果用户未指定）
if args.arch == "crate-test":
    pass
elif args.data == "cifar10" or args.data == "cifar100":
    if args.arch != "resnet18-cifar":
        print(f"Warning: Data is {args.data}, but arch is {args.arch}. Changing to resnet18-cifar.")
        args.arch = "resnet18-cifar"
elif args.data == "tinyimagenet":
    if args.arch != "resnet18-tinyimagenet":
        print(f"Warning: Data is {args.data}, but arch is {args.arch}. Changing to resnet18-tinyimagenet.")
        args.arch = "resnet18-tinyimagenet"
elif args.data == "imagenet":
    if args.arch != "resnet18-imagenet":
        print(f"Warning: Data is {args.data}, but arch is {args.arch}. Changing to resnet18-imagenet.")
        args.arch = "resnet18-imagenet"

# 加载数据集
train_dataset = load_dataset(args.data, train=True, num_patch=num_patches, path=args.data_path, use_class_pair=args.use_class_pair)

# 数据加载器
dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=0)
if args.use_class_pair:
    print("Using ClassPair mode: 2 same-class images, each augmented to 50 views, interleaved to 100 views")


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


# 关键区别：use_hyperbolic_proj 控制投影头是否走双曲映射 (expmap0)
# 但 loss 始终用欧氏 Similarity_Loss + TotalCodingRate
net = encoder(arch=args.arch,
              use_hyperbolic=args.use_hyperbolic_proj,
              curvature=args.hyp_curvature,
              hidden_dim=args.hyp_hidden_dim,
              z_dim=args.hyp_out_dim,
              image_size=image_size)
net = nn.DataParallel(net)
net.cuda()
# 计算并打印模型参数量
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
proj_type = "Hyperbolic (expmap0)" if args.use_hyperbolic_proj else "Euclidean (L2 normalize)"
print(f"Dataset: {args.data}, Architecture: {args.arch}, Image size: {image_size}x{image_size}")
print(f"Projection head: {proj_type}, Loss: Euclidean (Similarity + TCR), patch_sim={args.patch_sim}")


# AdamW 参数分组：权重做 weight decay，BN 参数/bias/标量不做
decay_params, no_decay_params = [], []
for name, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim <= 1 or name.endswith('.bias'):
        no_decay_params.append(p)
    else:
        decay_params.append(p)
opt = optim.AdamW([
    {'params': decay_params, 'weight_decay': 1e-4},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=args.lr)
print(f"Optimizer: AdamW (lr={args.lr}, wd=1e-4 on {len(decay_params)} tensors, wd=0 on {len(no_decay_params)} tensors)")

scaler = GradScaler()

# 线性 warmup + 余弦退火 (epoch 级)
def lr_lambda(epoch):
    if epoch < args.warmup_epoch:
        return (epoch + 1) / max(1, args.warmup_epoch)
    progress = (epoch - args.warmup_epoch) / max(1, args.epoch - args.warmup_epoch)
    eta_min_ratio = 1e-5 / args.lr
    return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = lr_scheduler.LambdaLR(opt, lr_lambda)

# Loss: 始终使用欧氏 loss，不因投影头类型而改变
print("使用标准 Euclidean loss (Similarity_Loss + TotalCodingRate)")
contractive_loss = Similarity_Loss()
criterion = TotalCodingRate(eps=args.eps)


##############
## Training ##
##############
def main():
    best_loss = float('inf')
    for epoch in range(args.epoch):
        for step, (data, label) in tqdm(enumerate(dataloader)):
            opt.zero_grad()

            data = torch.cat(data, dim=0)
            data = data.cuda()
            # 真 AMP：前向 + loss 在 autocast 下执行 (TCR 内部强制 fp32)
            with autocast():
                z_proj = net(data)

                z_list = z_proj.chunk(num_patches, dim=0)
                # 始终用欧氏 chunk_avg（不走 Möbius 均值）
                z_avg = chunk_avg(z_proj, num_patches)

                # Contractive Loss (Euclidean cosine similarity)
                loss_contract, _ = contractive_loss(z_list, z_avg)
                loss_TCR = cal_TCR(z_proj, criterion, num_patches)

            loss = args.patch_sim * loss_contract + args.tcr * loss_TCR

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

        scheduler.step()

        total_loss = args.patch_sim * loss_contract + args.tcr * loss_TCR
        loss_sim_contrib = args.patch_sim * loss_contract

        model_dir = dir_name + "/save_models/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # 只保存总损失最低的权重
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(net.state_dict(), model_dir + "best.pt")
            print(f"New best model saved at epoch {epoch}, total_loss: {best_loss:.4f}")

        print("At epoch:", epoch,
              "| loss_sim:", loss_contract.item(),
              "| loss_TCR:", loss_TCR.item(),
              "| loss_sim_weighted:", loss_sim_contrib.item(),
              "| total_loss:", total_loss.item(),
              "| lr:", opt.param_groups[0]['lr'],
              "| hyp_proj" if args.use_hyperbolic_proj else "")


if __name__ == '__main__':
    main()
