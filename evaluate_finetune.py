############
## Import ##
############
import argparse
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import encoder
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import os
import logging
from PIL import Image


# 各数据集配置（与 simclr_train.py 的 DATASET_CONFIG 对齐）
DATASET_CONFIG = {
    "cifar10": dict(
        root="E:/DATA/",
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],
        image_size=32,
        num_classes=10, epochs=50,
    ),
    "cifar100": dict(
        root="E:/DATA/",
        mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762],
        image_size=32,
        num_classes=100, epochs=100,  # 类别更多, 适当延长微调
    ),
    "tinyimagenet": dict(
        root="E:/DATA/TinyImgenet",
        mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821],
        image_size=64,
        num_classes=200, epochs=60,  # 200 类, 微调比 CIFAR 适当延长
    ),
}

######################
## Parsing Argument ##
######################
parser = argparse.ArgumentParser(description='End-to-End Finetune Evaluation (aligned with simclr_train.py)')

parser.add_argument('--data', type=str, default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet"],
                    help='dataset (default: cifar10)')
parser.add_argument('--arch', type=str, default="crate-test",
                    help='network architecture')
parser.add_argument('--z_dim', type=int, default=128,
                    help='projection dim, MUST match pretraining (simclr_train.py default: 128)')
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='hidden dim, MUST match pretraining (simclr_train.py default: 512)')
parser.add_argument('--use_hyperbolic', action='store_true', default=False,
                    help='pretraining used hyperbolic projection head (双曲预训练权重必须开启, 否则 projection 键不匹配)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='peak learning rate for SGD finetune (default: 0.01)')
parser.add_argument('--epochs', type=int, default=None,
                    help='number of finetune epochs (默认按数据集配置: cifar10=50, cifar100=100)')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='linear warmup epochs (default: 5)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size for finetune (default: 256)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--model_path', type=str,
                    default="./logs/EMP-SSL-Training/CRATE/Hyper/Direct/best.pt",
                    help='model path for eval')
parser.add_argument('--save_dir', type=str, default="./crate_finetune_logs",
                    help='directory to save logs and models')

args = parser.parse_args()

# 未显式指定的参数回退到对应数据集的推荐配置 (CLI 传入值优先)
_cfg = DATASET_CONFIG[args.data]
if args.epochs is None:
    args.epochs = _cfg["epochs"]
# cifar100 等非 cifar10 数据集的微调结果存独立目录, 避免覆盖 cifar10 结果
if args.data != "cifar10" and not args.save_dir.endswith(f"_{args.data}"):
    args.save_dir = f"{args.save_dir}_{args.data}"

######################
## Helper Functions ##
######################
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    log_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'finetune.log')),
            logging.StreamHandler()
        ],
    )
    # 兼容 Python 3.7 (basicConfig 无 force 参数): 重复调用时手动移除旧 handlers
    logger = logging.getLogger(__name__)
    # 确保 root logger 也会正确 propagate
    root = logging.getLogger()
    if len(root.handlers) > 2:
        # 移除重复的 handlers (只保留本次新添加的两个)
        to_remove = root.handlers[:-2]
        for h in to_remove:
            root.removeHandler(h)
    return logger


class TinyImageNetVal(torch.utils.data.Dataset):
    """Tiny-ImageNet 验证集: 图像平铺在 val/images/, 标签从 val_annotations.txt 读取
    (文件名 \\t wnid \\t bbox_*), 标签索引由 wnids.txt 顺序决定, 与 ImageFolder train 保持一致.
    """
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, "val", "images")
        self.transform = transform

        # 读 wnids.txt 决定 200 个类的索引顺序 (与 ImageFolder 按字母顺序一致)
        with open(os.path.join(root, "wnids.txt"), "r") as f:
            wnid_list = [line.strip() for line in f if line.strip()]
        self.wnid_to_idx = {w: i for i, w in enumerate(wnid_list)}

        # 读验证集标注
        self.samples = []
        with open(os.path.join(root, "val", "val_annotations.txt"), "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    fname, wnid = parts[0], parts[1]
                    self.samples.append((fname, self.wnid_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

######################
## End-to-End Finetune ##
######################
def finetune(net, train_loader, test_loader, num_classes, save_dir, logger):
    # 分类头: 输入维度 = hidden_dim (与 simclr_train.py 预训练模型的 pre_feature 输出一致)
    classifier = nn.Linear(args.hidden_dim, num_classes).cuda()

    # 优化器: SGD + nesterov (CIFAR10 监督微调标准配置)
    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # 学习率调度: linear warmup -> cosine annealing (与 simclr_train.py 一致)
    def lr_lambda(step):
        warmup_steps = args.warmup_epochs * len(train_loader)
        total_steps = args.epochs * len(train_loader)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = torch.nn.CrossEntropyLoss()

    best_test_acc = 0.0
    model_dir = os.path.join(save_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"Start finetuning for {args.epochs} epochs")
    logger.info(f"Initial learning rate: {args.lr}, warmup: {args.warmup_epochs} epochs")
    logger.info(f"Batch size: {args.batch_size}")

    # 混合精度 (与 simclr_train.py 一致)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(args.epochs):
        net.train()
        classifier.train()
        top1_train_accuracy = 0
        total_train = 0
        train_loss = 0.0

        for x, y in tqdm(train_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # 提取 pre_feature 输出特征 (hidden_dim 维) 用于分类
                _, z_pre = net(x, is_test=True)
                logits = classifier(z_pre)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * y.size(0)
            top1 = accuracy(logits, y, topk=(1,))
            top1_train_accuracy += top1[0].item() * y.size(0)
            total_train += y.size(0)

        train_acc = top1_train_accuracy / total_train
        avg_train_loss = train_loss / total_train

        # 测试
        net.eval()
        classifier.eval()
        top1_test_accuracy = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                _, z_pre = net(x, is_test=True)
                logits = classifier(z_pre)

                loss = criterion(logits, y)
                test_loss += loss.item() * y.size(0)
                top1 = accuracy(logits, y, topk=(1,))
                top1_test_accuracy += top1[0].item() * y.size(0)
                total_test += y.size(0)

        test_acc = top1_test_accuracy / total_test
        avg_test_loss = test_loss / total_test

        # 打印和记录
        log_msg = f"Epoch {epoch+1}/{args.epochs} | " \
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | " \
                  f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        print(log_msg)
        logger.info(log_msg)

        # 保存最新模型
        latest_model_path = os.path.join(model_dir, 'latest.pt')
        torch.save({
            'net': net.state_dict(),
            'classifier': classifier.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc
        }, latest_model_path)

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(model_dir, 'best.pt')
            torch.save({
                'net': net.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc
            }, best_model_path)
            logger.info(f"New best model saved! Test Acc: {best_test_acc:.2f}%")

    logger.info(f"Finetuning finished. Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")

######################
## Main Execution ##
######################
if __name__ == '__main__':
    # 设置日志
    logger = setup_logging(args.save_dir)

    num_classes = _cfg["num_classes"]
    img_size = _cfg["image_size"]

    # 数据管线与 simclr_train.py 对齐:
    # - normalize 参数按数据集自动匹配 (与预训练一致)
    # - image_size 按数据集: CIFAR=32, Tiny-ImageNet=64 (与预训练输入尺寸保持一致)
    # - 微调用标准监督增强 (RandomCrop+Flip + padding=image_size/8), 测试用确定性 transform
    normalize = transforms.Normalize(_cfg["mean"], _cfg["std"])
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=img_size // 8,
                              padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.data == "tinyimagenet":
        # train: ImageFolder 读取 <root>/train/<wnid>/images/*.JPEG (200 类)
        # val:   自定义 TinyImageNetVal (图片平铺 + val_annotations.txt 映射, 索引与 ImageFolder 对齐)
        train_dataset = ImageFolder(os.path.join(_cfg["root"], "train"), transform=train_transform)
        test_dataset = TinyImageNetVal(_cfg["root"], transform=test_transform)
    else:
        dataset_cls = CIFAR10 if args.data == "cifar10" else CIFAR100
        train_dataset = dataset_cls(root=_cfg["root"], train=True, download=False, transform=train_transform)
        test_dataset = dataset_cls(root=_cfg["root"], train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 关键: encoder 的 z_dim/hidden_dim/image_size 必须与预训练完全一致, 否则权重加载会失败
    # (不同 image_size 下 pos_embedding 形状不同, 一定不兼容)
    net = encoder(arch=args.arch, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                  use_hyperbolic=args.use_hyperbolic, image_size=img_size)
    net = nn.DataParallel(net)

    if args.model_path:
        save_dict = torch.load(args.model_path, map_location=device)
        # 预训练模型同样以 DataParallel 保存, 键应完全匹配
        # 显式检查 missing/unexpected, 避免 strict=False 静默吞掉加载失败
        missing, unexpected = net.load_state_dict(save_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        if not missing and not unexpected:
            logger.info("All pretrained weights loaded successfully")
        else:
            logger.warning("Some weights were NOT loaded - check z_dim/hidden_dim/image_size match pretraining!")
        logger.info(f"Loaded pretrained model from {args.model_path}")

    net.cuda()

    logger.info(f"Dataset: {args.data} | image_size: {img_size}x{img_size} | num_classes: {num_classes}")
    logger.info(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # 端到端微调
    finetune(net, train_loader, test_loader, num_classes, args.save_dir, logger)
