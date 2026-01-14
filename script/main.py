import os
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPVisionModel
import torchvision.models as models

import albumentations as A
from albumentations.augmentations.dropout import CoarseDropout

# ---------------------------------------------------------------------
# 0. 全局设置：随机种子、设备
# ---------------------------------------------------------------------

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 1. 路径与 YOLO 检测 + 裁剪
# ---------------------------------------------------------------------

# TODO：根据你自己的数据路径修改
image_folder = r"D:/yolo/YWaddC/yoloworld_clip/prepare_booktest/09_test"
cropped_images_folder = r"D:/yolo/YWaddC/yoloworld_clip/cut_img"

os.makedirs(cropped_images_folder, exist_ok=True)

# 加载 YOLO-World 模型（与你论文一致）
model_yolo = YOLO(r"D:/yolo/YWaddC/yolov8l-world.pt")

def detect_and_crop_yolo(image_path: str,
                         model: YOLO,
                         save_folder: str,
                         conf: float = 0.05,
                         iou: float = 0.20) -> str | None:
    """
    用 YOLO-World 对整幅 1080p 图像进行检测，
    选择面积最大的框进行裁剪并保存，返回裁剪后文件名。
    """
    image = Image.open(image_path).convert("RGB")

    # 推理（关闭 augment，以与你论文的阈值设定一致）
    results = model(image_path, conf=conf, iou=iou)

    boxes = []
    for result in results:
        if result.boxes is not None:
            for detection in result.boxes:
                boxes.append(detection.xyxy[0].cpu().numpy())

    if not boxes:
        print(f"未检测到目标：{image_path}")
        return None

    # 选取面积最大的框（与论文描述一致）
    best_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = [int(v) for v in best_box]

    cropped_img = image.crop((x1, y1, x2, y2))
    base = os.path.splitext(os.path.basename(image_path))[0]
    cropped_img_filename = f"{base}_best.jpg"
    cropped_img.save(os.path.join(save_folder, cropped_img_filename))

    return cropped_img_filename

# 只需在第一次运行时做一次裁剪；以后可以注释掉这段。
def prepare_crops():
    print("开始 YOLO-World 检测并裁剪 ROI ...")
    num_crops = 0
    for img_filename in os.listdir(image_folder):
        if img_filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_folder, img_filename)
            crop_name = detect_and_crop_yolo(img_path, model_yolo, cropped_images_folder)
            if crop_name is not None:
                num_crops += 1
    print(f"裁剪完成，共生成 {num_crops} 张 ROI 图像。")

# ---------------------------------------------------------------------
# 2. 数据增强与数据集
# ---------------------------------------------------------------------

def advanced_augment_image(image: Image.Image) -> Image.Image:
    """Albumentations 数据增强，用于训练阶段。"""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.MotionBlur(blur_limit=3, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.5),
        CoarseDropout(max_holes=8, max_height=8, max_width=8,
                      min_holes=1, p=0.5),
    ])
    image_np = np.array(image)
    augmented = transform(image=image_np)
    return Image.fromarray(augmented["image"])

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class ImageSimilarityDataset(Dataset):
    """
    将原图与对应的裁剪图成对加载：
    - pairs: (original_filename, cropped_filename)
    - augment: 是否使用 Albumentations 随机增强（训练用 True，验证/测试用 False）
    - triplet_mode: 是否返回 (anchor, positive, negative) 三元组
    """
    def __init__(self,
                 original_folder: str,
                 cropped_folder: str,
                 processor: CLIPProcessor,
                 augment: bool = False,
                 triplet_mode: bool = False):
        super().__init__()
        self.original_folder = original_folder
        self.cropped_folder = cropped_folder
        self.processor = processor
        self.augment = augment
        self.triplet_mode = triplet_mode

        # 用文件名匹配原图和 *_best.jpg，避免 listdir 顺序带来的错配
        self.pairs: list[tuple[str, str]] = []
        for fname in os.listdir(original_folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                stem, _ = os.path.splitext(fname)
                crop_name = f"{stem}_best.jpg"
                crop_path = os.path.join(cropped_folder, crop_name)
                if os.path.exists(crop_path):
                    self.pairs.append((fname, crop_name))

        self.pairs.sort()  # 确保顺序固定，便于复现

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        orig_name, crop_name = self.pairs[idx]
        original_img_path = os.path.join(self.original_folder, orig_name)
        cropped_img_path = os.path.join(self.cropped_folder, crop_name)

        original_img = Image.open(original_img_path).convert("RGB")
        cropped_img = Image.open(cropped_img_path).convert("RGB")

        # 训练集才做增强，验证/测试集保持原始图像（与论文描述一致）
        if self.augment:
            original_img = advanced_augment_image(original_img)
            cropped_img = advanced_augment_image(cropped_img)

        original_img_processed = self.processor(
            images=original_img, return_tensors="pt"
        )["pixel_values"].squeeze(0)  # (3, 224, 224)

        cropped_img_processed = self.processor(
            images=cropped_img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        if not self.triplet_mode:
            # Siamese 模式：返回 (anchor, positive)
            return original_img_processed, cropped_img_processed

        # Triplet 模式：构造一个随机负样本
        # 负样本只从裁剪图里选即可
        neg_idx = random.choice([i for i in range(len(self.pairs)) if i != idx])
        _, neg_crop_name = self.pairs[neg_idx]
        negative_img_path = os.path.join(self.cropped_folder, neg_crop_name)
        negative_img = Image.open(negative_img_path).convert("RGB")
        if self.augment:
            negative_img = advanced_augment_image(negative_img)
        negative_img_processed = self.processor(
            images=negative_img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        # 对正样本做 Mixup（与论文描述一致）
        lam = np.random.beta(1.0, 1.0)
        combined_positive = lam * cropped_img_processed + (1.0 - lam) * negative_img_processed

        return original_img_processed, combined_positive, negative_img_processed

# ---------------------------------------------------------------------
# 3. Enhanced Siamese Network (ESN) 模型
# ---------------------------------------------------------------------

class EnhancedSiameseNetwork(nn.Module):
    """
    对应论文中 ESN 的实现：
    - EfficientNet-B4 CNN 分支
    - CLIP ViT-B/32 视觉编码器分支
    - Patch + Multi-head Self-Attention + 6 层 Transformer Encoder
    - 两层全连接 (512 -> 256)，Dropout=0.5
    - 输出 256 维 L2 归一化特征
    """
    def __init__(self, patch_size: int = 16):
        super().__init__()

        # EfficientNet-B4 作为 CNN 特征提取
        self.cnn = models.efficientnet_b4(pretrained=True)
        cnn_out_feat = self.cnn.classifier[-1].in_features
        self.cnn.classifier = nn.Identity()

        # CLIP Vision Encoder (ViT-B/32)
        self.clip_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        for p in self.clip_model.parameters():
            p.requires_grad = True
        clip_out_feat = self.clip_model.config.hidden_size  # 768

        # Patch + Transformer
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size
        self.embedding = nn.Linear(self.patch_dim, 768)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=8, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 全连接融合：Transformer(768) + CNN + CLIP
        fused_dim = 768 + cnn_out_feat + clip_out_feat
        self.fc1 = nn.Linear(fused_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        p = self.patch_size

        # (B, C, H/p, W/p, p, p)
        x = x.unfold(2, p, p).unfold(3, p, p)
        # (B, C, num_patches_h, num_patches_w, p*p)
        x = x.contiguous().view(B, C, -1, p * p)
        # (B, num_patches, C * p*p)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, C * p * p)
        return x  # (B, N, patch_dim)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        # CNN 分支
        cnn_features = self.cnn(x)  # (B, cnn_out_feat)

        # CLIP 分支
        clip_features = self.clip_model(pixel_values=x).pooler_output  # (B, clip_out_feat)

        # Patch + Transformer 分支
        patches = self.extract_patches(x)           # (B, N, patch_dim)
        patch_embeddings = self.embedding(patches)  # (B, N, 768)

        attn_output, _ = self.multihead_attn(
            patch_embeddings, patch_embeddings, patch_embeddings
        )  # (B, N, 768)

        transformer_output = self.transformer(attn_output)  # (B, N, 768)
        transformer_feature = transformer_output.mean(dim=1)  # (B, 768)

        # 融合三个分支
        combined_feature = torch.cat(
            (transformer_feature, cnn_features, clip_features), dim=1
        )  # (B, fused_dim)

        x = F.relu(self.fc1(combined_feature))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # L2 归一化
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_once(x)

# ---------------------------------------------------------------------
# 4. Triplet ArcFace Loss（与论文一致）
# ---------------------------------------------------------------------

class TripletArcFaceLoss(nn.Module):
    def __init__(self, margin: float = 0.5, scale: float = 30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        cos_sim_apos = F.cosine_similarity(anchor, positive)
        cos_sim_aneg = F.cosine_similarity(anchor, negative)

        # ArcFace 角度空间加 margin
        arc_cos_apos = torch.acos(cos_sim_apos.clamp(-1 + 1e-7, 1 - 1e-7)) + self.margin
        arc_cos_aneg = torch.acos(cos_sim_aneg.clamp(-1 + 1e-7, 1 - 1e-7))

        arc_loss = F.relu(arc_cos_apos - arc_cos_aneg).mean() * self.scale
        contrastive_loss = F.relu(arc_cos_apos - arc_cos_aneg).mean()
        return arc_loss + contrastive_loss

# ---------------------------------------------------------------------
# 5. 训练与验证 ESN
# ---------------------------------------------------------------------

def train_esn():
    # 准备数据集：训练集做增强，验证集不做增强（这里示例用同一文件夹，你可以换成 train/val 两个文件夹）
    train_dataset = ImageSimilarityDataset(
        image_folder, cropped_images_folder,
        processor=processor, augment=True, triplet_mode=True
    )
    val_dataset = ImageSimilarityDataset(
        image_folder, cropped_images_folder,
        processor=processor, augment=False, triplet_mode=True
    )

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # 初始化模型、损失、优化器（与论文 4.2 参数一致）
    esn = EnhancedSiameseNetwork().to(device)
    criterion = TripletArcFaceLoss(margin=0.5, scale=30.0)
    optimizer = optim.Adam(esn.parameters(), lr=5e-5, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    num_epochs = 40
    for epoch in range(num_epochs):
        esn.train()
        running_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_feat = esn(anchor)
            positive_feat = esn(positive)
            negative_feat = esn(negative)

            loss = criterion(anchor_feat, positive_feat, negative_feat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(esn.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        # 验证集 loss
        esn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                anchor_feat = esn(anchor)
                positive_feat = esn(positive)
                negative_feat = esn(negative)

                loss = criterion(anchor_feat, positive_feat, negative_feat)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    return esn

# ---------------------------------------------------------------------
# 6. 推理阶段：相似度计算 + 运行时间测量
# ---------------------------------------------------------------------

def measure_runtime(esn: EnhancedSiameseNetwork,
                    num_warmup: int = 10,
                    max_images: int | None = None):
    """
    在测试集上测 YOLO 检测+裁剪、CLIP 预处理、ESN forward 三部分的平均时间。
    """
    esn.eval()
    model_yolo.eval()

    # 构造一个不带增强、不带 triplet 的测试集
    test_dataset = ImageSimilarityDataset(
        image_folder, cropped_images_folder,
        processor=processor, augment=False, triplet_mode=False
    )

    if max_images is not None:
        indices = list(range(min(max_images, len(test_dataset))))
    else:
        indices = list(range(len(test_dataset)))

    # 预热若干次，避免首批不稳定
    for i in range(min(num_warmup, len(indices))):
        idx = indices[i]
        img_name, _ = test_dataset.pairs[idx]
        img_path = os.path.join(image_folder, img_name)
        _ = detect_and_crop_yolo(img_path, model_yolo, cropped_images_folder)

    yolo_times = []
    clip_times = []
    esn_times = []

    for idx in indices:
        img_name, crop_name = test_dataset.pairs[idx]
        img_path = os.path.join(image_folder, img_name)

        # --- 1) YOLO-World 检测 + 裁剪 ---
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t0 = time.perf_counter()
        crop_name = detect_and_crop_yolo(img_path, model_yolo,
                                         cropped_images_folder)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t1 = time.perf_counter()
        if crop_name is None:
            continue
        yolo_times.append((t1 - t0) * 1000.0)

        crop_path = os.path.join(cropped_images_folder, crop_name)
        crop_img = Image.open(crop_path).convert("RGB")

        # --- 2) CLIP 预处理 ---
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t2 = time.perf_counter()
        crop_tensor = processor(
            images=crop_img, return_tensors="pt"
        )["pixel_values"].to(device)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t3 = time.perf_counter()
        clip_times.append((t3 - t2) * 1000.0)

        # --- 3) ESN 前向 ---
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t4 = time.perf_counter()
        with torch.no_grad():
            _ = esn(crop_tensor)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t5 = time.perf_counter()
        esn_times.append((t5 - t4) * 1000.0)

    def mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    yolo_ms = mean(yolo_times)
    clip_ms = mean(clip_times)
    esn_ms = mean(esn_times)
    total_ms = yolo_ms + clip_ms + esn_ms

    print("\n=== Runtime analysis (per image, ms) ===")
    print(f"YOLO-World detection + cropping: {yolo_ms:.2f} ms")
    print(f"CLIP preprocessing:              {clip_ms:.2f} ms")
    print(f"ESN forward pass:                {esn_ms:.2f} ms")
    print(f"Total pipeline:                  {total_ms:.2f} ms")
    print(f"Approx. throughput:              {1000.0 / total_ms:.2f} FPS")


# ---------------------------------------------------------------------
# 7. 主入口
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # 第一次运行时解开这一行，做一次 ROI 裁剪；之后可以注释掉。
    # prepare_crops()

    # 训练 ESN（如果你已经有训练好的权重，可以直接加载，跳过这一步）
    esn_model = train_esn()

    # 在测试集上测运行时间
    measure_runtime(esn_model, num_warmup=10, max_images=100)
