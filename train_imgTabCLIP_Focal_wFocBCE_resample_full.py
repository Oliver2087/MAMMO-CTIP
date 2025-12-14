# ============================================================ 
# 0. 全局导入 & 公共工具（FocalLoss 等）
# ============================================================
import os
import math
import pandas as pd
import numpy as np
import argparse
import json
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision import transforms, models
from PIL import Image

from tqdm import tqdm
import psutil
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score,
)

# ===================== class-wise alpha 配置区 =====================
# 这里单独调每个病灶的 α，默认都设成 0.25 和原来一致；
# 想更关注某类，就把那一类的 α 调大一些，比如 cancer: 0.5 或 0.75
CLASS_ALPHA_CONFIG = {
    "mass": 0.25,
    "calc": 0.25,
    "cancer": 0.25,
}


# ----------------- Focal Loss（用于对比学习部分） -----------------
class FocalLoss(nn.Module):
    """
    针对多类分类 logits 的 Focal Loss，用在 InfoNCE 对比损失上：
    - logits: [N, C]
    - targets: [N]，值在 [0, C-1]
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets_onehot = F.one_hot(targets, num_classes=logits.shape[-1]).float()

        pt = (probs * targets_onehot).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        alpha_t = self.alpha * targets_onehot + (1 - self.alpha) * (1 - targets_onehot)
        alpha_t = (alpha_t * targets_onehot).sum(dim=-1)

        ce_loss = -(log_probs * targets_onehot).sum(dim=-1)
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ----------------- Focal Loss（用于BCE部分） -----------------
class FocalBCELoss(nn.Module):
    """
    Multi-label Focal BCE Loss
    pred: [B, C]
    target: [B, C] (0/1)

    alpha 可以是：
      - float：对所有类相同
      - list / np.array / 1D tensor：per-class alpha，长度 = C
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        # 允许 alpha 是标量或 per-class 向量
        if isinstance(alpha, (list, tuple, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits, targets):
        """
        pred_logits: sigmoid BEFORE activation (raw logits) [B, C]
        targets: binary {0,1} [B, C]
        """
        pred_sigmoid = torch.sigmoid(pred_logits)

        pt = pred_sigmoid * targets + (1 - pred_sigmoid) * (1 - targets)

        # focal weight
        focal_weight = (1 - pt) ** self.gamma

        # alpha balance
        if isinstance(self.alpha, torch.Tensor):
            # per-class alpha: [C] → [1, C] → broadcast 到 [B, C]
            alpha_vec = self.alpha.to(pred_logits.device)
            alpha_factor = alpha_vec.unsqueeze(0) * targets + (1 - alpha_vec).unsqueeze(0) * (1 - targets)
        else:
            # scalar alpha
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")

        loss = alpha_factor * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# 1. Tabular 预处理 & 数据增强标注（data_preprocess_tabular）
# ============================================================
def expand_positive_with_aug(train_df: pd.DataFrame, positive_cols: List[str]) -> pd.DataFrame:
    train_df = train_df.copy()
    train_df["is_positive"] = (train_df[positive_cols] > 0).any(axis=1).astype(int)

    pos_df = train_df[train_df["is_positive"] == 1]

    rot_df = pos_df.copy()
    rot_df["aug_type"] = "rotate"

    flip_df = pos_df.copy()
    flip_df["aug_type"] = "flip"

    noise_df = pos_df.copy()
    noise_df["aug_type"] = "noise"

    base_df = train_df.copy()
    base_df["aug_type"] = "none"

    out = pd.concat([base_df, rot_df, flip_df, noise_df], ignore_index=True)
    print(f"[aug] Positive expanded: {len(pos_df)} → {len(pos_df)*4}")
    return out


def mark_positive_cases(df: pd.DataFrame, pos_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["is_positive"] = (df[pos_cols] > 0).any(axis=1).astype(int)
    return df


def _build_ethnicity_vocab(train_series: pd.Series):
    # reserve 0 for <UNK>
    vals = train_series.dropna().astype(str).unique().tolist()
    vals = sorted(vals)
    vocab = {"<UNK>": 0}
    for i, v in enumerate(vals, start=1):
        vocab[v] = i
    return vocab

def _apply_ethnicity_id(df: pd.DataFrame, vocab: dict, col: str = "ETHNICITY_DESC"):
    df = df.copy()
    if col not in df.columns:
        df["ETHNICITY_ID"] = 0
        return df
    s = df[col].fillna("<UNK>").astype(str)
    df["ETHNICITY_ID"] = s.map(lambda x: vocab.get(x, 0)).astype(int)
    return df

def _make_viewposition_binary(df: pd.DataFrame):
    # Optional helper if your CSV has ViewPosition-like columns
    df = df.copy()
    for c in ["ViewPosition", "ViewPosition_code", "ViewPositionFinal"]:
        if c in df.columns:
            vp = df[c].fillna("").astype(str).str.upper()
            df["ViewPosition_MLO"] = vp.str.contains("MLO").astype(int)
            df["ViewPosition_CC"]  = vp.str.contains("CC").astype(int)
            return df
    return df

def _make_laterality_binary(df: pd.DataFrame):
    df = df.copy()
    for c in ["ImageLateralityFinal", "Laterality", "ImageLaterality"]:
        if c in df.columns:
            lat = df[c].fillna("").astype(str).str.upper()
            df["Laterality_R"] = (lat == "R").astype(int)
            df["Laterality_L"] = (lat == "L").astype(int)
            return df
    return df

def prepare_tabular_data(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"[info] Loaded CSV with {len(df)} rows")

    # ---- labels (what your BCE head predicts) ----
    label_cols = ["mass", "calc", "cancer"]
    for c in label_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0
    df[label_cols] = df[label_cols].fillna(0)

    # ---- features (what TabNet encodes) ----
    # Keep this flexible: it only uses columns that exist.
    df = _make_viewposition_binary(df)
    df = _make_laterality_binary(df)

    candidate_num = [
        "BIRADS", "tissueden",
        "total_L_find", "total_R_find",
        "ViewPosition_MLO", "ViewPosition_CC",
        "Laterality_R", "Laterality_L",
    ]
    num_cols = [c for c in candidate_num if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ---- image path ----
    # keep your existing logic
    df["image_path"] = df["anon_dicom_path"].astype(str)

    # ---- folds split ----
    train_df = df[df["fold"].isin([2, 3])].reset_index(drop=True)
    val_df   = df[df["fold"] == 1].reset_index(drop=True)
    test_df  = df[df["fold"] == 0].reset_index(drop=True)
    print(f"[info] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ---- ETHNICITY_DESC -> ETHNICITY_ID (embedding input) ----
    vocab = _build_ethnicity_vocab(train_df.get("ETHNICITY_DESC", pd.Series([], dtype=str)))
    train_df = _apply_ethnicity_id(train_df, vocab, col="ETHNICITY_DESC")
    val_df   = _apply_ethnicity_id(val_df, vocab, col="ETHNICITY_DESC")
    test_df  = _apply_ethnicity_id(test_df, vocab, col="ETHNICITY_DESC")
    num_ethnicities = max(vocab.values()) + 1  # embedding size

    return train_df, val_df, test_df, num_cols, "ETHNICITY_ID", num_ethnicities, label_cols

# ============================================================
# 2. 模型定义（model_image_tabular + 多标签头）
# ============================================================
class GLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)

    def forward(self, x):
        x = self.bn(self.fc(x))
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)

class FeatureTransformer(nn.Module):
    def __init__(self, dim, n_glu=2):
        super().__init__()
        blocks = []
        for _ in range(n_glu):
            blocks.append(GLUBlock(dim, dim))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(x)  # residual
        return x

class AttentiveTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, a):
        # Softmax mask (simple + stable)
        m = self.bn(self.fc(a))
        return torch.softmax(m, dim=1)

class TabularEncoderTabNet(nn.Module):
    """
    TabNet-style encoder that outputs a single embedding vector per row.
    ETHNICITY is embedded via nn.Embedding and concatenated to numeric features.
    """
    def __init__(
        self,
        num_dim: int,
        num_ethnicities: int,
        embed_dim: int = 512,
        eth_emb_dim: int = 32,
        hidden_dim: int = 256,
        n_steps: int = 3,
        gamma: float = 1.5,
    ):
        super().__init__()
        self.eth_emb = nn.Embedding(num_ethnicities, eth_emb_dim)

        self.in_dim = num_dim + eth_emb_dim
        self.bn_in = nn.BatchNorm1d(self.in_dim)

        self.fc_in = nn.Linear(self.in_dim, hidden_dim)
        self.shared = FeatureTransformer(hidden_dim, n_glu=2)

        self.att = AttentiveTransformer(hidden_dim)

        self.n_steps = n_steps
        self.gamma = gamma

        self.fc_dec = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x_num: torch.Tensor, eth_id: torch.Tensor):
        # x_num: [B, num_dim] float
        # eth_id: [B] long
        e = self.eth_emb(eth_id)                     # [B, eth_emb_dim]
        x = torch.cat([x_num, e], dim=1)             # [B, in_dim]
        x = self.bn_in(x)

        h = F.relu(self.fc_in(x))                    # [B, hidden_dim]
        prior = torch.ones_like(h)                   # [B, hidden_dim]

        agg = 0.0
        for _ in range(self.n_steps):
            h_masked = h * prior
            z = self.shared(h_masked)                # [B, hidden_dim]

            d = F.relu(self.fc_dec(z))               # decision
            agg = agg + d

            m = self.att(z)                          # mask in hidden space
            prior = prior * (self.gamma - m)
            prior = torch.clamp(prior, min=0.0)

        out = self.fc_out(agg / float(self.n_steps)) # [B, embed_dim]
        return out


def build_image_encoder(embed_dim=512):
    try:
        model = models.efficientnet_b5(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, embed_dim),
        )
        print("[info] Loaded ImageNet pretrained EfficientNet-B5")
    except Exception as e:
        print(f"[warn] Could not load EfficientNet-B5 pretrained weights: {e}")
        print("[info] Falling back to random init EfficientNet-B5")
        model = models.efficientnet_b5(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, embed_dim),
        )
    return model


class ImageTabularCLIP(nn.Module):
    """
    Image + tabular CLIP model
    tabular encoder = TabNet-style embedding encoder (includes ETHNICITY embedding)
    """
    def __init__(self, tab_num_dim: int, num_ethnicities: int, label_dim: int, embed_dim=512):
        super().__init__()
        self.image_encoder = build_image_encoder(embed_dim)

        self.tabular_encoder = TabularEncoderTabNet(
            num_dim=tab_num_dim,
            num_ethnicities=num_ethnicities,
            embed_dim=embed_dim,
            eth_emb_dim=32,
            hidden_dim=256,
            n_steps=3,
            gamma=1.5,
        )

        # BCE head predicts multi-label lesions from image embedding
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, label_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

        # normal samples → same zero embedding (optional, kept from your code)
        self.normal_zero_embed = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, image, tab_num, eth_id, labels=None):
        img_feat = self.image_encoder(image)
        img_feat = F.normalize(img_feat, dim=-1)

        tab_feat = self.tabular_encoder(tab_num, eth_id)
        tab_feat = F.normalize(tab_feat, dim=-1)

        # normal rows (by labels) → map to shared zero embedding
        if labels is not None:
            normal_mask = (labels.sum(dim=1) == 0)
            if normal_mask.any():
                zero_emb = F.normalize(self.normal_zero_embed, dim=-1)
                tab_feat[normal_mask] = zero_emb

        pred_labels = self.cls_head(img_feat)
        return img_feat, tab_feat, pred_labels, self.logit_scale.exp()


# ============================================================
# 3. Dataset + 训练主循环
# ============================================================
class ImageTabularDataset(Dataset):
    def __init__(self, df, num_cols, eth_id_col, label_cols, img_root, use_aug=False):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.eth_id_col = eth_id_col
        self.label_cols = label_cols
        self.img_root = img_root
        self.use_aug = use_aug

        self.base_tf = transforms.Compose([transforms.Resize((912, 1520)), transforms.ToTensor()])
        self.rotate_tf = transforms.Compose([transforms.Resize((912, 1520)), transforms.RandomRotation(degrees=15), transforms.ToTensor()])
        self.flip_tf = transforms.Compose([transforms.Resize((912, 1520)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()])
        self.noise_tf = transforms.Compose([transforms.Resize((912, 1520)), transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except:
            img = Image.new("RGB", (912, 1520))

        aug_type = row.get("aug_type", "none") if self.use_aug else "none"
        if aug_type == "rotate":
            image = self.rotate_tf(img)
        elif aug_type == "flip":
            image = self.flip_tf(img)
        elif aug_type == "noise":
            image = self.noise_tf(img)
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)
        else:
            image = self.base_tf(img)

        # numeric features
        if len(self.num_cols) > 0:
            x_num = pd.to_numeric(row[self.num_cols], errors="coerce").fillna(0).values.astype(np.float32)
        else:
            x_num = np.zeros((0,), dtype=np.float32)
        x_num = torch.from_numpy(x_num)

        # ethnicity id (embedding input)
        eth_id = int(row.get(self.eth_id_col, 0))
        eth_id = torch.tensor(eth_id, dtype=torch.long)

        # labels (multi-label 0/1)
        y = pd.to_numeric(row[self.label_cols], errors="coerce").fillna(0).values.astype(np.float32)
        y = (y > 0).astype(np.float32)
        y = torch.from_numpy(y)

        is_positive = row.get("is_positive", 0)
        is_positive = torch.tensor(int(is_positive), dtype=torch.long)

        return image, x_num, eth_id, y, is_positive


def train_main(args):
    csv_path = args.csv_path
    img_root = args.img_root

    print(f"[debug] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[debug] Device name: {torch.cuda.get_device_name(0)}")
        print(f"[debug] Device count: {torch.cuda.device_count()}")
    else:
        print("[warn] CUDA not available — running on CPU.")

    # ===================== 可配置“病例定义列” =====================
    positive_cols = ["mass", "calc", "cancer"]

    train_df, val_df, test_df, num_cols, eth_id_col, num_ethnicities, label_cols = prepare_tabular_data(csv_path)

    positive_cols = label_cols
    train_df = expand_positive_with_aug(train_df, positive_cols)
    val_df   = mark_positive_cases(val_df, positive_cols)
    test_df  = mark_positive_cases(test_df, positive_cols)

    train_ds = ImageTabularDataset(train_df, num_cols, eth_id_col, label_cols, img_root, use_aug=True)
    val_ds   = ImageTabularDataset(val_df,   num_cols, eth_id_col, label_cols, img_root, use_aug=False)

    class_counts = train_df["is_positive"].value_counts().to_dict()
    print("[info] Train is_positive distribution:", class_counts)

    weight_for_class = {
        0: 1.0 / class_counts.get(0, 1),
        1: 1.0 / class_counts.get(1, 1)
    }
    sample_weights = train_df["is_positive"].map(weight_for_class).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler,
                              num_workers=16, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False,
                              num_workers=16, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageTabularCLIP(
        tab_num_dim=len(num_cols),
        num_ethnicities=num_ethnicities,
        label_dim=len(label_cols),
        embed_dim=512
    ).to(device)

    # 解冻 EfficientNet 高层
    for name, param in model.image_encoder.named_parameters():
        if ("features.6" in name) or ("features.7" in name) or ("features.8" in name) or ("classifier" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    vision_params  = [p for p in model.image_encoder.parameters() if p.requires_grad]
    tabular_params = list(model.tabular_encoder.parameters())
    cls_params     = list(model.cls_head.parameters())
    other_params   = [model.logit_scale, model.normal_zero_embed]

    optimizer = torch.optim.AdamW(
        vision_params + tabular_params + cls_params + other_params,
        lr=1e-4
    )

    focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25).to(device)

    # ====== class-wise alpha 向量，根据 tab_cols 顺序取 ======
    alpha_cfg = build_class_alpha_config(args)
    class_alpha_vec = [alpha_cfg.get(name, 0.25) for name in label_cols]
    focal_bce_loss_fn = FocalBCELoss(alpha=class_alpha_vec, gamma=2.0).to(device)

    num_epochs = args.num_epochs
    best_val_loss = float("inf")
    best_epoch = -1
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for img, x_num, eth_id, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", ncols=100):
            img = img.to(device)
            x_num = x_num.to(device)
            eth_id = eth_id.to(device)
            y = y.to(device)

            img_f, tab_f, pred_y, logit_scale = model(img, x_num, eth_id, labels=y)

            labels = torch.arange(img.size(0), device=device)
            logits_i2t = logit_scale * (img_f @ tab_f.T)
            logits_t2i = logit_scale * (tab_f @ img_f.T)

            loss_i2t = focal_loss_fn(logits_i2t, labels)
            loss_t2i = focal_loss_fn(logits_t2i, labels)
            loss_clip = 0.75 * loss_i2t + 0.25 * loss_t2i

            loss_bce = focal_bce_loss_fn(pred_y, y)
            reg_loss = 0.02 * tab_f.norm(dim=1).mean()

            loss = loss_clip + 0.5 * loss_bce + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0.0, 4.6052)

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- Val --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, x_num, eth_id, y, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", ncols=100):
                img = img.to(device)
                x_num = x_num.to(device)
                eth_id = eth_id.to(device)
                y = y.to(device)

                img_f, tab_f, pred_y, logit_scale = model(img, x_num, eth_id, labels=y)

                loss_bce = focal_bce_loss_fn(pred_y, y)
                val_loss += loss_bce.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train loss: {avg_train_loss:.4f}, Val BCE loss: {avg_val_loss:.4f}")
        print(f"[debug] logit_scale(exp)={model.logit_scale.exp().item():.2f}")

        last_path = os.path.join(save_dir, "last_epoch.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }, last_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"[info] New best model saved → {best_path} (val_BCE={avg_val_loss:.4f})")


# ============================================================
# 4. 增强评估（BCE + CLIP + Per-class 报告 + 自动阈值）
# ============================================================

def search_optimal_thresholds(probs: np.ndarray,
                              trues: np.ndarray,
                              class_names: List[str]):
    """
    针对每个类，在一组候选阈值上搜索 F1 最高的 threshold。
    只用于评估阶段，不反向传播，也不影响训练。
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    best_ths = []
    best_f1s = []

    for i, name in enumerate(class_names):
        y_true_i = trues[:, i]
        y_prob_i = probs[:, i]

        # 如果这一类在当前集合里全是 0 或全是 1，F1 没法区分，直接跳过
        if y_true_i.max() == y_true_i.min():
            best_ths.append(0.5)
            best_f1s.append(0.0)
            continue

        best_th = 0.5
        best_f1 = 0.0

        for th in thresholds:
            y_pred_i = (y_prob_i >= th).astype(int)
            f1_i = f1_score(y_true_i, y_pred_i, zero_division=0)
            if f1_i > best_f1:
                best_f1 = f1_i
                best_th = th

        best_ths.append(best_th)
        best_f1s.append(best_f1)

    return np.array(best_ths), np.array(best_f1s)


def evaluate_model(ckpt_path: str,
                   csv_path: str,
                   img_root: str,
                   eval_mode: str = "both"):
    """
    eval_mode:
      - "case": BCE 多标签分类评估（含 per-class + AUC + 自动阈值）
      - "clip": 只做 BCE head 的评估（含 per-class + AUC，不做阈值搜索）
      - "both": 两个都跑
    """
    print(f"[eval] Mode = {eval_mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 1) Load data (current labels = 3)
    # ----------------------------
    train_df, val_df, test_df, num_cols, eth_id_col, num_ethnicities, label_cols = prepare_tabular_data(csv_path)
    test_df = mark_positive_cases(test_df, label_cols)
    tab_cols = label_cols  # what you want to evaluate on (3)

    loader = DataLoader(
        ImageTabularDataset(test_df, num_cols, eth_id_col, label_cols, img_root, use_aug=False),
        batch_size=8, shuffle=False, num_workers=4
    )

    # ----------------------------
    # 2) Load checkpoint first to detect head dim (fix 4->3 mismatch)
    # ----------------------------
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    # find cls_head final linear layer weight key (usually "cls_head.2.weight")
    wkey = None
    for k in sd.keys():
        if k.endswith("cls_head.2.weight"):
            wkey = k
            break
    if wkey is None:
        # fallback: try any key that looks like last linear weight
        for k in sd.keys():
            if "cls_head" in k and k.endswith(".weight"):
                wkey = k
        if wkey is None:
            raise KeyError("Could not find cls_head.*.weight in checkpoint state_dict")

    ckpt_label_dim = sd[wkey].shape[0]  # 4 in your error, or 3 for new checkpoints
    print(f"[eval] checkpoint label_dim = {ckpt_label_dim}")

    # IMPORTANT: this must match the label order used when that checkpoint was trained.
    # Common old order (4): mass, asymmetry, calc, cancer
    if ckpt_label_dim == 4:
        ckpt_tab_cols = ["mass", "asymmetry", "calc", "cancer"]
    else:
        ckpt_tab_cols = tab_cols[:ckpt_label_dim]

    # map current eval labels -> checkpoint output indices
    keep_idx = []
    keep_names = []
    for name in tab_cols:
        if name in ckpt_tab_cols:
            keep_idx.append(ckpt_tab_cols.index(name))
            keep_names.append(name)

    if len(keep_idx) != len(tab_cols):
        print("[warn] Some eval labels not found in checkpoint label list.")
        print("[warn] ckpt_tab_cols =", ckpt_tab_cols)
        print("[warn] eval tab_cols  =", tab_cols)
        print("[warn] Will evaluate only on:", keep_names)

    print("[eval] ckpt_tab_cols =", ckpt_tab_cols)
    print("[eval] eval tab_cols  =", tab_cols)
    print("[eval] keep_idx      =", keep_idx)

    # ----------------------------
    # 3) Build model using checkpoint label_dim, then load
    # ----------------------------
    model = ImageTabularCLIP(
        tab_num_dim=len(num_cols),
        num_ethnicities=num_ethnicities,
        label_dim=ckpt_label_dim,   # <- key fix
        embed_dim=512
    ).to(device)

    model.load_state_dict(sd)
    model.eval()

    # ----------------------------
    # Helpers: safe AUC (handles all-0 / all-1 labels)
    # ----------------------------
    def _safe_roc_auc(y_true_1d, y_score_1d):
        if np.unique(y_true_1d).size < 2:
            return np.nan
        return roc_auc_score(y_true_1d, y_score_1d)

    def _safe_pr_auc(y_true_1d, y_score_1d):
        try:
            return average_precision_score(y_true_1d, y_score_1d)
        except Exception:
            return np.nan

    def _print_auc_block(trues, probs, names):
        # per-class
        roc_list, pr_list = [], []
        for i, name in enumerate(names):
            roc_i = _safe_roc_auc(trues[:, i], probs[:, i])
            pr_i  = _safe_pr_auc(trues[:, i], probs[:, i])
            roc_list.append(roc_i)
            pr_list.append(pr_i)

        roc_macro = float(np.nanmean(roc_list)) if len(roc_list) else np.nan
        pr_macro  = float(np.nanmean(pr_list))  if len(pr_list)  else np.nan

        # micro (flatten)
        roc_micro = _safe_roc_auc(trues.reshape(-1), probs.reshape(-1))
        pr_micro  = _safe_pr_auc(trues.reshape(-1), probs.reshape(-1))

        print("\n--- AUC Metrics ---")
        print(f"AUROC Macro: {roc_macro:.4f}" if not np.isnan(roc_macro) else "AUROC Macro: nan")
        print(f"AUROC Micro: {roc_micro:.4f}" if not np.isnan(roc_micro) else "AUROC Micro: nan")
        print(f"AUPRC Macro: {pr_macro:.4f}"  if not np.isnan(pr_macro)  else "AUPRC Macro: nan")
        print(f"AUPRC Micro: {pr_micro:.4f}"  if not np.isnan(pr_micro)  else "AUPRC Micro: nan")

        print("\n--- Per-class AUC (AUROC & AUPRC) ---")
        for i, name in enumerate(names):
            r = roc_list[i]
            p = pr_list[i]
            r_str = f"{r:.4f}" if not np.isnan(r) else "nan"
            p_str = f"{p:.4f}" if not np.isnan(p) else "nan"
            print(f"{name}: AUROC={r_str}, AUPRC={p_str}")

    # ----------------------------
    # 4) Forward once (logits from BCE head), then slice to eval labels
    # ----------------------------
    def _run_bce_forward():
        all_logits, all_trues = [], []
        with torch.no_grad():
            for img, x_num, eth_id, y, _ in tqdm(loader, desc="Eval[BCE]", ncols=100):
                img = img.to(device)
                x_num = x_num.to(device)
                eth_id = eth_id.to(device)
                y = y.to(device)

                _, _, pred_y, _ = model(img, x_num, eth_id, labels=y)  # pred_y: [B, ckpt_label_dim]
                all_logits.append(pred_y.cpu())
                all_trues.append(y.cpu())  # y: [B, len(tab_cols)]

        logits = torch.cat(all_logits, dim=0).numpy()
        trues  = torch.cat(all_trues,  dim=0).numpy().astype(int)

        # sigmoid probabilities for checkpoint outputs
        probs_full = 1 / (1 + np.exp(-logits))  # [N, ckpt_label_dim]

        # slice probs to match current eval labels (trues already in eval label order)
        probs = probs_full[:, keep_idx] if len(keep_idx) else probs_full[:, :trues.shape[1]]
        # if we dropped labels (rare), also drop trues columns to match
        trues = trues[:, :probs.shape[1]]

        names = tab_cols[:probs.shape[1]]
        preds = (probs > 0.5).astype(int)

        return trues, probs, preds, names

    # ============================================================
    # CASE mode (full report + AUC + threshold search)
    # ============================================================
    if eval_mode in ("case", "both"):
        print("\n====================== BCE Evaluation (CASE) ======================\n")
        trues, probs, preds, names = _run_bce_forward()

        print("Macro F1:", f1_score(trues, preds, average="macro", zero_division=0))
        print("Micro F1:", f1_score(trues, preds, average="micro", zero_division=0))
        print("Exact Match:", (preds == trues).all(axis=1).mean())

        print("\n--- Per-class report (thr=0.5) ---")
        print("Classes:", names)
        print(classification_report(trues, preds, target_names=names, zero_division=0))

        print("\n--- Per-class Accuracy (thr=0.5) ---")
        for i, name in enumerate(names):
            acc_i = (preds[:, i] == trues[:, i]).mean()
            print(f"{name}: {acc_i:.4f}")

        _print_auc_block(trues, probs, names)

        print("\n--- Optimal Thresholds per class (based on F1 on this split) ---")
        best_ths, best_f1s = search_optimal_thresholds(probs, trues, names)
        for i, name in enumerate(names):
            print(f"{name}: best_thr={best_ths[i]:.3f}, best_F1={best_f1s[i]:.4f}")

        opt_preds = np.zeros_like(preds)
        for i in range(len(names)):
            opt_preds[:, i] = (probs[:, i] >= best_ths[i]).astype(int)

        print("\n--- Metrics with optimal per-class thresholds (for reference) ---")
        print("Macro F1 (opt):", f1_score(trues, opt_preds, average="macro", zero_division=0))
        print("Micro F1 (opt):", f1_score(trues, opt_preds, average="micro", zero_division=0))
        print("Exact Match (opt):", (opt_preds == trues).all(axis=1).mean())

    # ============================================================
    # CLIP mode (BCE head eval + AUC)
    # ============================================================
    if eval_mode in ("clip", "both"):
        print("\n================== BCE Head Evaluation (CLIP mode) ==================\n")
        trues, probs, preds, names = _run_bce_forward()

        print("Macro F1:", f1_score(trues, preds, average="macro", zero_division=0))
        print("Micro F1:", f1_score(trues, preds, average="micro", zero_division=0))
        print("Exact Match:", (preds == trues).all(axis=1).mean())

        print("\n--- Per-class report (thr=0.5) ---")
        print(classification_report(trues, preds, target_names=names, zero_division=0))

        print("\n--- Per-class Accuracy (thr=0.5) ---")
        for i, name in enumerate(names):
            acc_i = (preds[:, i] == trues[:, i]).mean()
            print(f"{name}: {acc_i:.4f}")

        _print_auc_block(trues, probs, names)

def parse_args():
    p = argparse.ArgumentParser()

    # requested args
    p.add_argument("--save_dir", type=str, default="./TABWFOCFullSize")
    p.add_argument("--csv_path", type=str, default="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/Embed_classification_diagnostic_data_wo_dup_folds_update.csv")
    p.add_argument("--img_root", type=str, default="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/dataset/cohort_1_process")
    p.add_argument("--num_epochs", type=int, default=10)

    # CLASS_ALPHA_CONFIG input (two ways):
    # 1) JSON dict string
    p.add_argument(
        "--class_alpha_config",
        type=str,
        default=None,
        help='JSON dict, e.g. \'{"mass":0.25,"calc":0.25,"cancer":0.75}\''
    )
    # 2) repeatable key=value overrides
    p.add_argument(
        "--alpha",
        action="append",
        default=[],
        help="Override one class alpha as name=value (can repeat), e.g. --alpha cancer=0.75"
    )

    # optional (keeps your current default behavior: eval runs unless you choose train)
    p.add_argument("--run", choices=["train", "eval"], default="eval")
    p.add_argument("--ckpt_path", type=str, default="./TABWFOCFullSize/last_epoch.pt")
    p.add_argument("--eval_mode", choices=["case", "clip", "both"], default="clip")

    return p.parse_args()


def build_class_alpha_config(args):
    # start from the default dict in the file
    cfg = dict(CLASS_ALPHA_CONFIG)

    # full override via JSON
    if args.class_alpha_config is not None:
        try:
            loaded = json.loads(args.class_alpha_config)
            if not isinstance(loaded, dict):
                raise ValueError("class_alpha_config must be a JSON object/dict")
            cfg = {str(k): float(v) for k, v in loaded.items()}
        except Exception as e:
            raise ValueError(
                f"Failed to parse --class_alpha_config as JSON dict. "
                f"Example: --class_alpha_config '{{\"mass\":0.25,\"calc\":0.25,\"cancer\":0.75}}'. Error: {e}"
            )

    # per-key overrides
    for item in args.alpha:
        if "=" not in item:
            raise ValueError(f"Bad --alpha '{item}'. Use name=value, e.g. --alpha cancer=0.75")
        k, v = item.split("=", 1)
        cfg[k.strip()] = float(v)

    return cfg

# ============================================================
# 5. main（默认训练）
# ============================================================
# if __name__ == "__main__":
#     # 模式 C：保留训练 + 评估，通过注释切换
#     # train_main()
#     ckpt_path = "./TABWFOCFullSize/last_epoch.pt"
#     csv_path  = "/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/Embed_classification_diagnostic_data_wo_dup_folds_update.csv"
#     img_root  = "/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/dataset/cohort_1_process"
#     evaluate_model(ckpt_path, csv_path, img_root, eval_mode="clip")

if __name__ == "__main__":
    args = parse_args()

    if args.run == "train":
        train_main(args)
    else:
        evaluate_model(args.ckpt_path, args.csv_path, args.img_root, eval_mode=args.eval_mode)

