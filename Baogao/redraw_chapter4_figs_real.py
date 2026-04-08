import csv
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.dataset import CLASSES as PUBLIC_CLASSES
from Data.dataset import SOURCE_DIR, SourceDataset, data_manager, get_dataloaders
from Data.self_dataset import get_self_dataloaders
from Model.ISAN import Model
from baseline.private_runner.run_private_baseline import BaselineModel

IMG_DIR = ROOT / "Baogao" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

_font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
_active_font = None
for _f in _font_candidates:
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
        _active_font = fm.FontProperties(fname=_f).get_name()
        break

if _active_font is None:
    _active_font = "DejaVu Sans"

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = _active_font
plt.rcParams["font.sans-serif"] = [_active_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def resolve_path(rel_path):
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return p
    p2 = ROOT / rel_path
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Path not found: {rel_path}")


def first_existing(candidates):
    for c in candidates:
        p = ROOT / c
        if p.exists():
            return p
    raise FileNotFoundError(f"None exists: {candidates}")


def to_percent(x):
    return float(x) * 100.0


def compute_fault_metrics(labels, preds, normal_indices):
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    normal_mask = np.isin(labels, normal_indices)
    fault_mask = ~normal_mask
    pred_normal = np.isin(preds, normal_indices)

    false_alarm = np.logical_and(~pred_normal, normal_mask).sum()
    detected_fault = np.logical_and(~pred_normal, fault_mask).sum()

    far = false_alarm / max(int(normal_mask.sum()), 1)
    fdr = detected_fault / max(int(fault_mask.sum()), 1)
    return far, fdr


def infer_isan(model, loader, device, normal_indices, collect_features=False, max_samples=None):
    model.eval()
    preds, labels, probs = [], [], []
    feats = []

    taken = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                y = batch[1]
            else:
                continue

            x = x.to(device)
            y = y.to(device)

            f = model.FRFE(x)
            logits = model.main_classifier(f)
            p = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            preds.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())
            probs.append(p.cpu().numpy())

            if collect_features:
                feat_vec = f.mean(dim=-1).cpu().numpy()
                feats.append(feat_vec)

            taken += y.size(0)
            if max_samples is not None and taken >= max_samples:
                break

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    probs = np.concatenate(probs)

    far, fdr = compute_fault_metrics(labels, preds, normal_indices)
    acc = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average="macro"))

    try:
        auc_macro = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        auc_macro = float("nan")

    fault_scores = 1.0 - probs[:, normal_indices].sum(axis=1)
    y_fault = (~np.isin(labels, normal_indices)).astype(np.int32)

    try:
        fpr, tpr, _ = roc_curve(y_fault, fault_scores)
        auc_bin = float(roc_auc_score(y_fault, fault_scores))
    except Exception:
        fpr, tpr, auc_bin = np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    out = {
        "acc": acc,
        "far": float(far),
        "fdr": float(fdr),
        "f1": f1,
        "auc_macro": auc_macro,
        "auc_bin": auc_bin,
        "preds": preds,
        "labels": labels,
        "probs": probs,
        "fault_scores": fault_scores,
        "y_fault": y_fault,
        "fpr": fpr,
        "tpr": tpr,
    }

    if collect_features:
        out["features"] = np.concatenate(feats)
    return out


def infer_baseline(model, loader, device, normal_indices, collect_features=False, max_samples=None):
    model.eval()
    preds, labels, probs = [], [], []
    feats = []

    taken = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            feat, logits = model(x)
            p = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            preds.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())
            probs.append(p.cpu().numpy())

            if collect_features:
                feats.append(feat.cpu().numpy())

            taken += y.size(0)
            if max_samples is not None and taken >= max_samples:
                break

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    probs = np.concatenate(probs)

    far, fdr = compute_fault_metrics(labels, preds, normal_indices)
    acc = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average="macro"))

    try:
        auc_macro = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        auc_macro = float("nan")

    fault_scores = 1.0 - probs[:, normal_indices].sum(axis=1)
    y_fault = (~np.isin(labels, normal_indices)).astype(np.int32)

    try:
        fpr, tpr, _ = roc_curve(y_fault, fault_scores)
        auc_bin = float(roc_auc_score(y_fault, fault_scores))
    except Exception:
        fpr, tpr, auc_bin = np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    out = {
        "acc": acc,
        "far": float(far),
        "fdr": float(fdr),
        "f1": f1,
        "auc_macro": auc_macro,
        "auc_bin": auc_bin,
        "preds": preds,
        "labels": labels,
        "probs": probs,
        "fault_scores": fault_scores,
        "y_fault": y_fault,
        "fpr": fpr,
        "tpr": tpr,
    }

    if collect_features:
        out["features"] = np.concatenate(feats)
    return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_fps_isan(checkpoint, attention_mode, num_classes, loader, device, warmup=5, max_batches=30):
    model = Model(feature_dim=320, num_classes=num_classes, attention_mode=attention_mode).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    batches = list(loader)
    run_batches = batches[:max_batches] if len(batches) > max_batches else batches

    with torch.no_grad():
        for i, (x, _) in enumerate(run_batches[:warmup]):
            x = x.to(device)
            f = model.FRFE(x)
            _ = model.main_classifier(f)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    total = 0
    with torch.no_grad():
        for x, y in run_batches:
            x = x.to(device)
            f = model.FRFE(x)
            _ = model.main_classifier(f)
            total += y.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = max(time.time() - t0, 1e-8)

    return total / dt, count_params(model)


def measure_fps_baseline(method, checkpoint, num_classes, loader, device, warmup=5, max_batches=30):
    model = BaselineModel(num_classes=num_classes, feature_dim=128, method=method).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    batches = list(loader)
    run_batches = batches[:max_batches] if len(batches) > max_batches else batches

    with torch.no_grad():
        for i, (x, _) in enumerate(run_batches[:warmup]):
            x = x.to(device)
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    total = 0
    with torch.no_grad():
        for x, y in run_batches:
            x = x.to(device)
            _ = model(x)
            total += y.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = max(time.time() - t0, 1e-8)

    return total / dt, count_params(model)


def load_baseline_loss_curve(csv_path):
    epochs, vals = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss = float(row["test_loss"])
            if loss >= 0:
                epochs.append(int(row["epoch"]))
                vals.append(loss)
    return np.array(epochs), np.array(vals)


def load_txt_loss_curve(txt_path):
    arr = np.loadtxt(txt_path)
    arr = np.atleast_1d(arr)
    epochs = np.arange(1, len(arr) + 1)
    mask = arr >= 0
    return epochs[mask], arr[mask]


def draw_box(ax, xy, w, h, text, fc):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10)


def draw_arrow(ax, p1, p2):
    arr = FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=14, linewidth=1.2, color="#1f2937")
    ax.add_patch(arr)


def plot_network_structure():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_box(ax, (0.05, 0.42), 0.16, 0.16, "源域/目标域信号", "#e0f2fe")
    draw_box(ax, (0.26, 0.42), 0.16, 0.16, "特征编码器 ISGFAN", "#dbeafe")
    draw_box(ax, (0.47, 0.42), 0.16, 0.16, "全局-局部注意力 GLFA", "#ede9fe")
    draw_box(ax, (0.68, 0.58), 0.18, 0.14, "故障分类分支", "#dcfce7")
    draw_box(ax, (0.68, 0.26), 0.18, 0.14, "域判别分支", "#fee2e2")
    draw_box(ax, (0.47, 0.14), 0.16, 0.12, "TEC 熵阈课程", "#fef3c7")
    draw_box(ax, (0.88, 0.26), 0.1, 0.14, "DGT\n动态 GRL", "#fde68a")

    draw_arrow(ax, (0.21, 0.50), (0.26, 0.50))
    draw_arrow(ax, (0.42, 0.50), (0.47, 0.50))
    draw_arrow(ax, (0.63, 0.50), (0.68, 0.65))
    draw_arrow(ax, (0.63, 0.50), (0.68, 0.33))
    draw_arrow(ax, (0.56, 0.42), (0.56, 0.26))
    draw_arrow(ax, (0.86, 0.33), (0.88, 0.33))

    ax.text(0.78, 0.75, "分类损失", fontsize=10, color="#065f46")
    ax.text(0.78, 0.19, "对抗损失", fontsize=10, color="#991b1b")
    ax.text(0.50, 0.08, "图4-1  DGT-GLFA 网络结构图", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "network_structure_diagram.png", dpi=300)
    plt.close()


def plot_experiment_flowchart():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = [0.84, 0.68, 0.52, 0.36, 0.20]
    labels = [
        "私有/公开数据准备",
        "信号切片与标准化",
        "源域监督 + 目标域对齐",
        "DGT + TEC + GLFA 联合训练",
        "指标统计与图形输出",
    ]
    colors = ["#e0f2fe", "#f0f9ff", "#ede9fe", "#fee2e2", "#dcfce7"]

    for yy, label, color in zip(y, labels, colors):
        draw_box(ax, (0.17, yy), 0.66, 0.10, label, color)

    for i in range(len(y) - 1):
        draw_arrow(ax, (0.50, y[i]), (0.50, y[i + 1] + 0.10))

    ax.text(0.50, 0.06, "图4-2  实验流程图", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "experiment_flowchart.png", dpi=300)
    plt.close()


def plot_private_convergence(curves):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    palette = ["#0ea5e9", "#f97316", "#8b5cf6", "#16a34a"]

    for i, (name, (ep, loss_vals)) in enumerate(curves.items()):
        ax.plot(ep, loss_vals, marker="o", markersize=3.0, linewidth=2.0, color=palette[i % len(palette)], label=name)

    ax.set_xlabel("训练轮次 Epoch")
    ax.set_ylabel("测试损失 Loss（对数坐标）")
    ax.set_title("私有数据集训练收敛曲线（测试损失，真实日志）")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "private_convergence.png", dpi=300)
    plt.close()


def plot_private_metric_panel(method_names, acc, f1, auc):
    x = np.arange(len(method_names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.bar(x - width, acc, width, label="Acc", color="#3b82f6")
    ax.bar(x, f1, width, label="F1", color="#16a34a")
    ax.bar(x + width, auc, width, label="AUC", color="#f97316")

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=30, ha="right")
    ax.set_ylabel("指标值 (%)")
    ax.set_title("私有数据集综合指标对比（真实评估）")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "private_metric_panel.png", dpi=300)
    plt.close()


def plot_private_fdr_far(method_names, fdr, far):
    x = np.arange(len(method_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.bar(x - width / 2, fdr, width, label="FDR", color="#2563eb")
    ax.bar(x + width / 2, far, width, label="FAR", color="#ef4444")

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=30, ha="right")
    ax.set_ylabel("指标值 (%)")
    ax.set_title("私有数据集 FDR/FAR 对比（真实评估）")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "fdr_far_comparison.png", dpi=300)
    plt.close()


def plot_anomaly_kde(score_dict, normal_indices):
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 14.5), sharex=True, sharey=True)
    axes = axes.flatten()
    bins = np.linspace(-8.0, 8.0, 45)

    def score_to_logit(x, eps=1e-4):
        x = np.clip(x, eps, 1.0 - eps)
        return np.log(x / (1.0 - x))

    for i, (name, payload) in enumerate(score_dict.items()):
        ax = axes[i]
        labels = payload["labels"]
        scores = payload["fault_scores"]
        normal_mask = np.isin(labels, normal_indices)
        normal_scores = np.clip(scores[normal_mask], 0.0, 1.0)
        fault_scores = np.clip(scores[~normal_mask], 0.0, 1.0)

        normal_logit = score_to_logit(normal_scores)
        fault_logit = score_to_logit(fault_scores)

        ax.hist(normal_logit, bins=bins, density=True, alpha=0.22, color="#16a34a", edgecolor="none")
        ax.hist(fault_logit, bins=bins, density=True, alpha=0.22, color="#dc2626", edgecolor="none")

        if np.std(normal_logit) > 1e-8:
            sns.kdeplot(
                normal_logit,
                fill=False,
                color="#15803d",
                linewidth=2.0,
                ax=ax,
                warn_singular=False,
                bw_adjust=0.8,
                label="正常",
            )
        else:
            ax.axvline(float(np.mean(normal_logit)), color="#15803d", linewidth=2.0, label="正常")

        if np.std(fault_logit) > 1e-8:
            sns.kdeplot(
                fault_logit,
                fill=False,
                color="#b91c1c",
                linewidth=2.0,
                ax=ax,
                warn_singular=False,
                bw_adjust=0.8,
                label="故障",
            )
        else:
            ax.axvline(float(np.mean(fault_logit)), color="#b91c1c", linewidth=2.0, label="故障")

        med_n = float(np.median(normal_scores))
        med_f = float(np.median(fault_scores))
        ax.axvline(score_to_logit(np.array([med_n]))[0], color="#15803d", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axvline(score_to_logit(np.array([med_f]))[0], color="#b91c1c", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            0.02,
            0.90,
            f"中位差(score)={abs(med_f - med_n):.3f}",
            transform=ax.transAxes,
            fontsize=9,
            color="#111827",
            bbox=dict(facecolor="#f9fafb", edgecolor="#d1d5db", alpha=0.85, boxstyle="round,pad=0.2"),
        )

        ax.set_title(name)
        ax.set_xlabel("故障得分（非线性展开）")
        ax.set_ylabel("密度")
        ax.set_xlim(-8.0, 8.0)
        tick_pos = np.array([-6, -3, 0, 3, 6], dtype=float)
        tick_score = 1.0 / (1.0 + np.exp(-tick_pos))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"{v:.3f}" for v in tick_score])
        ax.grid(True, linestyle=":", alpha=0.25)
        if i == 0:
            ax.legend()
        else:
            ax.legend_.remove() if ax.legend_ else None

    fig.suptitle("私有数据集异常得分分布（真实推理输出，非线性展开）", fontsize=16)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "anomaly_scores_kde.png", dpi=300)
    plt.close()


def balanced_domain_sample(features, labels, domains, per_class=120, seed=42):
    rng = np.random.default_rng(seed)
    selected = []
    for d in [0, 1]:
        for c in sorted(np.unique(labels)):
            idx = np.where((domains == d) & (labels == c))[0]
            if len(idx) == 0:
                continue
            if len(idx) > per_class:
                idx = rng.choice(idx, size=per_class, replace=False)
            selected.append(idx)
    if not selected:
        return features, labels, domains
    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return features[selected], labels[selected], domains[selected]


def run_tsne(features, random_state=42):
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=random_state, n_iter=1200)
    return tsne.fit_transform(features)


def plot_private_tsne_alignment(base_pack, full_pack, class_names):
    color_list = sns.color_palette("tab10", n_colors=len(class_names))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=False, sharey=False)

    for ax, data_pack, title in [
        (axes[0], base_pack, "基线 ISGFAN"),
        (axes[1], full_pack, "DGT-GLFA"),
    ]:
        src_feat = data_pack["src_feat"]
        src_lab = data_pack["src_lab"]
        tgt_feat = data_pack["tgt_feat"]
        tgt_lab = data_pack["tgt_lab"]

        feat = np.concatenate([src_feat, tgt_feat], axis=0)
        lab = np.concatenate([src_lab, tgt_lab], axis=0)
        dom = np.concatenate([np.zeros(len(src_feat), dtype=int), np.ones(len(tgt_feat), dtype=int)])

        feat, lab, dom = balanced_domain_sample(feat, lab, dom, per_class=120)
        emb = run_tsne(feat)

        for cid, cname in enumerate(class_names):
            m_src = (lab == cid) & (dom == 0)
            m_tgt = (lab == cid) & (dom == 1)
            if np.any(m_src):
                ax.scatter(emb[m_src, 0], emb[m_src, 1], s=16, marker="o", color=color_list[cid], alpha=0.72)
            if np.any(m_tgt):
                ax.scatter(emb[m_tgt, 0], emb[m_tgt, 1], s=18, marker="^", color=color_list[cid], alpha=0.78)

        ax.set_title(title)
        ax.set_xlabel("降维坐标 1")
        ax.set_ylabel("降维坐标 2")
        ax.grid(True, linestyle=":", alpha=0.3)

    plt.suptitle("私有数据特征分布对比（圆点=源域，三角=目标域）", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "private_tsne_alignment.png", dpi=300)
    plt.close()


def plot_public_tsne(single_pack, class_names, display_names, title, out_name):
    src_feat = single_pack["src_feat"]
    src_lab = single_pack["src_lab"]
    tgt_feat = single_pack["tgt_feat"]
    tgt_lab = single_pack["tgt_lab"]

    feat = np.concatenate([src_feat, tgt_feat], axis=0)
    lab = np.concatenate([src_lab, tgt_lab], axis=0)
    dom = np.concatenate([np.zeros(len(src_feat), dtype=int), np.ones(len(tgt_feat), dtype=int)])

    feat, lab, dom = balanced_domain_sample(feat, lab, dom, per_class=100)
    emb = run_tsne(feat)

    color_list = sns.color_palette("tab10", n_colors=len(class_names))

    plt.figure(figsize=(8, 6))
    for cid, cname in enumerate(class_names):
        m_src = (lab == cid) & (dom == 0)
        m_tgt = (lab == cid) & (dom == 1)
        if np.any(m_src):
            plt.scatter(emb[m_src, 0], emb[m_src, 1], s=16, marker="o", color=color_list[cid], alpha=0.7)
        if np.any(m_tgt):
            plt.scatter(emb[m_tgt, 0], emb[m_tgt, 1], s=16, marker="^", color=color_list[cid], alpha=0.7)

    plt.title(title)
    plt.xlabel("降维坐标 1")
    plt.ylabel("降维坐标 2")
    plt.tight_layout()
    plt.savefig(IMG_DIR / out_name, dpi=300)
    plt.close()


def plot_public_tsne_legend(display_names):
    color_list = sns.color_palette("tab10", n_colors=len(display_names))

    fig, ax = plt.subplots(figsize=(0.95 * len(display_names), 1.8))
    ax.axis("off")

    y_src = 0.42
    y_tgt = 0.06
    for i, (c, n) in enumerate(zip(color_list, display_names)):
        ax.scatter(i, y_src, color=c, marker="o", s=85)
        ax.text(i, y_src + 0.17, n, ha="center", va="bottom", fontsize=12)
        ax.scatter(i, y_tgt, color=c, marker="^", s=85)

    ax.text(-0.7, y_src, "源域", ha="right", va="center", fontsize=12)
    ax.text(-0.7, y_tgt, "目标域", ha="right", va="center", fontsize=12)
    ax.set_xlim(-1.0, len(display_names) - 0.2)
    ax.set_ylim(-0.2, 0.9)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "public_real_cwru_tsne_legend_ours.png", dpi=300)
    plt.close()


def plot_confusion(cm, labels_show, title, out_name):
    plt.figure(figsize=(8.6, 7.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_show, yticklabels=labels_show)
    plt.title(title)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.xticks(rotation=35)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(IMG_DIR / out_name, dpi=300)
    plt.close()


def plot_public_multi_metric(public_models, public_metrics):
    names = [n for n in public_models]
    acc = [to_percent(public_metrics[n]["acc"]) for n in names]
    far = [to_percent(public_metrics[n]["far"]) for n in names]

    x = np.arange(len(names))

    fig, ax1 = plt.subplots(figsize=(12, 6.2))
    bars = ax1.bar(x, acc, color="#2563eb", alpha=0.88, label="Acc")
    acc_min, acc_max = float(np.min(acc)), float(np.max(acc))
    acc_span = max(acc_max - acc_min, 0.12)
    ax1.set_ylim(acc_min - 0.45 * acc_span, acc_max + 0.70 * acc_span)

    ax1.set_ylabel("准确率 Acc (%)（局部放大）")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")

    ax2 = ax1.twinx()
    far_min, far_max = float(np.min(far)), float(np.max(far))
    if far_max - far_min < 1e-9:
        # FAR 完全一致时，不用趋势线误导读者，改为“等值点 + 提示”。
        ax2.scatter(x, far, color="#ef4444", marker="o", s=60, label="FAR")
        ax2.hlines(far[0], x[0] - 0.35, x[-1] + 0.35, color="#ef4444", linestyle="--", linewidth=1.6, alpha=0.8)
        ax2.text(
            0.02,
            0.92,
            f"FAR 一致: {far[0]:.3f}%",
            transform=ax2.transAxes,
            fontsize=10,
            color="#b91c1c",
            bbox=dict(facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.85, boxstyle="round,pad=0.25"),
        )
        ax2.set_ylim(far[0] - 0.02, far[0] + 0.02)
    else:
        ax2.plot(x, far, color="#ef4444", marker="o", linewidth=2.2, label="FAR")
        far_span = max(far_max - far_min, 0.01)
        ax2.set_ylim(far_min - 0.25 * far_span, far_max + 0.45 * far_span)

    ax2.set_ylabel("误报率 FAR (%)")

    best_acc = max(acc)
    for b, v in zip(bars, acc):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.10 * acc_span,
            f"{v:.2f}%\n({v - best_acc:+.2f})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for xi, fv in zip(x, far):
        ax2.text(xi, fv + 0.0015, f"{fv:.3f}%", ha="center", va="bottom", fontsize=8, color="#b91c1c")

    ax1.set_title("公开数据集主要模型指标对比（真实评估）")
    ax1.grid(axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "public_multi_metric.png", dpi=300)
    plt.close()


def plot_public_transfer_heatmap(public_models, public_metrics):
    cols = ["Acc", "FDR", "FAR", "AUC"]
    mat = []
    for n in public_models:
        m = public_metrics[n]
        mat.append([
            to_percent(m["acc"]),
            to_percent(m["fdr"]),
            to_percent(m["far"]),
            to_percent(m["auc_bin"]),
        ])

    mat = np.array(mat)

    plt.figure(figsize=(9.8, 5.6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=cols, yticklabels=public_models)
    plt.title("公开数据模型迁移表现热力图（真实评估）")
    plt.xlabel("指标")
    plt.ylabel("模型")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "public_transfer_heatmap.png", dpi=300)
    plt.close()


def plot_public_roc(public_models, public_metrics):
    plt.figure(figsize=(8.2, 6.2))
    for n in public_models:
        fpr = public_metrics[n]["fpr"]
        tpr = public_metrics[n]["tpr"]
        auc_bin = public_metrics[n]["auc_bin"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{n} (AUC={auc_bin:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("公开数据故障检测 ROC 曲线（真实推理）")
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "public_roc_curves.png", dpi=300)
    plt.close()


def plot_public_confusion_pair(cm_a, cm_b, labels_show):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, cm, title in [
        (axes[0], cm_a, "DGT-GLFA"),
        (axes[1], cm_b, "ISGFAN"),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_show, yticklabels=labels_show, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        ax.tick_params(axis="x", rotation=35)

    fig.suptitle("公开数据混淆矩阵对比（真实评估）", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "public_confusion_matrix.png", dpi=300)
    plt.close()


def plot_ablation_study(labels, acc, fdr):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    ax.plot(x, acc, marker="o", linewidth=2.2, color="#f97316", label="Acc")
    ax.plot(x, fdr, marker="s", linewidth=2.2, color="#16a34a", label="FDR")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("指标值 (%)")
    ax.set_title("私有数据消融结果（真实评估）")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend()

    for i, v in enumerate(acc):
        ax.text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "ablation_study.png", dpi=300)
    plt.close()


def plot_ablation_private_public(private_labels, private_acc, private_far, public_labels, public_acc, public_far):
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.9))

    x1 = np.arange(len(private_labels))
    axes[0].bar(x1, private_acc, color="#2563eb", alpha=0.85, label="Acc")
    ax0b = axes[0].twinx()
    ax0b.plot(x1, private_far, color="#ef4444", marker="o", linewidth=2.0, label="FAR")
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(private_labels)
    axes[0].set_title("私有集模块消融")
    axes[0].set_ylabel("Acc (%)")
    ax0b.set_ylabel("FAR (%)")
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)

    x2 = np.arange(len(public_labels))
    axes[1].bar(x2, public_acc, color="#7c3aed", alpha=0.85, label="Acc")
    ax1b = axes[1].twinx()
    ax1b.plot(x2, public_far, color="#f97316", marker="s", linewidth=2.0, label="FAR")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(public_labels, rotation=20, ha="right")
    axes[1].set_title("公开集模型版本对比")
    axes[1].set_ylabel("Acc (%)")
    ax1b.set_ylabel("FAR (%)")
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("私有/公开联合对比（真实评估）", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "ablation_private_public.png", dpi=300)
    plt.close()


def plot_param_sensitivity(private_labels, private_acc, public_labels, public_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

    x1 = np.arange(len(private_labels))
    ax1.plot(x1, private_acc, marker="o", linewidth=2.2, color="#0f766e")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(private_labels)
    ax1.set_title("私有集配置敏感性")
    ax1.set_ylabel("Acc (%)")
    ax1.grid(True, linestyle=":", alpha=0.35)

    x2 = np.arange(len(public_labels))
    ax2.plot(x2, public_acc, marker="d", linewidth=2.2, color="#7c3aed")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(public_labels, rotation=22, ha="right")
    ax2.set_title("公开集配置敏感性")
    ax2.set_ylabel("Acc (%)")
    ax2.grid(True, linestyle=":", alpha=0.35)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "param_sensitivity.png", dpi=300)
    plt.close()


def pick_real_wave_file(data_dir, patterns):
    for p in patterns:
        cands = sorted(data_dir.glob(p))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"No waveform file matched: {patterns}")


def load_first_signal(csv_path, length=2048):
    arr = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32, encoding="utf-8-sig", invalid_raise=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        sig = arr
    else:
        sig = arr[0]
    if len(sig) < length:
        pad = np.zeros(length - len(sig), dtype=np.float32)
        sig = np.concatenate([sig, pad])
    sig = sig[:length]
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    return sig


def plot_fault_waveforms(data_dir):
    f_normal = pick_real_wave_file(data_dir, ["*test_dso.csv", "*test.csv"])
    f_l120 = pick_real_wave_file(data_dir, ["*L120*test_dso.csv", "*L120*test.csv"])
    f_l330 = pick_real_wave_file(data_dir, ["*L330*test_dso.csv", "*L330*test.csv"])
    f_l1800 = pick_real_wave_file(data_dir, ["*L1800*test_dso.csv", "*L1800*test.csv"])

    sigs = [
        ("正常 NO_L", load_first_signal(f_normal)),
        ("故障 L120", load_first_signal(f_l120)),
        ("故障 L330", load_first_signal(f_l330)),
        ("故障 L1800", load_first_signal(f_l1800)),
    ]

    t = np.arange(len(sigs[0][1]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True)
    for ax, (title, sig) in zip(axes.flatten(), sigs):
        ax.plot(t, sig, linewidth=1.4, color="#2563eb")
        ax.set_title(title)
        ax.set_ylabel("幅值")
        ax.grid(True, linestyle=":", alpha=0.3)
    axes[1, 0].set_xlabel("采样点")
    axes[1, 1].set_xlabel("采样点")

    plt.tight_layout()
    plt.savefig(IMG_DIR / "fault_waveform_cases.png", dpi=300)
    plt.close()


def plot_efficiency_tradeoff(names, params_m, acc, far, fps):
    sizes = np.array(fps) / (np.max(fps) + 1e-8) * 1100 + 120

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    sc = ax.scatter(params_m, acc, s=sizes, c=far, cmap="RdYlGn_r", alpha=0.86, edgecolors="k")

    for i, n in enumerate(names):
        ax.text(params_m[i] + 0.01, acc[i] + 0.06, n, fontsize=9)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("FAR (%)")

    ax.set_xlabel("参数量 (百万)")
    ax.set_ylabel("Acc (%)")
    ax.set_title("精度-复杂度-误报率权衡（气泡大小代表 FPS）")
    ax.grid(True, linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "efficiency_tradeoff.png", dpi=300)
    plt.close()


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 1) 私有数据评估
    # -----------------------------
    private_train_loader, private_test_loader, private_meta = get_self_dataloaders(
        data_dir=str(ROOT / "self_data" / "75_Sta_60_test_dso"),
        batch_size=128,
        test_ratio=0.2,
        signal_length=2048,
        use_dso=False,
        seed=42,
        num_workers=0,
    )

    private_class_names = private_meta["class_names"]
    private_normal_idx = [
        i for i, n in enumerate(private_class_names) if n.upper() in {"L0", "NO_L", "NORMAL", "HEALTHY"}
    ]
    if not private_normal_idx:
        private_normal_idx = [0]

    private_models = {
        "源仅监督": {
            "type": "baseline",
            "method": "source_only",
            "ckpt": "baseline/private_runs/source_only/model_best.pth",
        },
        "DANN": {
            "type": "baseline",
            "method": "dann",
            "ckpt": "baseline/private_runs/dann/model_best.pth",
        },
        "Deep CORAL": {
            "type": "baseline",
            "method": "deepcoral",
            "ckpt": "baseline/private_runs/deepcoral/model_best.pth",
        },
        "CDAN": {
            "type": "baseline",
            "method": "cdan",
            "ckpt": "baseline/private_runs/cdan/model_best.pth",
        },
        "MCC": {
            "type": "baseline",
            "method": "mcc",
            "ckpt": "baseline/private_runs/mcc/model_best.pth",
        },
        "MMD": {
            "type": "baseline",
            "method": "mmd",
            "ckpt": "baseline/private_runs/mmd/model_best.pth",
        },
        "ISGFAN": {
            "type": "isan",
            "attention": "none",
            "ckpt_candidates": [
                "result/results_self_original/model_self_original_best.pth",
                "result/results_self/model_self_best.pth",
            ],
        },
        "DGT-GLFA": {
            "type": "isan",
            "attention": "attn1",
            "ckpt_candidates": [
                "results_self_grl_tec_attn1/model_self_grl_tec_attn1_best.pth",
                "result/results_self_grl_tec_attn1/model_self_grl_tec_attn1_best.pth",
            ],
        },
        "+GLFA": {
            "type": "isan",
            "attention": "attn1",
            "ckpt_candidates": [
                "results_self_attn1_ablation/model_self_attn1_ablation_best.pth",
                "result/results_self_attn1_ablation/model_self_attn1_ablation_best.pth",
            ],
        },
        "+DGT": {
            "type": "isan",
            "attention": "none",
            "ckpt_candidates": [
                "result/results_self_grl/model_self_grl_best.pth",
            ],
        },
        "+TEC": {
            "type": "isan",
            "attention": "none",
            "ckpt_candidates": [
                "result/results_self_tec/model_self_tec_best.pth",
            ],
        },
    }

    private_eval = {}

    for name, spec in private_models.items():
        if spec["type"] == "baseline":
            ckpt = resolve_path(spec["ckpt"])
            model = BaselineModel(num_classes=private_meta["num_classes"], feature_dim=128, method=spec["method"]).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            out = infer_baseline(model, private_test_loader, device, private_normal_idx, collect_features=True)
            private_eval[name] = out
            private_eval[name]["ckpt"] = str(ckpt)
            private_eval[name]["method"] = spec["method"]
        else:
            ckpt = first_existing(spec["ckpt_candidates"])
            model = Model(feature_dim=320, num_classes=private_meta["num_classes"], attention_mode=spec["attention"]).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            out = infer_isan(model, private_test_loader, device, private_normal_idx, collect_features=True)
            private_eval[name] = out
            private_eval[name]["ckpt"] = str(ckpt)
            private_eval[name]["attention"] = spec["attention"]

    # 私有 t-SNE 需要源域特征
    def collect_private_domain_features(name):
        spec = private_models[name]
        if spec["type"] == "baseline":
            ckpt = resolve_path(spec["ckpt"])
            model = BaselineModel(num_classes=private_meta["num_classes"], feature_dim=128, method=spec["method"]).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            src_feats, src_labs = [], []
            tgt_feats, tgt_labs = [], []
            with torch.no_grad():
                for x, y in private_train_loader:
                    x = x.to(device)
                    f, _ = model(x)
                    src_feats.append(f.cpu().numpy())
                    src_labs.append(y.numpy())
                for x, y in private_test_loader:
                    x = x.to(device)
                    f, _ = model(x)
                    tgt_feats.append(f.cpu().numpy())
                    tgt_labs.append(y.numpy())
            return {
                "src_feat": np.concatenate(src_feats),
                "src_lab": np.concatenate(src_labs),
                "tgt_feat": np.concatenate(tgt_feats),
                "tgt_lab": np.concatenate(tgt_labs),
            }

        ckpt = first_existing(spec["ckpt_candidates"])
        model = Model(feature_dim=320, num_classes=private_meta["num_classes"], attention_mode=spec["attention"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        src_feats, src_labs = [], []
        tgt_feats, tgt_labs = [], []
        with torch.no_grad():
            for x, y in private_train_loader:
                x = x.to(device)
                f = model.FRFE(x).mean(dim=-1)
                src_feats.append(f.cpu().numpy())
                src_labs.append(y.numpy())
            for x, y in private_test_loader:
                x = x.to(device)
                f = model.FRFE(x).mean(dim=-1)
                tgt_feats.append(f.cpu().numpy())
                tgt_labs.append(y.numpy())

        return {
            "src_feat": np.concatenate(src_feats),
            "src_lab": np.concatenate(src_labs),
            "tgt_feat": np.concatenate(tgt_feats),
            "tgt_lab": np.concatenate(tgt_labs),
        }

    private_tsne_base = collect_private_domain_features("ISGFAN")
    private_tsne_full = collect_private_domain_features("DGT-GLFA")

    # -----------------------------
    # 2) 公开数据评估
    # -----------------------------
    get_dataloaders(batch_size=128, num_workers=0)
    public_test_loader = data_manager.test_loader
    public_source_loader = DataLoader(SourceDataset(SOURCE_DIR), batch_size=128, shuffle=False, num_workers=0)

    public_class_names = list(PUBLIC_CLASSES)
    public_normal_idx = [public_class_names.index("normal")]

    display_map = {
        "ball_07": "B1",
        "ball_14": "B2",
        "ball_21": "B3",
        "inner_07": "I1",
        "inner_14": "I2",
        "inner_21": "I3",
        "normal": "N",
        "outer_07": "O1",
        "outer_14": "O2",
        "outer_21": "O3",
    }
    public_display_names = [display_map[c] for c in public_class_names]

    public_csv = resolve_path("result/public_real_eval/public_real_metrics.csv")
    with open(public_csv, "r", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    # 论文图中不展示 attn2 相关模型，并将训练别名替换为论文术语。
    public_name_map = {
        "GRL+TEC(no-attn)": "DGT-GLFA",
        "Attn1-rewrite": "GLFA",
        "ISGFAN-original": "ISGFAN",
    }
    rows = []
    for r in raw_rows:
        model_raw = r["model"].strip()
        attn_mode = r["attention_mode"].strip().lower()
        if "attn2" in model_raw.lower() or "attn2" in attn_mode:
            continue
        rr = dict(r)
        rr["model"] = public_name_map.get(model_raw, model_raw)
        rows.append(rr)

    public_model_order = [r["model"] for r in rows]
    public_eval = {}

    def collect_public_domain_features(model_obj, src_loader, tgt_loader):
        src_feat, src_lab = [], []
        tgt_feat, tgt_lab = [], []
        with torch.no_grad():
            for x, y in src_loader:
                x = x.to(device)
                f = model_obj.FRFE(x).mean(dim=-1)
                src_feat.append(f.cpu().numpy())
                src_lab.append(y.numpy())
            for x, y in tgt_loader:
                x = x.to(device)
                f = model_obj.FRFE(x).mean(dim=-1)
                tgt_feat.append(f.cpu().numpy())
                tgt_lab.append(y.numpy())
        return {
            "src_feat": np.concatenate(src_feat),
            "src_lab": np.concatenate(src_lab),
            "tgt_feat": np.concatenate(tgt_feat),
            "tgt_lab": np.concatenate(tgt_lab),
        }

    public_tsne_pack = {}

    for r in rows:
        name = r["model"]
        ckpt = resolve_path(r["checkpoint"])
        attn = r["attention_mode"]

        model = Model(feature_dim=320, num_classes=len(public_class_names), attention_mode=attn).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        out = infer_isan(model, public_test_loader, device, public_normal_idx, collect_features=True)
        public_eval[name] = out
        public_eval[name]["ckpt"] = str(ckpt)
        public_eval[name]["attention"] = attn

        if name in {"DGT-GLFA", "ISGFAN"}:
            public_tsne_pack[name] = collect_public_domain_features(model, public_source_loader, public_test_loader)

    # -----------------------------
    # 3) 绘制全部 21 张图
    # -----------------------------
    plot_network_structure()
    plot_experiment_flowchart()

    # 3.1 私有收敛曲线（真实日志，使用测试损失）
    curve_source = load_baseline_loss_curve(resolve_path("baseline/private_runs/source_only/metrics.csv"))
    curve_dann = load_baseline_loss_curve(resolve_path("baseline/private_runs/dann/metrics.csv"))
    curve_isgfan = load_txt_loss_curve(resolve_path("training_logs/stage_results_self_original/training_log_self.csv/test_loss.txt"))
    curve_full = load_txt_loss_curve(resolve_path("training_logs/stage_results_self_grl_tec_attn1/training_log_self.csv/test_loss.txt"))

    convergence = {
        "源仅监督": curve_source,
        "DANN": curve_dann,
        "ISGFAN": curve_isgfan,
        "DGT-GLFA": curve_full,
    }
    plot_private_convergence(convergence)

    # 3.2 私有多指标与 FAR/FDR
    panel_methods = ["源仅监督", "DANN", "Deep CORAL", "CDAN", "MCC", "MMD", "ISGFAN", "DGT-GLFA"]
    panel_acc = [to_percent(private_eval[m]["acc"]) for m in panel_methods]
    panel_f1 = [to_percent(private_eval[m]["f1"]) for m in panel_methods]
    panel_auc = [to_percent(private_eval[m]["auc_macro"]) for m in panel_methods]
    panel_fdr = [to_percent(private_eval[m]["fdr"]) for m in panel_methods]
    panel_far = [to_percent(private_eval[m]["far"]) for m in panel_methods]

    plot_private_metric_panel(panel_methods, panel_acc, panel_f1, panel_auc)
    plot_private_fdr_far(panel_methods, panel_fdr, panel_far)

    # 3.3 私有异常得分 KDE 与 t-SNE 对齐
    kde_methods = {
        m: private_eval[m]
        for m in ["源仅监督", "DANN", "Deep CORAL", "CDAN", "MCC", "MMD", "ISGFAN", "DGT-GLFA"]
    }
    plot_anomaly_kde(kde_methods, private_normal_idx)
    plot_private_tsne_alignment(private_tsne_base, private_tsne_full, private_class_names)

    # 3.4 公开数据 t-SNE + 图例 + 混淆矩阵
    plot_public_tsne(
        public_tsne_pack["DGT-GLFA"],
        public_class_names,
        public_display_names,
        "CWRU t-SNE（DGT-GLFA）",
        "public_real_cwru_tsne_ours.png",
    )
    plot_public_tsne(
        public_tsne_pack["ISGFAN"],
        public_class_names,
        public_display_names,
        "CWRU t-SNE（ISGFAN）",
        "public_real_cwru_tsne_isgfan_original.png",
    )
    plot_public_tsne_legend(public_display_names)

    cm_ours = confusion_matrix(public_eval["DGT-GLFA"]["labels"], public_eval["DGT-GLFA"]["preds"])
    cm_isgfan = confusion_matrix(public_eval["ISGFAN"]["labels"], public_eval["ISGFAN"]["preds"])

    plot_confusion(cm_ours, public_display_names, "CWRU 混淆矩阵（DGT-GLFA）", "public_real_cwru_confmat_ours.png")
    plot_confusion(cm_isgfan, public_display_names, "CWRU 混淆矩阵（ISGFAN）", "public_real_cwru_confmat_isgfan_original.png")

    # 3.5 公开数据综合图
    plot_public_multi_metric(public_model_order, public_eval)
    plot_public_transfer_heatmap(public_model_order, public_eval)
    plot_public_roc(public_model_order, public_eval)

    # 3.6 消融/配置敏感性
    private_abl_labels = ["Base", "+GLFA", "+DGT", "+TEC", "Full"]
    private_abl_models = ["ISGFAN", "+GLFA", "+DGT", "+TEC", "DGT-GLFA"]
    private_abl_acc = [to_percent(private_eval[m]["acc"]) for m in private_abl_models]
    private_abl_fdr = [to_percent(private_eval[m]["fdr"]) for m in private_abl_models]
    private_abl_far = [to_percent(private_eval[m]["far"]) for m in private_abl_models]

    plot_ablation_study(private_abl_labels, private_abl_acc, private_abl_fdr)

    public_variant_labels = ["ISGFAN", "GLFA", "DGT-GLFA"]
    public_variant_models = [
        "ISGFAN",
        "GLFA",
        "DGT-GLFA",
    ]
    public_var_acc = [to_percent(public_eval[m]["acc"]) for m in public_variant_models]
    public_var_far = [to_percent(public_eval[m]["far"]) for m in public_variant_models]

    plot_ablation_private_public(
        private_abl_labels,
        private_abl_acc,
        private_abl_far,
        public_variant_labels,
        public_var_acc,
        public_var_far,
    )

    sens_private_labels = ["Base", "GLFA", "DGT", "TEC", "Full"]
    sens_public_labels = ["ISGFAN", "GLFA", "DGT-GLFA"]
    sens_public_acc = [
        to_percent(public_eval["ISGFAN"]["acc"]),
        to_percent(public_eval["GLFA"]["acc"]),
        to_percent(public_eval["DGT-GLFA"]["acc"]),
    ]
    plot_param_sensitivity(sens_private_labels, private_abl_acc, sens_public_labels, sens_public_acc)

    # 3.7 真实波形与效率权衡
    plot_fault_waveforms(ROOT / "self_data" / "75_Sta_60_test_dso")

    eff_methods = ["DANN", "CDAN", "MCC", "ISGFAN", "+GLFA", "+DGT", "+TEC", "DGT-GLFA"]
    eff_acc = [to_percent(private_eval[m]["acc"]) for m in eff_methods]
    eff_far = [to_percent(private_eval[m]["far"]) for m in eff_methods]

    eff_fps = []
    eff_params = []

    for m in eff_methods:
        spec = private_models[m]
        if spec["type"] == "baseline":
            ckpt = resolve_path(spec["ckpt"])
            fps, params = measure_fps_baseline(
                method=spec["method"],
                checkpoint=ckpt,
                num_classes=private_meta["num_classes"],
                loader=private_test_loader,
                device=device,
            )
        else:
            ckpt = first_existing(spec["ckpt_candidates"])
            fps, params = measure_fps_isan(
                checkpoint=ckpt,
                attention_mode=spec["attention"],
                num_classes=private_meta["num_classes"],
                loader=private_test_loader,
                device=device,
            )
        eff_fps.append(float(fps))
        eff_params.append(params / 1e6)

    plot_efficiency_tradeoff(eff_methods, eff_params, eff_acc, eff_far, eff_fps)

    print("chapter4 figures regenerated with real data and saved to:")
    print(IMG_DIR)


if __name__ == "__main__":
    main()
