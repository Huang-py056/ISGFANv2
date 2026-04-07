import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Data.self_dataset import get_self_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Private dataset baseline runner")
    parser.add_argument("--method", type=str, required=True, choices=[
        "source_only",
        "dann",
        "cdan",
        "deepcoral",
        "mmd",
        "jan",
        "entropy_min",
        "mcc",
    ])
    parser.add_argument("--data-dir", type=str, default="self_data/75_Sta_60_test_dso")
    parser.add_argument("--source-use-dso", action="store_true", help="Use *_dso.csv as source domain. Default uses target-train split as source for stability.")
    parser.add_argument("--target-use-dso", action="store_true", help="Use *_dso.csv as target domain/test split.")
    parser.add_argument("--signal-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-steps-per-epoch", type=int, default=80, help="Limit optimization steps per epoch to keep runs tractable.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--lambda-transfer", type=float, default=1.0)
    parser.add_argument("--lambda-adv-cdan", type=float, default=0.08, help="Adversarial loss weight used by CDAN baseline")
    parser.add_argument("--lambda-transfer-mcc", type=float, default=0.08, help="Transfer weight used by MCC baseline")
    parser.add_argument("--lambda-transfer-mmd", type=float, default=0.005, help="Transfer weight used by MMD baseline")
    parser.add_argument("--lambda-transfer-jan", type=float, default=0.005, help="Transfer weight used by JAN baseline")
    parser.add_argument("--transfer-warmup-epochs", type=int, default=20, help="Warmup epochs for transfer weight ramp-up")
    parser.add_argument("--adapt-peak-ratio", type=float, default=0.45, help="Progress ratio where adaptation weight reaches peak")
    parser.add_argument("--adapt-end-ratio", type=float, default=0.85, help="Progress ratio where adaptation weight decays to zero")
    parser.add_argument("--mmd-jan-balance-source", action="store_true", help="Use class-balanced source sampler for MMD/JAN")
    parser.add_argument("--mmd-jan-class-weighted-ce", action="store_true", help="Use class-weighted CE for MMD/JAN")
    parser.add_argument("--lambda-adv", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--print-interval", type=int, default=5)
    parser.add_argument("--output-root", type=str, default="baseline/private_runs")
    parser.add_argument("--log-file", type=str, default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_coeff):
        ctx.lambda_coeff = lambda_coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_coeff * grad_output, None


def gradient_reverse(x, lambda_coeff):
    return GRL.apply(x, lambda_coeff)


class ConvFeatureExtractor(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        x = self.proj(x)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = max(128, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class BaselineModel(nn.Module):
    def __init__(self, num_classes, feature_dim, method):
        super().__init__()
        self.method = method
        self.feature = ConvFeatureExtractor(out_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.domain_discriminator = None
        if method == "dann":
            self.domain_discriminator = DomainDiscriminator(feature_dim)
        elif method == "cdan":
            self.domain_discriminator = DomainDiscriminator(feature_dim * num_classes)

    def forward(self, x):
        feat = self.feature(x)
        logits = self.classifier(feat)
        return feat, logits


def covariance(features):
    n = features.size(0)
    if n <= 1:
        return torch.zeros(features.size(1), features.size(1), device=features.device)
    mean = torch.mean(features, dim=0, keepdim=True)
    centered = features - mean
    return centered.t().mm(centered) / (n - 1)


def coral_loss(source_feat, target_feat):
    d = source_feat.size(1)
    src_cov = covariance(source_feat)
    tgt_cov = covariance(target_feat)
    return torch.sum((src_cov - tgt_cov) ** 2) / (4.0 * d * d)


def _rbf_kernel_matrix(x, y, bandwidth, kernel_mul=2.0, kernel_num=5):
    # Supports unequal batch sizes between x and y.
    sq_dist = torch.cdist(x, y, p=2).pow(2)
    base = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = []
    for i in range(kernel_num):
        bw = base * (kernel_mul ** i)
        kernels.append(torch.exp(-sq_dist / (bw + 1e-8)))
    return sum(kernels)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5):
    if source.numel() == 0 or target.numel() == 0:
        return torch.tensor(0.0, device=source.device if source.numel() > 0 else target.device)

    all_feat = torch.cat([source, target], dim=0)
    pair_sq = torch.cdist(all_feat, all_feat, p=2).pow(2)
    bandwidth = pair_sq.detach().mean()
    if (not torch.isfinite(bandwidth)) or bandwidth.item() <= 0:
        bandwidth = torch.tensor(1.0, device=source.device, dtype=source.dtype)

    k_ss = _rbf_kernel_matrix(source, source, bandwidth, kernel_mul=kernel_mul, kernel_num=kernel_num)
    k_tt = _rbf_kernel_matrix(target, target, bandwidth, kernel_mul=kernel_mul, kernel_num=kernel_num)
    k_st = _rbf_kernel_matrix(source, target, bandwidth, kernel_mul=kernel_mul, kernel_num=kernel_num)

    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()


def mcc_loss(probs):
    if probs.numel() == 0:
        return torch.tensor(0.0, device=probs.device)
    conf_matrix = probs.t().mm(probs)
    conf_matrix = conf_matrix / (conf_matrix.sum(dim=1, keepdim=True) + 1e-8)
    return (conf_matrix.sum() - torch.trace(conf_matrix)) / probs.size(1)


def compute_grl_lambda(epoch_idx, step_idx, total_epochs, steps_per_epoch):
    progress = (epoch_idx * steps_per_epoch + step_idx) / max(total_epochs * steps_per_epoch, 1)
    return 2.0 / (1.0 + math.exp(-10 * progress)) - 1.0


def find_normal_class_indices(class_names):
    normal_indices = []
    for idx, name in enumerate(class_names):
        up_name = name.upper()
        if up_name in {"L0", "NO_L", "NORMAL", "HEALTHY"}:
            normal_indices.append(idx)
    if normal_indices:
        return normal_indices
    for idx, name in enumerate(class_names):
        if "L0" in name.upper():
            return [idx]
    return [0]


def evaluate(model, loader, criterion, device, normal_class_indices):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    normal_mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in normal_class_indices:
        normal_mask = normal_mask | (labels == idx)
    fault_mask = ~normal_mask

    is_pred_normal = torch.zeros_like(preds, dtype=torch.bool)
    for idx in normal_class_indices:
        is_pred_normal = is_pred_normal | (preds == idx)

    false_alarm_count = ((~is_pred_normal) & normal_mask).sum().item()
    detected_fault_count = ((~is_pred_normal) & fault_mask).sum().item()

    normal_total = max(int(normal_mask.sum().item()), 1)
    fault_total = max(int(fault_mask.sum().item()), 1)

    far = false_alarm_count / normal_total
    fdr = detected_fault_count / fault_total

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc, far, fdr


def setup_logger(log_file):
    logger = logging.getLogger("baseline_runner")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def conditional_map(feat, logits):
    probs = torch.softmax(logits, dim=1)
    return torch.bmm(probs.unsqueeze(2), feat.unsqueeze(1)).reshape(feat.size(0), -1)


def adaptation_scale(progress, peak_ratio, end_ratio):
    # Triangular schedule: ramp-up -> peak -> ramp-down to avoid late-stage over-alignment collapse.
    p = min(max(progress, 0.0), 1.0)
    peak = min(max(peak_ratio, 1e-4), 0.99)
    end = min(max(end_ratio, peak + 1e-4), 1.0)

    if p <= peak:
        return p / peak
    if p >= end:
        return 0.0
    return (end - p) / (end - peak)


def run(args):
    set_seed(args.seed)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    method_out_dir = os.path.join(args.output_root, args.method)
    os.makedirs(method_out_dir, exist_ok=True)

    log_file = args.log_file if args.log_file else os.path.join("baseline", "logs", f"{args.method}.log")
    logger = setup_logger(log_file)

    logger.info(f"method={args.method}, device={device}, data_dir={args.data_dir}")
    logger.info(
        f"epochs={args.epochs}, batch_size={args.batch_size}, max_steps_per_epoch={args.max_steps_per_epoch}, "
        f"lr={args.lr}, min_lr={args.min_lr}, source_use_dso={args.source_use_dso}, target_use_dso={args.target_use_dso}"
    )

    target_train_loader, target_test_loader, target_meta = get_self_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        signal_length=args.signal_length,
        use_dso=args.target_use_dso,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    if args.source_use_dso:
        source_train_loader, _, source_meta = get_self_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            signal_length=args.signal_length,
            use_dso=True,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    else:
        source_train_loader = DataLoader(
            target_train_loader.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )
        source_meta = {
            "class_names": target_meta["class_names"],
            "num_classes": target_meta["num_classes"],
            "num_train": len(target_train_loader.dataset),
            "num_test": target_meta["num_test"],
        }

    if source_meta["class_names"] != target_meta["class_names"]:
        raise RuntimeError(
            "Class name mismatch between source and target splits. "
            f"source={source_meta['class_names']}, target={target_meta['class_names']}"
        )

    class_names = source_meta["class_names"]
    num_classes = source_meta["num_classes"]
    normal_class_indices = find_normal_class_indices(class_names)

    logger.info(f"class_names={class_names}")
    logger.info(f"source_train={source_meta['num_train']}, target_train={target_meta['num_train']}, target_test={target_meta['num_test']}")
    normal_names = [class_names[idx] for idx in normal_class_indices]
    logger.info(f"normal_class_indices={normal_class_indices}, normal_class_names={normal_names}")

    src_labels_tensor = torch.tensor(source_train_loader.dataset.labels, dtype=torch.long)
    class_counts = torch.bincount(src_labels_tensor, minlength=num_classes).float()
    logger.info(f"source_class_counts={class_counts.tolist()}")

    # Rebuild train loaders with drop_last=False to support tiny source-domain splits.
    source_batch_size = min(args.batch_size, max(len(source_train_loader.dataset), 1))
    target_batch_size = min(args.batch_size, max(len(target_train_loader.dataset), 1))
    source_train_loader = DataLoader(
        source_train_loader.dataset,
        batch_size=source_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )
    target_train_loader = DataLoader(
        target_train_loader.dataset,
        batch_size=target_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    # Optional balancing for MMD/JAN when explicitly enabled.
    if args.method in {"mmd", "jan"} and args.mmd_jan_balance_source:
        sample_weights = (1.0 / (class_counts[src_labels_tensor] + 1e-8)).double()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(source_train_loader.dataset),
            replacement=True,
        )
        source_train_loader = DataLoader(
            source_train_loader.dataset,
            batch_size=source_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )

    model = BaselineModel(num_classes=num_classes, feature_dim=args.feature_dim, method=args.method).to(device)

    if args.method in {"mmd", "jan"} and args.mmd_jan_class_weighted_ce:
        class_weights = class_counts.sum() / (class_counts * num_classes + 1e-8)
        class_weights = class_weights / class_weights.mean()
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info(f"class_weights_for_ce={class_weights.tolist()}")
    else:
        criterion_cls = nn.CrossEntropyLoss()
    criterion_dom = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    metrics_csv = os.path.join(method_out_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "epoch", "train_cls_loss", "train_transfer_loss", "train_adv_loss", "train_total_loss",
            "test_loss", "test_acc", "FAR", "FDR", "lr"
        ])

        best_acc = -1.0
        best_state = None

        steps_per_epoch = min(len(source_train_loader), len(target_train_loader))
        if args.max_steps_per_epoch > 0:
            steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
        if steps_per_epoch <= 0:
            raise RuntimeError(
                "No training steps available. "
                f"source_batches={len(source_train_loader)}, target_batches={len(target_train_loader)}"
            )
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            src_iter = iter(source_train_loader)
            tgt_iter = iter(target_train_loader)

            cls_sum = 0.0
            transfer_sum = 0.0
            adv_sum = 0.0
            total_sum = 0.0

            for step in range(steps_per_epoch):
                progress = ((epoch - 1) * steps_per_epoch + step) / max(args.epochs * steps_per_epoch, 1)
                adapt_scale = adaptation_scale(progress, args.adapt_peak_ratio, args.adapt_end_ratio)

                x_s, y_s = next(src_iter)
                x_t, _ = next(tgt_iter)

                x_s = x_s.to(device)
                y_s = y_s.to(device)
                x_t = x_t.to(device)

                optimizer.zero_grad()

                feat_s, logits_s = model(x_s)
                feat_t, logits_t = model(x_t)

                loss_cls = criterion_cls(logits_s, y_s)
                loss_transfer = torch.tensor(0.0, device=device)
                loss_adv = torch.tensor(0.0, device=device)

                if args.method == "dann":
                    grl_lambda = compute_grl_lambda(epoch - 1, step, args.epochs, steps_per_epoch)
                    dom_in = torch.cat([feat_s, feat_t], dim=0)
                    dom_logits = model.domain_discriminator(gradient_reverse(dom_in, grl_lambda))
                    dom_labels = torch.cat([
                        torch.zeros(feat_s.size(0), dtype=torch.long, device=device),
                        torch.ones(feat_t.size(0), dtype=torch.long, device=device),
                    ])
                    loss_adv = criterion_dom(dom_logits, dom_labels)

                elif args.method == "cdan":
                    grl_lambda = compute_grl_lambda(epoch - 1, step, args.epochs, steps_per_epoch) * adapt_scale
                    cond_s = conditional_map(feat_s, logits_s)
                    cond_t = conditional_map(feat_t, logits_t)
                    dom_in = torch.cat([cond_s, cond_t], dim=0)
                    dom_logits = model.domain_discriminator(gradient_reverse(dom_in, grl_lambda))
                    dom_labels = torch.cat([
                        torch.zeros(cond_s.size(0), dtype=torch.long, device=device),
                        torch.ones(cond_t.size(0), dtype=torch.long, device=device),
                    ])
                    loss_adv = criterion_dom(dom_logits, dom_labels)

                elif args.method == "deepcoral":
                    loss_transfer = coral_loss(feat_s, feat_t)

                elif args.method == "mmd":
                    loss_transfer = mmd_loss(feat_s, feat_t)

                elif args.method == "jan":
                    prob_s = torch.softmax(logits_s, dim=1)
                    prob_t = torch.softmax(logits_t, dim=1)
                    joint_s = torch.cat([feat_s, prob_s], dim=1)
                    joint_t = torch.cat([feat_t, prob_t], dim=1)
                    loss_transfer = mmd_loss(joint_s, joint_t)

                elif args.method == "entropy_min":
                    prob_t = torch.softmax(logits_t, dim=1)
                    loss_transfer = -torch.sum(prob_t * torch.log(prob_t + 1e-8), dim=1).mean()

                elif args.method == "mcc":
                    prob_t = torch.softmax(logits_t, dim=1)
                    loss_transfer = mcc_loss(prob_t)

                transfer_weight = args.lambda_transfer
                if args.method == "mmd":
                    transfer_weight = args.lambda_transfer_mmd
                elif args.method == "jan":
                    transfer_weight = args.lambda_transfer_jan
                elif args.method == "mcc":
                    transfer_weight = args.lambda_transfer_mcc

                if args.transfer_warmup_epochs > 0:
                    warmup_scale = min(1.0, epoch / float(args.transfer_warmup_epochs))
                else:
                    warmup_scale = 1.0

                if args.method in {"cdan", "mcc"}:
                    warmup_scale *= adapt_scale

                adv_weight = args.lambda_adv
                if args.method == "cdan":
                    adv_weight = args.lambda_adv_cdan

                loss_total = loss_cls + (transfer_weight * warmup_scale) * loss_transfer + adv_weight * loss_adv
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                cls_sum += loss_cls.item()
                transfer_sum += loss_transfer.item()
                adv_sum += loss_adv.item()
                total_sum += loss_total.item()

            scheduler.step()
            avg_cls = cls_sum / max(steps_per_epoch, 1)
            avg_transfer = transfer_sum / max(steps_per_epoch, 1)
            avg_adv = adv_sum / max(steps_per_epoch, 1)
            avg_total = total_sum / max(steps_per_epoch, 1)

            test_loss = -1.0
            test_acc = -1.0
            far = -1.0
            fdr = -1.0
            if epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs:
                test_loss, test_acc, far, fdr = evaluate(
                    model,
                    target_test_loader,
                    criterion_cls,
                    device,
                    normal_class_indices,
                )
                logger.info(
                    f"[Eval @ Epoch {epoch}/{args.epochs}] "
                    f"test_loss={test_loss:.6f}, test_acc={test_acc:.4f}, FAR={far:.4f}, FDR={fdr:.4f}"
                )

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            lr_val = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            eta = elapsed / epoch * (args.epochs - epoch)

            if epoch == 1 or epoch % args.print_interval == 0 or epoch == args.epochs:
                logger.info(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"train_cls={avg_cls:.6f}, train_transfer={avg_transfer:.6f}, train_adv={avg_adv:.6f}, "
                    f"train_total={avg_total:.6f}, lr={lr_val:.7f}, eta_sec={int(eta)}"
                )

            writer.writerow([
                epoch, avg_cls, avg_transfer, avg_adv, avg_total,
                test_loss, test_acc, far, fdr, lr_val
            ])
            f_csv.flush()

    last_ckpt = os.path.join(method_out_dir, "model_last.pth")
    best_ckpt = os.path.join(method_out_dir, "model_best.pth")
    torch.save(model.state_dict(), last_ckpt)
    if best_state is not None:
        torch.save(best_state, best_ckpt)

    config_path = os.path.join(method_out_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger.info(f"Training done for method={args.method}")
    logger.info(f"Artifacts: {method_out_dir}")


if __name__ == "__main__":
    run(parse_args())
