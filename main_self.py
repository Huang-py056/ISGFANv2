import copy
import time
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from Model.ISAN import Model
from Data.self_dataset import get_self_dataloaders
from utils.training_logger import TrainingLogger
from losses.grl import GradientReverseLayer

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Private CSV dataset training script')

    parser.add_argument('--data-dir', type=str, default='self_data/75_Sta_60_test_dso', help='Directory containing private CSV files')
    parser.add_argument('--use-dso', action='store_true', help='Use *_dso.csv files instead of *_test.csv files')
    parser.add_argument('--signal-length', type=int, default=2048, help='Resampled signal length')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')

    parser.add_argument('--feature-dim', type=int, default=320, help='Feature dimension output by encoder')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')

    parser.add_argument('--attention-mode', type=str, default='none', choices=['none', 'attn1', 'attn2'],
                        help='Optional lightweight attention mode for feature blocks')
    parser.add_argument('--attn-heads', type=int, default=4, help='Attention heads used by attn2 mode')
    parser.add_argument('--attn-dropout', type=float, default=0.1, help='Attention dropout used by attn2 mode')

    parser.add_argument('--disable-grl-schedule', action='store_true', help='Disable GRL schedule and use fixed coefficient')
    parser.add_argument('--grl-gamma', type=float, default=10.0, help='Gamma value for GRL schedule')
    parser.add_argument('--grl-max', type=float, default=1.0, help='Maximum GRL coefficient')
    parser.add_argument('--lambda-li', type=float, default=0.01, help='Label-invariant adversarial loss weight')

    parser.add_argument('--disable-entropy-curriculum', action='store_true', help='Disable target entropy curriculum regularization')
    parser.add_argument('--lambda-entropy', type=float, default=0.02, help='Loss weight for target entropy curriculum')
    parser.add_argument('--entropy-warmup-epochs', type=int, default=30, help='Warmup epochs for entropy curriculum')
    parser.add_argument('--pseudo-conf-threshold', type=float, default=0.7, help='Base confidence threshold for entropy curriculum')
    parser.add_argument('--use-eval-ema', action='store_true', help='Use EMA model for evaluation to reduce metric spikes')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay for eval model when --use-eval-ema is enabled')

    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--print-interval', type=int, default=10, help='Print every N epochs')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Computation device')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id when using cuda')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')

    parser.add_argument('--output-dir', type=str, default='results_self', help='Output directory')
    parser.add_argument('--save-prefix', type=str, default='model_self', help='Model checkpoint prefix')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')

    return parser.parse_args()


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def evaluate(model, loader, criterion, device):
    model_was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            feats = model.FRFE(signals)
            logits = model.main_classifier(feats)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += bs
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    if model_was_training:
        model.train()

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)

    preds = torch.cat(all_preds) if all_preds else torch.tensor([], dtype=torch.long)
    labels = torch.cat(all_labels) if all_labels else torch.tensor([], dtype=torch.long)
    num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 0

    per_class_metrics = []
    for class_idx in range(num_classes):
        pred_pos = preds == class_idx
        true_pos_mask = labels == class_idx
        tp = torch.logical_and(pred_pos, true_pos_mask).sum().item()
        fp = torch.logical_and(pred_pos, torch.logical_not(true_pos_mask)).sum().item()
        fn = torch.logical_and(torch.logical_not(pred_pos), true_pos_mask).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_class_metrics.append({
            'class_idx': class_idx,
            'precision': precision,
            'recall': recall,
            'support': int(true_pos_mask.sum().item()),
        })

    return avg_loss, acc, per_class_metrics


def compute_grl_lambda(global_step, total_steps, gamma=10.0, max_coeff=1.0):
    progress = min(max(global_step / max(total_steps, 1), 0.0), 1.0)
    coeff = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
    return max_coeff * coeff


def compute_entropy_curriculum_loss(probs, confidence_mask, epoch, warmup_epochs):
    if probs.numel() == 0 or (not confidence_mask.any()):
        return torch.tensor(0.0, device=probs.device)

    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    confident_entropy = entropy[confidence_mask]
    curriculum = min(1.0, float(epoch + 1) / max(warmup_epochs, 1))
    return curriculum * confident_entropy.mean()


def main():
    args = parse_args()

    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device(args.device)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader, metadata = get_self_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        signal_length=args.signal_length,
        use_dso=args.use_dso,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    num_classes = metadata['num_classes']

    logger.info(f"Private dataset loaded from: {args.data_dir}")
    logger.info(f"use_dso={args.use_dso}, signal_length={metadata['signal_length']}")
    logger.info(f"num_classes={num_classes}, class_names={metadata['class_names']}")
    logger.info(f"train_samples={metadata['num_train']}, test_samples={metadata['num_test']}")

    model = Model(
        args.feature_dim,
        num_classes,
        attention_mode=args.attention_mode,
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
    ).to(device)

    ema_model = None
    if args.use_eval_ema:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.CrossEntropyLoss()

    train_logger = TrainingLogger(os.path.join(args.output_dir, 'training_log_self.csv'))

    best_test_acc = 0.0
    best_state_dict = None
    train_start_time = time.time()
    use_entropy_curriculum = not args.disable_entropy_curriculum

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()

        loss_main_sum = 0.0
        loss_li_sum = 0.0
        loss_entropy_sum = 0.0
        loss_total_sum = 0.0
        batch_count = 0
        min_steps = len(train_loader)

        for step_idx, (signals, labels) in enumerate(train_loader):
            global_step = epoch * min_steps + step_idx
            total_steps = args.epochs * min_steps
            if args.disable_grl_schedule:
                grl_lambda = args.grl_max
            else:
                grl_lambda = compute_grl_lambda(global_step, total_steps, args.grl_gamma, args.grl_max)

            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            feats = model.FRFE(signals)
            logits = model.main_classifier(feats)
            loss_main = criterion(logits, labels)

            reversed_feats = GradientReverseLayer.apply(feats, grl_lambda)
            out_label_inv = model.label_invariant_discriminator(reversed_feats)
            loss_li = criterion(out_label_inv, labels)

            probs = F.softmax(logits, dim=1)
            max_probs, _ = probs.max(dim=1)
            confidence_mask = max_probs > args.pseudo_conf_threshold
            if use_entropy_curriculum:
                loss_entropy = compute_entropy_curriculum_loss(
                    probs,
                    confidence_mask,
                    epoch,
                    args.entropy_warmup_epochs,
                )
            else:
                loss_entropy = torch.tensor(0.0, device=device)

            loss_total = (
                loss_main
                + args.lambda_li * loss_li
                + args.lambda_entropy * loss_entropy
            )
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if ema_model is not None:
                with torch.no_grad():
                    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.mul_(args.ema_decay).add_(model_param.data, alpha=1.0 - args.ema_decay)
                    for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
                        ema_buffer.copy_(model_buffer)

            loss_main_sum += loss_main.item()
            loss_li_sum += loss_li.item()
            loss_entropy_sum += loss_entropy.item()
            loss_total_sum += loss_total.item()
            batch_count += 1

        scheduler.step()

        avg_train_loss = loss_main_sum / max(batch_count, 1)
        avg_li_loss = loss_li_sum / max(batch_count, 1)
        avg_entropy_loss = loss_entropy_sum / max(batch_count, 1)
        avg_total_loss = loss_total_sum / max(batch_count, 1)
        current_lr = optimizer.param_groups[0]['lr']

        test_loss = None
        test_acc = None
        if ((epoch + 1) == 1) or ((epoch + 1) % 5 == 0) or ((epoch + 1) == args.epochs):
            eval_model = ema_model if ema_model is not None else model
            test_loss, test_acc, per_class_metrics = evaluate(eval_model, test_loader, criterion, device)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state_dict = copy.deepcopy(eval_model.state_dict())
            logger.info(
                f"[Eval @ Epoch {epoch + 1}/{args.epochs}] "
                f"test_loss={test_loss:.6f}, test_acc={test_acc:.4f}"
            )
            class_names = metadata.get('class_names', [])
            metric_chunks = []
            for m in per_class_metrics:
                class_name = class_names[m['class_idx']] if m['class_idx'] < len(class_names) else str(m['class_idx'])
                metric_chunks.append(
                    f"{class_name}(prec={m['precision']:.4f}, rec={m['recall']:.4f}, n={m['support']})"
                )
            if metric_chunks:
                logger.info("[Eval Per-Class] " + "; ".join(metric_chunks))

        epoch_time = time.time() - epoch_start
        avg_epoch_time = (time.time() - train_start_time) / (epoch + 1)
        eta_seconds = avg_epoch_time * (args.epochs - epoch - 1)

        train_logger.log_losses({
            'loss_main': avg_train_loss,
            'loss_label_inv': avg_li_loss,
            'loss_entropy': avg_entropy_loss,
            'loss_total': avg_total_loss,
            'test_loss': test_loss if test_loss is not None else -1.0,
            'test_acc': test_acc if test_acc is not None else -1.0,
        })

        should_print = (epoch == 0) or ((epoch + 1) % args.print_interval == 0) or ((epoch + 1) == args.epochs)
        if should_print:
            msg = (
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"train_loss={avg_train_loss:.6f}, "
                f"loss_label_inv={avg_li_loss:.6f}, "
                f"loss_entropy={avg_entropy_loss:.6f}, "
                f"loss_total={avg_total_loss:.6f}, "
                f"grl={grl_lambda:.4f}, "
                f"lr={current_lr:.8f}, "
                f"epoch_time={format_duration(epoch_time)}, "
                f"eta={format_duration(eta_seconds)}"
            )
            if test_loss is not None and test_acc is not None:
                msg += f", test_loss={test_loss:.6f}, test_acc={test_acc:.4f}"
            logger.info(msg)

    best_model_path = os.path.join(args.output_dir, f'{args.save_prefix}_best.pth')
    last_model_path = os.path.join(args.output_dir, f'{args.save_prefix}_last.pth')

    if best_state_dict is not None:
        torch.save(best_state_dict, best_model_path)
    torch.save(model.state_dict(), last_model_path)

    train_logger.save_to_file()
    logger.info(f'Training complete. Best test acc: {best_test_acc:.4f}')
    logger.info(f'Best model: {best_model_path}')
    logger.info(f'Last model: {last_model_path}')


if __name__ == '__main__':
    main()
