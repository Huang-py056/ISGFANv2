import copy
import time
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.ISAN import Model
from Data.dataset import get_dataloaders, data_manager
from torch.optim import AdamW
from utils.training_logger import TrainingLogger
from losses.grl import GradientReverseLayer
from losses.FD_Loss_SAM import SAM, compute_focal_domain_loss
from losses.ortho_loss import orthogonality_loss
import argparse
import os

# cd /data2/ranran/chenghaoyue/ISGFAN && /data2/ranran/chenghaoyue/anaconda3/bin/python main.py
# cd /data2/ranran/chenghaoyue/ISGFAN && nohup /data2/ranran/chenghaoyue/anaconda3/bin/python main.py >> train.log 2>&1 &

# ---------------------------
# Parameter Settings (using argparse)
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Domain Adaptation Model Training Parameters")

    # Basic training parameters
    parser.add_argument('--feature-dim', type=int, default=320, help='Feature dimension output by feature extractor')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of label categories')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--epochs', type=int, default=3500, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for dataloaders')
    parser.add_argument('--eval-interval', type=int, default=100, help='Run evaluation every N epochs')
    parser.add_argument('--print-interval', type=int, default=10, help='Print training info every N epochs')
    # GRL调度
    parser.add_argument('--disable-grl-schedule', action='store_true', help='Disable GRL schedule and use fixed coefficient')
    parser.add_argument('--grl-gamma', type=float, default=10.0, help='Gamma value for GRL schedule')
    parser.add_argument('--grl-max', type=float, default=1.0, help='Maximum GRL coefficient')

    # 新模块：Target Entropy Curriculum (TEC)
    parser.add_argument('--disable-entropy-curriculum', action='store_true', help='Disable target entropy curriculum regularization')
    parser.add_argument('--lambda-entropy', type=float, default=0.02, help='Loss weight for target entropy curriculum')
    parser.add_argument('--entropy-warmup-epochs', type=int, default=30, help='Warmup epochs for entropy curriculum')
    parser.add_argument('--pseudo-conf-threshold', type=float, default=0.7, help='Base confidence threshold for target pseudo labels')

    # Loss weighting parameters
    parser.add_argument('--lambda-ortho-s', type=float, default=0.01, help='Initial source orthogonality loss weight')
    parser.add_argument('--lambda-dom', type=float, default=0.5, help='Domain discriminator loss weight')
    parser.add_argument('--lambda-li', type=float, default=0.01, help='Label-independent discriminator loss weight')
    parser.add_argument('--lambda-recon', type=float, default=0.01, help='Initial reconstruction loss weight')
    parser.add_argument('--lambda-dom-local', type=float, default=0.1, help='Local domain loss weight')

    # Experiment settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Computation device')
    parser.add_argument('--gpu-id', type=int, default=2, help='GPU id to use when device is cuda, e.g. 0')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save-prefix', type=str, default='model', help='Model save prefix')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level for console/file output')

    return parser.parse_args()

# Initialize parameters
args = parse_args()

# Set device using parameters
if args.device.startswith('cuda') and args.gpu_id is not None and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu_id}')
else:
    device = torch.device(args.device)

logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dynamic loss weighting function
def dynamic_weight_lambda(loss_val, ref_loss_val, base_lambda, max_ratio=1e1):
    if loss_val > max_ratio * ref_loss_val:
        return base_lambda * (max_ratio * ref_loss_val / (loss_val + 1e-18))
    return base_lambda

# Initialize subdomain attention
subdomain_attention = SAM(args.num_classes, device)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def evaluate_target_domain(model, test_loader, criterion_cls):
    model_was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for seg, label in test_loader:
            seg = seg.to(device)
            label = label.to(device)
            logits = model.main_classifier(model.FRFE(seg))
            loss = criterion_cls(logits, label)

            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == label).sum().item()
            total_samples += batch_size

    if model_was_training:
        model.train()

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


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
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    model = Model(args.feature_dim, args.num_classes)
    model.to(device)

    train_loader, target_loader = get_dataloaders(batch_size=args.batch_size)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4
    )
    criterion_cls = nn.CrossEntropyLoss()

    # SWA configuration
    swa_start = int(0.95 * args.epochs)  # SWA start epoch
    swa_n = 10
    swa_state_dict = None
    best_train_loss = float('inf')
    best_state_dict = None

    # Initialize training logger (using output directory)
    train_logger = TrainingLogger(os.path.join(args.output_dir, "training_log.csv"))

    # Learning rate scheduler (using parameters)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # Set modules to trainable
    model.FRFE.requires_grad_(True)
    model.domain_discriminator.requires_grad_(True)
    model.label_invariant_discriminator.requires_grad_(True)
    model.FIFE.requires_grad_(True)
    model.main_classifier.requires_grad_(True)
    model.decoder.requires_grad_(True)
    for ld in model.local_discriminators:
        ld.requires_grad_(True)

    test_loader = data_manager.test_loader
    train_start_time = time.time()
    use_entropy_curriculum = not args.disable_entropy_curriculum

    # Training loop
    for epoch in range(args.epochs):
        model.train()

        epoch_weight_sum = torch.zeros(args.num_classes)
        epoch_loss_sums = {
            'loss_main': 0.0,
            'loss_diff_source': 0.0,
            'loss_label_inv': 0.0,
            'loss_global_dom': 0.0,
            'loss_recon': 0.0,
            'loss_local_dom': 0.0,
            'loss_entropy': 0.0,
            'loss_total': 0.0
        }
        epoch_start_time = time.time()
        min_steps = min(len(train_loader), len(target_loader))
        src_iter = iter(train_loader)
        tgt_iter = iter(target_loader)

        for step_idx in range(min_steps):
            global_step = epoch * min_steps + step_idx
            total_steps = args.epochs * min_steps
            if args.disable_grl_schedule:
                grl_lambda = args.grl_max
            else:
                grl_lambda = compute_grl_lambda(global_step, total_steps, args.grl_gamma, args.grl_max)

            src_imgs, src_labels = next(src_iter)
            tgt_imgs = next(tgt_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)
            optimizer.zero_grad()

            # Feature extraction
            shared_feats_src = model.FRFE(src_imgs)
            shared_feats_tgt = model.FRFE(tgt_imgs)
            source_private_feats = model.FIFE(src_imgs)

            out_main_src = model.main_classifier(shared_feats_src)
            loss_main = criterion_cls(out_main_src, src_labels)

            with torch.no_grad():
                logits_tgt = model.main_classifier(shared_feats_tgt)
                probs = F.softmax(logits_tgt, dim=1)
                max_probs, tgt_pseudo_labels = probs.max(dim=1)

            # Label-independent adversarial on source domain
            reversed_feats_src = GradientReverseLayer.apply(source_private_feats, grl_lambda)
            out_label_inv_src = model.label_invariant_discriminator(reversed_feats_src)
            loss_label_inv = criterion_cls(out_label_inv_src, src_labels)

            # Orthogonality loss
            shared_feats_src_1d = shared_feats_src.mean(dim=-1)
            source_private_feats_1d = source_private_feats.mean(dim=-1)
            loss_ortho_src = orthogonality_loss(shared_feats_src_1d, source_private_feats_1d)

            # Global domain discriminator
            dom_labels_src = torch.zeros(src_imgs.size(0), dtype=torch.long, device=device)
            dom_labels_tgt = torch.ones(tgt_imgs.size(0), dtype=torch.long, device=device)
            feats_dom = torch.cat([shared_feats_src, shared_feats_tgt], dim=0)
            labels_dom = torch.cat([dom_labels_src, dom_labels_tgt], dim=0)
            reversed_feats_dom = GradientReverseLayer.apply(feats_dom, grl_lambda)
            global_dom_pred = model.domain_discriminator(reversed_feats_dom)
            loss_global_dom = criterion_cls(global_dom_pred, labels_dom)

            # Local domain discriminators
            with torch.no_grad():
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

                confidence_threshold = max(args.pseudo_conf_threshold, 0.9 - epoch * 0.002)
                entropy_threshold = 0.9
                confident_mask = (max_probs > confidence_threshold) & (entropy < entropy_threshold)

            shared_feats_tgt_filtered = shared_feats_tgt[confident_mask]
            tgt_pseudo_labels = tgt_pseudo_labels[confident_mask]

            if shared_feats_tgt_filtered.numel() == 0:
                weighted_local_loss = torch.tensor(0.0, device=device)
                weight_vec = torch.zeros(args.num_classes)
            else:
                weighted_local_loss, weight_vec, _ = compute_focal_domain_loss(
                    shared_feats_src, shared_feats_tgt_filtered,
                    src_labels, tgt_pseudo_labels, model, subdomain_attention, grl_lambda=grl_lambda)

            epoch_weight_sum += weight_vec.cpu()

            if use_entropy_curriculum:
                loss_entropy = compute_entropy_curriculum_loss(
                    probs,
                    confident_mask,
                    epoch,
                    args.entropy_warmup_epochs,
                )
            else:
                loss_entropy = torch.tensor(0.0, device=device)

            # Reconstruction
            combined_feats_src = torch.cat([shared_feats_src, source_private_feats], dim=1)
            reconstructed_src = model.decoder(combined_feats_src)
            loss_recon_src = F.mse_loss(reconstructed_src, src_imgs)
            loss_recon = loss_recon_src

            # Dynamic weight calculation
            lambda_label_inv_dynamic = dynamic_weight_lambda(loss_label_inv.item(), loss_main.item(), args.lambda_li)
            lambda_ortho_src_dynamic = dynamic_weight_lambda(loss_ortho_src.item(), loss_main.item(), args.lambda_ortho_s)
            lambda_dom_dynamic = dynamic_weight_lambda(loss_global_dom.item(), loss_main.item(), args.lambda_dom)
            lambda_recon_dynamic = dynamic_weight_lambda(loss_recon.item(), loss_main.item(), args.lambda_recon)
            lambda_local_loss_dynamic = dynamic_weight_lambda(weighted_local_loss.item(), loss_main.item(), args.lambda_dom_local)

            # Total loss (with dynamic weights)
            loss = (loss_main
                    + lambda_ortho_src_dynamic * loss_ortho_src
                    + lambda_dom_dynamic * loss_global_dom
                    + lambda_label_inv_dynamic * loss_label_inv
                    + lambda_recon_dynamic * loss_recon
                    + lambda_local_loss_dynamic * weighted_local_loss
                    + args.lambda_entropy * loss_entropy)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Log losses
            loss_dict = {
                'loss_main': loss_main.item(),
                'loss_diff_source': loss_ortho_src.item(),
                'loss_label_inv': loss_label_inv.item(),
                'loss_global_dom': loss_global_dom.item(),
                'loss_recon': loss_recon.item(),
                'loss_local_dom': weighted_local_loss.item(),
                'loss_entropy': loss_entropy.item(),
                'loss_total': loss.item()
            }
            train_logger.log_losses(loss_dict)
            for key in epoch_loss_sums:
                epoch_loss_sums[key] += loss_dict[key]

            # Memory cleanup
            del src_imgs, src_labels, tgt_imgs, shared_feats_src, source_private_feats
            del shared_feats_tgt, feats_dom, global_dom_pred
            del tgt_pseudo_labels, shared_feats_tgt_filtered
            torch.cuda.empty_cache()

        epoch_losses = {k: v / max(min_steps, 1) for k, v in epoch_loss_sums.items()}
        epoch_time = time.time() - epoch_start_time
        avg_epoch_time = (time.time() - train_start_time) / (epoch + 1)
        eta_seconds = avg_epoch_time * (args.epochs - epoch - 1)
        current_lr = optimizer.param_groups[0]['lr']

        should_print = (epoch == 0) or ((epoch + 1) % args.print_interval == 0) or ((epoch + 1) == args.epochs)
        if should_print:
            logger.info(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"loss_total={epoch_losses['loss_total']:.6f}, "
                f"loss_main={epoch_losses['loss_main']:.6f}, "
                f"loss_diff_source={epoch_losses['loss_diff_source']:.6f}, "
                f"loss_label_inv={epoch_losses['loss_label_inv']:.6f}, "
                f"loss_global_dom={epoch_losses['loss_global_dom']:.6f}, "
                f"loss_recon={epoch_losses['loss_recon']:.6f}, "
                f"loss_local_dom={epoch_losses['loss_local_dom']:.6f}, "
                f"loss_entropy={epoch_losses['loss_entropy']:.6f}, "
                f"grl={grl_lambda:.4f}, "
                f"lr={current_lr:.8f}, "
                f"epoch_time={format_duration(epoch_time)}, "
                f"eta={format_duration(eta_seconds)}"
            )

        if test_loader is not None and (((epoch + 1) == 1) or ((epoch + 1) % args.eval_interval == 0)):
            eval_loss, eval_acc = evaluate_target_domain(model, test_loader, criterion_cls)
            logger.info(
                f"[Eval @ Epoch {epoch + 1}/{args.epochs}] "
                f"test_loss={eval_loss:.6f}, test_acc={eval_acc:.4f}"
            )

        # Update learning rate
        lr_scheduler.step()

        # SWA weight averaging
        if epoch + 3490 >= swa_start:
            if swa_state_dict is None:
                swa_state_dict = copy.deepcopy(model.state_dict())
            else:
                for k in swa_state_dict.keys():
                    swa_state_dict[k] = (swa_state_dict[k].cpu() * swa_n + model.state_dict()[k].cpu()) / (swa_n + 1)
            swa_n += 10

        # Save best weights
        if epoch_losses['loss_main'] < best_train_loss:
            best_train_loss = epoch_losses['loss_main']
            best_state_dict = copy.deepcopy(model.state_dict())

    # Save final model (using output directory and save prefix)
    best_model_path = os.path.join(args.output_dir, f"{args.save_prefix}_best.pth")
    torch.save(best_state_dict, best_model_path)

    if swa_state_dict is not None:
        swa_model_path = os.path.join(args.output_dir, f"{args.save_prefix}_swa.pth")
        torch.save(swa_state_dict, swa_model_path)

    train_logger.save_to_file()
    logger.info(f"Training complete! Best training loss: {best_train_loss:.4f}")
    logger.info(f"Model saved to: {best_model_path}")


if __name__ == "__main__":
    main()