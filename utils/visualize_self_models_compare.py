import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Data.self_dataset import get_self_dataloaders
from Model.ISAN import Model


def build_model(checkpoint_path, num_classes, device, attention_mode='none', attn_heads=4, attn_dropout=0.1):
    model = Model(
        feature_dim=320,
        num_classes=num_classes,
        attention_mode=attention_mode,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_outputs(model, loader, device, max_tsne_samples=3000):
    preds_all, labels_all = [], []
    feats_all = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            feats = model.FRFE(signals)
            logits = model.main_classifier(feats)
            preds = logits.argmax(dim=1)

            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            feats_all.append(feats.flatten(1).cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    feats_all = np.concatenate(feats_all)

    if len(labels_all) > max_tsne_samples:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(labels_all), size=max_tsne_samples, replace=False)
        tsne_feats = feats_all[sample_idx]
        tsne_labels = labels_all[sample_idx]
    else:
        tsne_feats = feats_all
        tsne_labels = labels_all

    return preds_all, labels_all, tsne_feats, tsne_labels


def save_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def tsne_embed(feats):
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, verbose=0)
    return tsne.fit_transform(feats)


def balanced_sample(feats, labels, per_class=150, seed=42):
    rng = np.random.default_rng(seed)
    selected = []
    for cid in sorted(np.unique(labels)):
        idx = np.where(labels == cid)[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        selected.append(idx)
    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return feats[selected], labels[selected]


def save_tsne_plot(emb2d, labels, class_names, save_path, title):
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette('tab10', n_colors=len(class_names))
    for cid, cname in enumerate(class_names):
        mask = labels == cid
        if np.any(mask):
            plt.scatter(emb2d[mask, 0], emb2d[mask, 1], s=14, alpha=0.85, color=colors[cid], label=cname)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_compare_confusion(cm_a, cm_b, class_names, label_a, label_b, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    sns.heatmap(cm_a, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Confusion: {label_a}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_b, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'Confusion: {label_b}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].tick_params(axis='x', rotation=45)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def save_compare_tsne(emb_a, labels_a, emb_b, labels_b, class_names, label_a, label_b, save_path):
    colors = sns.color_palette('tab10', n_colors=len(class_names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for cid, cname in enumerate(class_names):
        mask_a = labels_a == cid
        if np.any(mask_a):
            axes[0].scatter(emb_a[mask_a, 0], emb_a[mask_a, 1], s=10, alpha=0.8, color=colors[cid], label=cname)
    axes[0].set_title(f't-SNE: {label_a}')

    for cid, cname in enumerate(class_names):
        mask_b = labels_b == cid
        if np.any(mask_b):
            axes[1].scatter(emb_b[mask_b, 0], emb_b[mask_b, 1], s=10, alpha=0.8, color=colors[cid], label=cname)
    axes[1].set_title(f't-SNE: {label_b}')

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize confusion matrix and t-SNE for self-model original vs re.')
    parser.add_argument('--data-dir', type=str, default='self_data/75_Sta_60_test_dso')
    parser.add_argument('--model-original', type=str, required=True)
    parser.add_argument('--model-re', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--signal-length', type=int, default=2048)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-original-dir', type=str, required=True)
    parser.add_argument('--out-re-dir', type=str, required=True)
    parser.add_argument('--out-compare-dir', type=str, required=True)
    parser.add_argument('--tsne-max-samples', type=int, default=3000)
    parser.add_argument('--balanced-per-class', type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.out_original_dir, exist_ok=True)
    os.makedirs(args.out_re_dir, exist_ok=True)
    os.makedirs(args.out_compare_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_loader, metadata = get_self_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        signal_length=args.signal_length,
        use_dso=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    class_names = metadata['class_names']
    num_classes = metadata['num_classes']

    model_original = build_model(args.model_original, num_classes, device)
    model_re = build_model(args.model_re, num_classes, device)

    pred_o, label_o, feats_o, tsne_label_o = collect_outputs(model_original, test_loader, device, args.tsne_max_samples)
    pred_r, label_r, feats_r, tsne_label_r = collect_outputs(model_re, test_loader, device, args.tsne_max_samples)

    cm_o = confusion_matrix(label_o, pred_o)
    cm_r = confusion_matrix(label_r, pred_r)

    emb_o = tsne_embed(feats_o)
    emb_r = tsne_embed(feats_r)

    feats_o_bal, labels_o_bal = balanced_sample(feats_o, tsne_label_o, per_class=args.balanced_per_class, seed=42)
    feats_r_bal, labels_r_bal = balanced_sample(feats_r, tsne_label_r, per_class=args.balanced_per_class, seed=42)
    emb_o_bal = tsne_embed(feats_o_bal)
    emb_r_bal = tsne_embed(feats_r_bal)

    save_confusion_matrix(
        cm_o,
        class_names,
        os.path.join(args.out_original_dir, 'confusion_matrix_trainself_original.png'),
        'Confusion Matrix: trainself_original',
    )
    save_tsne_plot(
        emb_o,
        tsne_label_o,
        class_names,
        os.path.join(args.out_original_dir, 'tsne_trainself_original.png'),
        't-SNE: trainself_original',
    )
    save_tsne_plot(
        emb_o_bal,
        labels_o_bal,
        class_names,
        os.path.join(args.out_original_dir, 'tsne_trainself_original_balanced.png'),
        f't-SNE (balanced): trainself_original (per_class={args.balanced_per_class})',
    )

    save_confusion_matrix(
        cm_r,
        class_names,
        os.path.join(args.out_re_dir, 'confusion_matrix_trainself_grl_tec_re.png'),
        'Confusion Matrix: trainself_grl_tec_re',
    )
    save_tsne_plot(
        emb_r,
        tsne_label_r,
        class_names,
        os.path.join(args.out_re_dir, 'tsne_trainself_grl_tec_re.png'),
        't-SNE: trainself_grl_tec_re',
    )
    save_tsne_plot(
        emb_r_bal,
        labels_r_bal,
        class_names,
        os.path.join(args.out_re_dir, 'tsne_trainself_grl_tec_re_balanced.png'),
        f't-SNE (balanced): trainself_grl_tec_re (per_class={args.balanced_per_class})',
    )

    save_compare_confusion(
        cm_o,
        cm_r,
        class_names,
        'trainself_original',
        'trainself_grl_tec_re',
        os.path.join(args.out_compare_dir, 'compare_confusion_original_vs_re.png'),
    )
    save_compare_tsne(
        emb_o,
        tsne_label_o,
        emb_r,
        tsne_label_r,
        class_names,
        'trainself_original',
        'trainself_grl_tec_re',
        os.path.join(args.out_compare_dir, 'compare_tsne_original_vs_re.png'),
    )
    save_compare_tsne(
        emb_o_bal,
        labels_o_bal,
        emb_r_bal,
        labels_r_bal,
        class_names,
        'trainself_original (balanced)',
        'trainself_grl_tec_re (balanced)',
        os.path.join(args.out_compare_dir, 'compare_tsne_original_vs_re_balanced.png'),
    )

    acc_o = float((pred_o == label_o).mean())
    acc_r = float((pred_r == label_r).mean())
    print(f'Original acc: {acc_o:.4f}')
    print(f'Re acc: {acc_r:.4f}')
    print(f'Saved outputs to {args.out_original_dir}, {args.out_re_dir}, {args.out_compare_dir}')


if __name__ == '__main__':
    main()
