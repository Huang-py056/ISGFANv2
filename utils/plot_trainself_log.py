import argparse
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\]\s+"
    r".*?train_loss=(?P<train_loss>[\d.]+),\s+"
    r"loss_label_inv=(?P<loss_label_inv>[\d.]+),\s+"
    r"loss_entropy=(?P<loss_entropy>[\d.]+),\s+"
    r"loss_total=(?P<loss_total>[\d.]+),\s+"
    r"grl=(?P<grl>[\d.]+),\s+lr=(?P<lr>[\d.]+)",
    re.IGNORECASE,
)

EVAL_PATTERN = re.compile(
    r"\[Eval\s+@\s+Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\]\s+"
    r"test_loss=(?P<test_loss>[\d.]+),\s+test_acc=(?P<test_acc>[\d.]+)",
    re.IGNORECASE,
)


def parse_log(log_path: str) -> Dict[str, List[float]]:
    train_epochs, train_loss, loss_label_inv = [], [], []
    loss_entropy, loss_total, grl_vals, lr_vals = [], [], [], []
    eval_epochs, test_loss, test_acc = [], [], []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                train_epochs.append(int(epoch_match.group("epoch")))
                train_loss.append(float(epoch_match.group("train_loss")))
                loss_label_inv.append(float(epoch_match.group("loss_label_inv")))
                loss_entropy.append(float(epoch_match.group("loss_entropy")))
                loss_total.append(float(epoch_match.group("loss_total")))
                grl_vals.append(float(epoch_match.group("grl")))
                lr_vals.append(float(epoch_match.group("lr")))
                continue

            eval_match = EVAL_PATTERN.search(line)
            if eval_match:
                eval_epochs.append(int(eval_match.group("epoch")))
                test_loss.append(float(eval_match.group("test_loss")))
                test_acc.append(float(eval_match.group("test_acc")))

    return {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "loss_label_inv": loss_label_inv,
        "loss_entropy": loss_entropy,
        "loss_total": loss_total,
        "grl": grl_vals,
        "lr": lr_vals,
        "eval_epochs": eval_epochs,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def _plot_series(x, y, title: str, ylabel: str, save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=1.8)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_eval(eval_epochs, test_acc, test_loss, save_path: str):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(eval_epochs, test_acc, color="tab:blue", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eval_epochs, test_loss, color="tab:red", linewidth=1.6, marker="s", markersize=4)
    ax2.set_ylabel("Test Loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Evaluation Curves (Accuracy / Loss)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot self-training/eval curves from log.")
    parser.add_argument("--log", type=str, required=True, help="Path to self-training log")
    parser.add_argument("--out-dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--tag", type=str, default="", help="Output filename suffix; default uses log filename")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    parsed = parse_log(args.log)

    if not parsed["train_epochs"]:
        raise RuntimeError(f"No training entries parsed from log: {args.log}")

    if args.tag:
        tag = args.tag
    else:
        tag = os.path.splitext(os.path.basename(args.log))[0]

    _plot_series(
        parsed["train_epochs"],
        parsed["train_loss"],
        "Training Curve: train_loss",
        "train_loss",
        os.path.join(args.out_dir, f"curve_train_loss_{tag}.png"),
    )
    _plot_series(
        parsed["train_epochs"],
        parsed["loss_total"],
        "Training Curve: loss_total",
        "loss_total",
        os.path.join(args.out_dir, f"curve_loss_total_{tag}.png"),
    )
    _plot_series(
        parsed["train_epochs"],
        parsed["loss_label_inv"],
        "Training Curve: loss_label_inv",
        "loss_label_inv",
        os.path.join(args.out_dir, f"curve_loss_label_inv_{tag}.png"),
    )
    _plot_series(
        parsed["train_epochs"],
        parsed["loss_entropy"],
        "Training Curve: loss_entropy",
        "loss_entropy",
        os.path.join(args.out_dir, f"curve_loss_entropy_{tag}.png"),
    )
    _plot_series(
        parsed["train_epochs"],
        parsed["grl"],
        "GRL Curve",
        "grl_lambda",
        os.path.join(args.out_dir, f"curve_grl_{tag}.png"),
    )
    _plot_series(
        parsed["train_epochs"],
        parsed["lr"],
        "Learning Rate Curve",
        "learning_rate",
        os.path.join(args.out_dir, f"curve_lr_{tag}.png"),
    )

    if parsed["eval_epochs"]:
        _plot_eval(
            parsed["eval_epochs"],
            parsed["test_acc"],
            parsed["test_loss"],
            os.path.join(args.out_dir, f"curve_eval_acc_loss_{tag}.png"),
        )

    print(f"Saved figures to: {args.out_dir}")
    print(f"Tag: {tag}, Train points: {len(parsed['train_epochs'])}, Eval points: {len(parsed['eval_epochs'])}")


if __name__ == "__main__":
    main()
