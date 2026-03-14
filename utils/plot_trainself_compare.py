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


def plot_overlay(x_a, y_a, x_b, y_b, label_a, label_b, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x_a, y_a, linewidth=1.8, label=label_a)
    plt.plot(x_b, y_b, linewidth=1.8, label=label_b)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_eval_overlay(parsed_a, parsed_b, label_a, label_b, save_path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(parsed_a["eval_epochs"], parsed_a["test_acc"], color="tab:blue", linewidth=2, label=f"{label_a} acc")
    ax1.plot(parsed_b["eval_epochs"], parsed_b["test_acc"], color="tab:cyan", linewidth=2, label=f"{label_b} acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(parsed_a["eval_epochs"], parsed_a["test_loss"], color="tab:red", linewidth=1.4, linestyle="--", label=f"{label_a} loss")
    ax2.plot(parsed_b["eval_epochs"], parsed_b["test_loss"], color="tab:orange", linewidth=1.4, linestyle="--", label=f"{label_b} loss")
    ax2.set_ylabel("Test Loss")

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

    plt.title("Eval Comparison (Acc/Loss)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare two self-training logs with overlaid curves.")
    parser.add_argument("--log-a", type=str, required=True, help="Path of log A")
    parser.add_argument("--log-b", type=str, required=True, help="Path of log B")
    parser.add_argument("--label-a", type=str, default="A", help="Legend label for log A")
    parser.add_argument("--label-b", type=str, default="B", help="Legend label for log B")
    parser.add_argument("--out-dir", type=str, default="visualizations/self_compare", help="Output directory")
    parser.add_argument("--prefix", type=str, default="trainself_compare", help="Output file prefix")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    parsed_a = parse_log(args.log_a)
    parsed_b = parse_log(args.log_b)

    if not parsed_a["train_epochs"] or not parsed_b["train_epochs"]:
        raise RuntimeError("Failed to parse one or both logs.")

    plot_overlay(
        parsed_a["train_epochs"], parsed_a["train_loss"],
        parsed_b["train_epochs"], parsed_b["train_loss"],
        args.label_a, args.label_b,
        "Train Loss Comparison", "train_loss",
        os.path.join(args.out_dir, f"{args.prefix}_train_loss.png"),
    )
    plot_overlay(
        parsed_a["train_epochs"], parsed_a["loss_total"],
        parsed_b["train_epochs"], parsed_b["loss_total"],
        args.label_a, args.label_b,
        "Loss Total Comparison", "loss_total",
        os.path.join(args.out_dir, f"{args.prefix}_loss_total.png"),
    )
    plot_overlay(
        parsed_a["train_epochs"], parsed_a["loss_label_inv"],
        parsed_b["train_epochs"], parsed_b["loss_label_inv"],
        args.label_a, args.label_b,
        "Loss Label Invariant Comparison", "loss_label_inv",
        os.path.join(args.out_dir, f"{args.prefix}_loss_label_inv.png"),
    )
    plot_overlay(
        parsed_a["train_epochs"], parsed_a["loss_entropy"],
        parsed_b["train_epochs"], parsed_b["loss_entropy"],
        args.label_a, args.label_b,
        "Loss Entropy Comparison", "loss_entropy",
        os.path.join(args.out_dir, f"{args.prefix}_loss_entropy.png"),
    )
    plot_overlay(
        parsed_a["train_epochs"], parsed_a["grl"],
        parsed_b["train_epochs"], parsed_b["grl"],
        args.label_a, args.label_b,
        "GRL Comparison", "grl_lambda",
        os.path.join(args.out_dir, f"{args.prefix}_grl.png"),
    )
    plot_overlay(
        parsed_a["train_epochs"], parsed_a["lr"],
        parsed_b["train_epochs"], parsed_b["lr"],
        args.label_a, args.label_b,
        "Learning Rate Comparison", "learning_rate",
        os.path.join(args.out_dir, f"{args.prefix}_lr.png"),
    )
    plot_eval_overlay(
        parsed_a,
        parsed_b,
        args.label_a,
        args.label_b,
        os.path.join(args.out_dir, f"{args.prefix}_eval_acc_loss.png"),
    )

    best_a = max(parsed_a["test_acc"]) if parsed_a["test_acc"] else float("nan")
    best_b = max(parsed_b["test_acc"]) if parsed_b["test_acc"] else float("nan")
    print(f"Saved comparison figures to: {args.out_dir}")
    print(f"Best acc: {args.label_a}={best_a:.4f}, {args.label_b}={best_b:.4f}")


if __name__ == "__main__":
    main()
