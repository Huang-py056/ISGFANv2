import re
import matplotlib.pyplot as plt

def parse_metric(log_path, metric_names):
    values = []
    epochs = []
    metric_pattern = '|'.join(re.escape(name) for name in metric_names)
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(rf'\[Epoch (\d+)/(\d+)\].*?(?:{metric_pattern})=([\d\.]+)', line)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(3))
                epochs.append(epoch)
                values.append(loss)
    return epochs, values

def plot_loss_group(log1, log2, name1, name2, out_path, title, metrics1, metrics2):
    epochs1, loss1 = parse_metric(log1, metrics1)
    epochs2, loss2 = parse_metric(log2, metrics2)
    plt.figure(figsize=(8,6))
    plt.plot(epochs1, loss1, label=name1, linewidth=2)
    plt.plot(epochs2, loss2, label=name2, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == '__main__':
    # 第一组：public
    plot_loss_group(
        '/data2/ranran/chenghaoyue/ISGFAN/train.log',
        '/data2/ranran/chenghaoyue/ISGFAN/train_grl_tec.log',
        'ISGFAN',
        'ISGFANv2 (ours)',
        '/data2/ranran/chenghaoyue/ISGFAN/train_loss_compare_group1.png',
        'Train Loss Comparison (public)',
        ['train_loss', 'loss_total'],
        ['train_loss', 'loss_total']
    )
    # 第二组：private
    plot_loss_group(
        '/data2/ranran/chenghaoyue/ISGFAN/trainself.log',
        '/data2/ranran/chenghaoyue/ISGFAN/trainself_grl_tec.log',
        'ISGFAN',
        'ISGFANv2 (ours)',
        '/data2/ranran/chenghaoyue/ISGFAN/train_loss_compare_group2.png',
        'Train Loss Comparison (private)',
        ['train_loss', 'loss_total'],
        ['train_loss', 'loss_total']
    )
    print('Train loss comparison figures saved.')
