import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SelfSignalDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def _resample_signal(signal, target_length):
    if len(signal) == target_length:
        return signal.astype(np.float32)
    old_x = np.linspace(0.0, 1.0, num=len(signal), endpoint=True)
    new_x = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(new_x, old_x, signal).astype(np.float32)


def _normalize_signal(signal):
    mean = signal.mean()
    std = signal.std() + 1e-8
    return (signal - mean) / std


def _extract_fault_label(file_name):
    stem = os.path.splitext(file_name)[0]
    parts = stem.split('_')
    fault_tag = 'NO_L'
    for token in parts:
        if token.startswith('L') and token[1:].isdigit():
            fault_tag = token
            break
    return fault_tag


def _list_csv_files(data_dir, use_dso):
    files = []
    for name in os.listdir(data_dir):
        if not name.endswith('.csv'):
            continue
        if use_dso and name.endswith('_dso.csv'):
            files.append(name)
        if (not use_dso) and name.endswith('_test.csv'):
            files.append(name)
    return sorted(files)


def _load_rows_from_csv(file_path):
    array = np.genfromtxt(
        file_path,
        delimiter=',',
        dtype=np.float32,
        encoding='utf-8-sig',
        invalid_raise=False,
    )
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    return array


def _stratified_split(indices_by_class, test_ratio, seed):
    rng = random.Random(seed)
    train_indices, test_indices = [], []

    for _, idx_list in indices_by_class.items():
        idx_list = idx_list[:]
        rng.shuffle(idx_list)

        if len(idx_list) == 1:
            train_indices.extend(idx_list)
            continue

        n_test = max(1, int(round(len(idx_list) * test_ratio)))
        n_test = min(n_test, len(idx_list) - 1)

        test_indices.extend(idx_list[:n_test])
        train_indices.extend(idx_list[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def get_self_dataloaders(
    data_dir,
    batch_size=64,
    test_ratio=0.2,
    signal_length=2048,
    use_dso=False,
    seed=42,
    num_workers=4,
):
    file_names = _list_csv_files(data_dir, use_dso=use_dso)
    if not file_names:
        raise ValueError(f'No csv files found in {data_dir} (use_dso={use_dso})')

    all_signals = []
    all_label_names = []

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        rows = _load_rows_from_csv(file_path)
        fault_name = _extract_fault_label(file_name)

        for row in rows:
            signal = _resample_signal(row, signal_length)
            signal = _normalize_signal(signal)
            signal_tensor = torch.from_numpy(signal).unsqueeze(0)
            all_signals.append(signal_tensor)
            all_label_names.append(fault_name)

    class_names = sorted(set(all_label_names))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    all_labels = [class_to_idx[name] for name in all_label_names]

    indices_by_class = defaultdict(list)
    for idx, label in enumerate(all_labels):
        indices_by_class[label].append(idx)

    train_indices, test_indices = _stratified_split(indices_by_class, test_ratio=test_ratio, seed=seed)

    train_samples = [all_signals[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    test_samples = [all_signals[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]

    train_dataset = SelfSignalDataset(train_samples, train_labels)
    test_dataset = SelfSignalDataset(test_samples, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    metadata = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'num_train': len(train_dataset),
        'num_test': len(test_dataset),
        'use_dso': use_dso,
        'signal_length': signal_length,
    }
    return train_loader, test_loader, metadata
