"""
visualization.py — Утилиты визуализации для SNN

Функции:
    - plot_spike_raster: растровый график спайков
    - plot_dataset_samples: примеры сигналов из датасета
    - plot_class_distribution: распределение классов
    - plot_spike_rate_analysis: анализ частоты спайков по классам
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..snn.snn_classifier import (
    BottleDefectDataset,
    CLASS_NAMES,
    CLASS_COLORS,
    SNNDefectClassifier,
    get_device,
)


# ---------------------------------------------------------------------------
# Растровый график спайков (Spike Raster Plot)
# ---------------------------------------------------------------------------

def plot_spike_raster(
    spikes: torch.Tensor,
    title: str = "Spike Raster Plot",
    neuron_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Растровый график спайков — классическая нейронаучная визуализация.

    Каждая строка = один нейрон, каждая точка = спайк во времени.

    Args:
        spikes: тензор формы [T, num_neurons] или [T, batch, num_neurons]
        title: заголовок
        neuron_labels: подписи нейронов (опционально)
        figsize: размер фигуры
        save_path: путь сохранения
    """
    if spikes.dim() == 3:
        # Берём первый батч
        spikes = spikes[:, 0, :]

    spikes_np = spikes.detach().cpu().numpy()  # [T, N]
    T, N = spikes_np.shape

    fig, ax = plt.subplots(figsize=figsize)

    for neuron_idx in range(N):
        spike_times = np.where(spikes_np[:, neuron_idx] > 0)[0]
        color = CLASS_COLORS[neuron_idx % len(CLASS_COLORS)]
        ax.vlines(spike_times, neuron_idx - 0.4, neuron_idx + 0.4,
                  color=color, linewidth=1.5, alpha=0.85)

    ax.set_xlim(-1, T)
    ax.set_ylim(-0.8, N - 0.2)
    ax.set_xlabel("Временной шаг", fontsize=11)
    ax.set_ylabel("Нейрон", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    if neuron_labels:
        ax.set_yticks(range(N))
        ax.set_yticklabels(neuron_labels, fontsize=10)
    else:
        ax.set_yticks(range(N))
        ax.set_yticklabels([f"Нейрон {i}" for i in range(N)], fontsize=10)

    # Частота спайков
    for i in range(N):
        rate = spikes_np[:, i].mean() * 100
        ax.text(T + 0.5, i, f"{rate:.0f}%", va="center", fontsize=9,
                color=CLASS_COLORS[i % len(CLASS_COLORS)])

    ax.text(T + 0.5, N, "Частота", va="center", fontsize=9,
            fontweight="bold", color="gray")

    ax.grid(True, alpha=0.2, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Примеры сигналов из датасета
# ---------------------------------------------------------------------------

def plot_dataset_samples(
    dataset: BottleDefectDataset,
    num_per_class: int = 3,
    figsize: Optional[Tuple] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Отображает по несколько примеров сигналов для каждого класса дефекта.

    Args:
        dataset: датасет дефектов закупорки
        num_per_class: сколько примеров на класс
        figsize: размер фигуры
        save_path: путь сохранения
    """
    num_classes = dataset.num_classes
    if figsize is None:
        figsize = (14, num_classes * 2.5)

    fig, axes = plt.subplots(num_classes, num_per_class, figsize=figsize,
                              sharey=False)
    fig.suptitle(
        "Примеры сигналов датчиков\nКонтроль дефектов закупорки молочной тары",
        fontsize=13, fontweight="bold"
    )

    for class_id in range(num_classes):
        indices = (dataset.labels == class_id).nonzero(as_tuple=True)[0]
        chosen = indices[:num_per_class]

        for col, idx in enumerate(chosen):
            ax = axes[class_id][col] if num_per_class > 1 else axes[class_id]
            signal = dataset.signals[idx].numpy()
            color = CLASS_COLORS[class_id]

            ax.plot(signal, color=color, linewidth=1.5, alpha=0.85)
            ax.fill_between(range(len(signal)), signal, alpha=0.15, color=color)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col == 0:
                ax.set_ylabel(CLASS_NAMES[class_id], fontsize=10,
                              fontweight="bold", color=color)
            if class_id == num_classes - 1:
                ax.set_xlabel("Шаг", fontsize=9)
            if class_id == 0:
                ax.set_title(f"Образец {col + 1}", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Распределение классов
# ---------------------------------------------------------------------------

def plot_class_distribution(
    dataset: BottleDefectDataset,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Столбчатая диаграмма распределения классов в датасете."""
    dist = dataset.get_class_distribution()
    names = [CLASS_NAMES[k] for k in sorted(dist)]
    counts = [dist[k] for k in sorted(dist)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Распределение классов дефектов", fontsize=12, fontweight="bold")

    bars = ax1.bar(range(len(names)), counts, color=CLASS_COLORS, alpha=0.85, edgecolor="white")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Количество образцов")
    ax1.set_title("Абсолютное число")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(cnt), ha="center", fontsize=10)

    wedges, texts, autotexts = ax2.pie(
        counts, labels=names, colors=CLASS_COLORS,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 9}
    )
    ax2.set_title("Доли классов")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Анализ частоты спайков по классам
# ---------------------------------------------------------------------------

def plot_spike_rate_analysis(
    model: SNNDefectClassifier,
    dataset: BottleDefectDataset,
    num_samples_per_class: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Анализирует и визуализирует частоту выходных спайков для каждого класса.

    Показывает, как SNN «кодирует» разные типы дефектов через паттерны спайков.

    Args:
        model: обученная SNN
        dataset: датасет
        num_samples_per_class: число образцов для анализа
        save_path: путь сохранения
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    num_classes = dataset.num_classes
    # spike_rates[class_id][output_neuron] = list of rates
    spike_rates = {c: {o: [] for o in range(num_classes)} for c in range(num_classes)}

    with torch.no_grad():
        for class_id in range(num_classes):
            indices = (dataset.labels == class_id).nonzero(as_tuple=True)[0]
            chosen = indices[:num_samples_per_class]

            for idx in chosen:
                x = dataset.signals[idx].unsqueeze(0).to(device)
                spike_counts, all_spikes = model(x)
                # all_spikes: [T, 1, num_classes]
                rates = all_spikes.squeeze(1).mean(0).cpu().numpy()  # [num_classes]
                for out_neuron in range(num_classes):
                    spike_rates[class_id][out_neuron].append(rates[out_neuron])

    # Матрица средних частот: строка = истинный класс, столбец = выходной нейрон
    rate_matrix = np.zeros((num_classes, num_classes))
    for c in range(num_classes):
        for o in range(num_classes):
            rate_matrix[c, o] = np.mean(spike_rates[c][o])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Анализ частоты спайков по классам дефектов", fontsize=12, fontweight="bold")

    # Тепловая карта
    im = ax1.imshow(rate_matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax1, label="Частота спайков")
    ax1.set_xticks(range(num_classes))
    ax1.set_yticks(range(num_classes))
    ax1.set_xticklabels(
        [f"Нейрон {i}\n({CLASS_NAMES[i][:8]}...)" for i in range(num_classes)],
        fontsize=8
    )
    ax1.set_yticklabels([CLASS_NAMES[i] for i in range(num_classes)], fontsize=9)
    ax1.set_xlabel("Выходной нейрон")
    ax1.set_ylabel("Истинный класс")
    ax1.set_title("Средняя частота спайков\n(строка=класс, столбец=нейрон)")
    for i in range(num_classes):
        for j in range(num_classes):
            ax1.text(j, i, f"{rate_matrix[i,j]:.2f}",
                     ha="center", va="center", fontsize=10,
                     color="white" if rate_matrix[i, j] > rate_matrix.max() * 0.6 else "black")

    # Box plots по классам
    for class_id in range(num_classes):
        data = [spike_rates[class_id][o] for o in range(num_classes)]
        positions = np.arange(num_classes) + class_id * 0.2 - 0.3
        bp = ax2.boxplot(data, positions=positions, widths=0.15,
                         patch_artist=True,
                         boxprops=dict(facecolor=CLASS_COLORS[class_id], alpha=0.5),
                         medianprops=dict(color=CLASS_COLORS[class_id], linewidth=2),
                         whiskerprops=dict(color=CLASS_COLORS[class_id], alpha=0.7),
                         capprops=dict(color=CLASS_COLORS[class_id]),
                         flierprops=dict(marker=".", color=CLASS_COLORS[class_id], alpha=0.4))

    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels([f"Нейрон {i}" for i in range(num_classes)], fontsize=9)
    ax2.set_ylabel("Частота спайков")
    ax2.set_title("Распределение частот по классам")
    ax2.grid(True, alpha=0.3, axis="y")

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(num_classes)]
    ax2.legend(handles=patches, fontsize=8, loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    return fig
