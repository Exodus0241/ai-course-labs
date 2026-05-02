"""
comparison.py — Сравнительный анализ SNN vs ANN

Метрики сравнения:
    1. Точность классификации (Accuracy, F1)
    2. Время обучения (Training Time)
    3. Время инференса (Inference Latency)
    4. Число параметров (Model Size)
    5. Оценка энергоэффективности (Synaptic Operations, SynOps)

Контекст: дефекты закупорки молочной тары.
SNN потенциально выгоднее для edge-устройств на производственной линии
(низкое энергопотребление, событийная обработка сигналов датчиков).
"""

import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, random_split

from snn_classifier import (
    BottleDefectDataset,
    CLASS_NAMES,
    CLASS_COLORS,
    SNNDefectClassifier,
    SNNTrainer,
    get_device,
    plot_training_history,
)


# ---------------------------------------------------------------------------
# ANN-классификатор (аналог по архитектуре)
# ---------------------------------------------------------------------------

class ANNDefectClassifier(nn.Module):
    """
    Классическая полносвязная ANN — аналог SNN по числу слоёв.

    Намеренно сохранена схожая глубина для честного сравнения.
    """

    def __init__(
        self,
        input_size: int = 64,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ANNTrainer:
    """Тренер для ANN — симметричный SNNTrainer для честного сравнения."""

    def __init__(
        self,
        model: ANNDefectClassifier,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
    ):
        self.model = model
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def _step(self, batch, train: bool) -> Tuple[float, float]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        if train:
            self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        if train:
            loss.backward()
            self.optimizer.step()
        acc = (logits.argmax(1) == y).float().mean().item()
        return loss.item(), acc

    def train_epoch(self, loader) -> Tuple[float, float]:
        self.model.train()
        ls, accs = zip(*[self._step(b, True) for b in loader])
        return float(np.mean(ls)), float(np.mean(accs))

    @torch.no_grad()
    def eval_epoch(self, loader) -> Tuple[float, float]:
        self.model.eval()
        ls, accs = zip(*[self._step(b, False) for b in loader])
        return float(np.mean(ls)), float(np.mean(accs))

    def fit(self, train_loader, val_loader, num_epochs=20, verbose=True):
        if verbose:
            print(f"\nОбучение ANN | параметров: {self.model.count_parameters():,}")
            print("-" * 55)
        for epoch in range(1, num_epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.eval_epoch(val_loader)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(
                    f"Эпоха {epoch:3d}/{num_epochs} | "
                    f"Train: loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                    f"Val:   loss={val_loss:.4f} acc={val_acc:.3f}"
                )
        return self.history

    @torch.no_grad()
    def predict(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds, labels = [], []
        for x, y in loader:
            x = x.to(self.device)
            p = self.model(x).argmax(1).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())
        return np.array(preds), np.array(labels)


# ---------------------------------------------------------------------------
# Подсчёт синаптических операций (SynOps) — прокси энергопотребления
# ---------------------------------------------------------------------------

def estimate_synaptic_ops(
    model: SNNDefectClassifier,
    loader: DataLoader,
    num_timesteps: int,
) -> float:
    """
    Оценивает среднее количество синаптических операций (SynOps) для SNN.

    SynOps считается только при наличии спайка (sparse computation).
    Это ключевое преимущество SNN: большинство нейронов молчат → экономия энергии.

    Формула: SynOps = mean(spike_count) * num_output_neurons_prev_layer
    """
    model.eval()
    total_ops = 0.0
    total_samples = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            batch = x.shape[0]

            # Считаем спайки на каждом слое
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()

            layer1_spikes = 0
            layer2_spikes = 0
            layer3_spikes = 0

            for _ in range(num_timesteps):
                x_enc = (torch.rand_like(x) < x).float()
                cur1 = model.fc1(x_enc)
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.fc2(spk1)
                spk2, mem2 = model.lif2(cur2, mem2)
                cur3 = model.fc3(spk2)
                spk3, mem3 = model.lif3(cur3, mem3)

                layer1_spikes += spk1.sum().item()
                layer2_spikes += spk2.sum().item()
                layer3_spikes += spk3.sum().item()

            # SynOps: спайки × число синапсов следующего слоя
            fc1_out, fc2_out, fc3_out = 128, 64, 4
            ops = (
                layer1_spikes * fc2_out +
                layer2_spikes * fc3_out +
                layer3_spikes
            )
            total_ops += ops
            total_samples += batch

    return total_ops / total_samples


def estimate_ann_ops(model: ANNDefectClassifier, input_size: int) -> float:
    """
    Оценивает число MACs (Multiply-Accumulate) для ANN на один образец.
    В ANN каждое соединение активно всегда — нет разреженности.
    """
    # Каждый Linear слой: input_features * output_features MACs
    layers = [l for l in model.net if isinstance(l, nn.Linear)]
    total = sum(l.in_features * l.out_features for l in layers)
    return float(total)


# ---------------------------------------------------------------------------
# Основной класс сравнения
# ---------------------------------------------------------------------------

class SNNvsANNComparison:
    """
    Полный сравнительный анализ SNN и ANN для классификации дефектов закупорки.

    Запускает оба обучения, замеряет метрики и строит итоговый отчёт.
    """

    def __init__(
        self,
        num_samples: int = 2000,
        signal_length: int = 64,
        num_classes: int = 4,
        num_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        num_timesteps: int = 25,
        seed: int = 42,
    ):
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.device = get_device()
        self.results: Dict = {}

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Датасет
        dataset = BottleDefectDataset(
            num_samples=num_samples,
            num_classes=num_classes,
            signal_length=signal_length,
            noise_level=0.1,
            seed=seed,
        )
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        g = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=g)

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Модели
        self.snn_model = SNNDefectClassifier(
            input_size=signal_length, hidden1=128, hidden2=64,
            num_classes=num_classes, beta=0.9, num_timesteps=num_timesteps,
        )
        self.ann_model = ANNDefectClassifier(
            input_size=signal_length, hidden1=128, hidden2=64,
            num_classes=num_classes,
        )

        self.snn_trainer = SNNTrainer(self.snn_model, self.device, lr)
        self.ann_trainer = ANNTrainer(self.ann_model, self.device, lr)

        self.num_epochs = num_epochs

    def run(self, verbose: bool = True) -> Dict:
        """Запускает полное сравнение SNN vs ANN."""
        if verbose:
            print("=" * 60)
            print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ: SNN vs ANN")
            print("Задача: классификация дефектов закупорки молочной тары")
            print("=" * 60)

        # --- SNN ---
        t0 = time.perf_counter()
        snn_history = self.snn_trainer.fit(
            self.train_loader, self.val_loader, self.num_epochs, verbose
        )
        snn_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        snn_preds, snn_labels = self.snn_trainer.predict(self.test_loader)
        snn_inference_time = (time.perf_counter() - t0) / len(snn_preds) * 1000  # мс/образец

        snn_acc = accuracy_score(snn_labels, snn_preds)
        snn_f1 = f1_score(snn_labels, snn_preds, average="macro")
        snn_synops = estimate_synaptic_ops(self.snn_model, self.test_loader, self.num_timesteps)

        # --- ANN ---
        t0 = time.perf_counter()
        ann_history = self.ann_trainer.fit(
            self.train_loader, self.val_loader, self.num_epochs, verbose
        )
        ann_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        ann_preds, ann_labels = self.ann_trainer.predict(self.test_loader)
        ann_inference_time = (time.perf_counter() - t0) / len(ann_preds) * 1000

        ann_acc = accuracy_score(ann_labels, ann_preds)
        ann_f1 = f1_score(ann_labels, ann_preds, average="macro")
        ann_ops = estimate_ann_ops(self.ann_model, 64)

        # --- Итог ---
        self.results = {
            "snn": {
                "accuracy": snn_acc,
                "f1_macro": snn_f1,
                "train_time_s": snn_train_time,
                "inference_ms": snn_inference_time,
                "parameters": self.snn_model.count_parameters(),
                "synops_per_sample": snn_synops,
                "history": snn_history,
                "preds": snn_preds,
                "labels": snn_labels,
            },
            "ann": {
                "accuracy": ann_acc,
                "f1_macro": ann_f1,
                "train_time_s": ann_train_time,
                "inference_ms": ann_inference_time,
                "parameters": self.ann_model.count_parameters(),
                "macs_per_sample": ann_ops,
                "history": ann_history,
                "preds": ann_preds,
                "labels": ann_labels,
            },
        }

        if verbose:
            self._print_summary()

        return self.results

    def _print_summary(self) -> None:
        snn = self.results["snn"]
        ann = self.results["ann"]

        print("\n" + "=" * 60)
        print("ИТОГОВОЕ СРАВНЕНИЕ")
        print("=" * 60)
        print(f"{'Метрика':<30} {'SNN':>12} {'ANN':>12}")
        print("-" * 56)
        print(f"{'Точность (Accuracy)':<30} {snn['accuracy']:>11.3f} {ann['accuracy']:>11.3f}")
        print(f"{'F1-мера (macro)':<30} {snn['f1_macro']:>11.3f} {ann['f1_macro']:>11.3f}")
        print(f"{'Параметров':<30} {snn['parameters']:>12,} {ann['parameters']:>12,}")
        print(f"{'Время обучения (с)':<30} {snn['train_time_s']:>11.1f} {ann['train_time_s']:>11.1f}")
        print(f"{'Инференс (мс/образец)':<30} {snn['inference_ms']:>11.4f} {ann['inference_ms']:>11.4f}")
        print(f"{'SynOps / MACs (на образец)':<30} {snn['synops_per_sample']:>11.0f} {ann['macs_per_sample']:>11.0f}")

        energy_ratio = ann["macs_per_sample"] / max(snn["synops_per_sample"], 1)
        print(f"\n→ SNN использует операций в {energy_ratio:.1f}x меньше (оценка энергоэффективности)")

    def plot_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Строит итоговый дашборд сравнения SNN vs ANN."""
        snn = self.results["snn"]
        ann = self.results["ann"]

        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)
        fig.suptitle(
            "Сравнительный анализ SNN vs ANN\nКонтроль дефектов закупорки молочной тары",
            fontsize=14, fontweight="bold"
        )

        epochs = range(1, self.num_epochs + 1)
        snn_color, ann_color = "#3498db", "#e74c3c"

        # 1. Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, snn["history"]["val_loss"], color=snn_color, label="SNN", linewidth=2)
        ax1.plot(epochs, ann["history"]["val_loss"], color=ann_color, label="ANN", linewidth=2, linestyle="--")
        ax1.set_title("Val Loss", fontweight="bold")
        ax1.set_xlabel("Эпоха")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, snn["history"]["val_acc"], color=snn_color, label="SNN", linewidth=2)
        ax2.plot(epochs, ann["history"]["val_acc"], color=ann_color, label="ANN", linewidth=2, linestyle="--")
        ax2.set_title("Val Accuracy", fontweight="bold")
        ax2.set_xlabel("Эпоха")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Гистограмма метрик
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = ["Accuracy", "F1-macro"]
        snn_vals = [snn["accuracy"], snn["f1_macro"]]
        ann_vals = [ann["accuracy"], ann["f1_macro"]]
        x = np.arange(len(metrics))
        w = 0.35
        ax3.bar(x - w/2, snn_vals, w, color=snn_color, alpha=0.85, label="SNN")
        ax3.bar(x + w/2, ann_vals, w, color=ann_color, alpha=0.85, label="ANN")
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.set_ylim(0, 1.1)
        ax3.set_title("Качество классификации", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")
        for i, (sv, av) in enumerate(zip(snn_vals, ann_vals)):
            ax3.text(i - w/2, sv + 0.02, f"{sv:.3f}", ha="center", fontsize=9, color=snn_color)
            ax3.text(i + w/2, av + 0.02, f"{av:.3f}", ha="center", fontsize=9, color=ann_color)

        # 4. Матрица ошибок — SNN
        ax4 = fig.add_subplot(gs[1, 0])
        cm_snn = confusion_matrix(snn["labels"], snn["preds"])
        self._plot_confusion_matrix(ax4, cm_snn, title="Матрица ошибок (SNN)", color=snn_color)

        # 5. Матрица ошибок — ANN
        ax5 = fig.add_subplot(gs[1, 1])
        cm_ann = confusion_matrix(ann["labels"], ann["preds"])
        self._plot_confusion_matrix(ax5, cm_ann, title="Матрица ошибок (ANN)", color=ann_color)

        # 6. Параметры и время обучения
        ax6 = fig.add_subplot(gs[1, 2])
        cats = ["Параметры\n(тыс.)", "Время обучения\n(с)"]
        snn_norm = [snn["parameters"] / 1000, snn["train_time_s"]]
        ann_norm = [ann["parameters"] / 1000, ann["train_time_s"]]
        x6 = np.arange(len(cats))
        ax6.bar(x6 - w/2, snn_norm, w, color=snn_color, alpha=0.85, label="SNN")
        ax6.bar(x6 + w/2, ann_norm, w, color=ann_color, alpha=0.85, label="ANN")
        ax6.set_xticks(x6)
        ax6.set_xticklabels(cats, fontsize=9)
        ax6.set_title("Ресурсы", fontweight="bold")
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")

        # 7. Энергоэффективность (SynOps vs MACs)
        ax7 = fig.add_subplot(gs[2, :2])
        names = ["SNN (SynOps)", "ANN (MACs)"]
        vals = [snn["synops_per_sample"], ann["macs_per_sample"]]
        colors_bar = [snn_color, ann_color]
        bars = ax7.barh(names, vals, color=colors_bar, alpha=0.85)
        ax7.set_xlabel("Операций на образец (меньше = лучше)")
        ax7.set_title("Оценка энергоэффективности\n(SynOps для SNN vs MACs для ANN)", fontweight="bold")
        ax7.grid(True, alpha=0.3, axis="x")
        for bar, val in zip(bars, vals):
            ax7.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:,.0f}", va="center", fontsize=10)

        energy_ratio = ann["macs_per_sample"] / max(snn["synops_per_sample"], 1)
        ax7.text(0.98, 0.1, f"SNN экономнее в {energy_ratio:.1f}x",
                 transform=ax7.transAxes, ha="right", fontsize=11,
                 fontweight="bold", color=snn_color,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=snn_color))

        # 8. Инференс
        ax8 = fig.add_subplot(gs[2, 2])
        inf_vals = [snn["inference_ms"], ann["inference_ms"]]
        ax8.bar(["SNN", "ANN"], inf_vals, color=[snn_color, ann_color], alpha=0.85)
        ax8.set_ylabel("Время (мс/образец)")
        ax8.set_title("Скорость инференса", fontweight="bold")
        ax8.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(inf_vals):
            ax8.text(i, v + max(inf_vals) * 0.02, f"{v:.4f}", ha="center", fontsize=10)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Дашборд сохранён: {save_path}")

        return fig

    @staticmethod
    def _plot_confusion_matrix(ax, cm, title, color):
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontweight="bold", fontsize=10)
        tick_labels = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
        ax.set_xticks(range(len(tick_labels)))
        ax.set_yticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Предсказано", fontsize=9)
        ax.set_ylabel("Истина", fontsize=9)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    comparison = SNNvsANNComparison(
        num_samples=20000,
        signal_length=64,
        num_classes=4,
        num_epochs=1,
        batch_size=64,
        lr=1e-3,
        num_timesteps=2,
        seed=42,
    )
    results = comparison.run(verbose=True)
    fig = comparison.plot_comparison(save_path="comparison_dashboard.png")
    plt.show()
