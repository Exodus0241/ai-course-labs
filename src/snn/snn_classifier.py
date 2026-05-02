"""
snn_classifier.py — SNN-классификатор дефектов закупорки молочной тары

Архитектура:
    Вход (синтетический сигнал датчика)
        → rate encoding (интенсивность → частота спайков)
        → FC(signal_length → 128) + LIF
        → FC(128 → 64) + LIF
        → FC(64 → num_classes)
        → суммирование спайков по времени → softmax → класс дефекта

Классы дефектов:
    0 — Норма (герметичная закупорка)
    1 — Дефект давления (неплотная крышка)
    2 — Вибрационный дефект (крышка дребезжит)
    3 — Разгерметизация (утечка продукта)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Определение устройства (MPS → CPU fallback для Apple Silicon)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Автодетект устройства: MPS → CPU fallback."""
    if torch.backends.mps.is_available():
        try:
            # Простая проверка работоспособности MPS
            _test = torch.zeros(1).to("mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Синтетический датасет дефектов закупорки
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "Норма",
    1: "Дефект давления",
    2: "Вибрационный дефект",
    3: "Разгерметизация",
}

CLASS_COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]


def _generate_single_signal(
    class_id: int,
    signal_length: int,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Генерирует один синтетический сигнал датчика по классу дефекта."""
    t = np.linspace(0, 1, signal_length)

    if class_id == 0:  # Норма
        signal = np.ones(signal_length) * 0.25 + rng.normal(0, noise_level, signal_length)

    elif class_id == 1:  # Дефект давления
        signal = np.zeros(signal_length)
        pulse_center = int(rng.uniform(0.25, 0.65) * signal_length)
        pulse_width = signal_length // 8
        start = max(0, pulse_center - pulse_width // 2)
        end = min(signal_length, pulse_center + pulse_width // 2)
        signal[start:end] = rng.uniform(1.5, 2.5)
        signal += rng.normal(0, noise_level, signal_length)

    elif class_id == 2:  # Вибрационный дефект
        freq = rng.uniform(4, 8)
        signal = 0.7 * np.sin(2 * np.pi * freq * t) + 0.5
        signal += rng.normal(0, noise_level * 2, signal_length)

    elif class_id == 3:  # Разгерметизация
        signal = np.linspace(0.1, rng.uniform(1.5, 2.2), signal_length)
        signal += rng.normal(0, noise_level, signal_length)

    else:
        raise ValueError(f"Неизвестный класс: {class_id}")

    return np.clip(signal, 0.0, 3.0).astype(np.float32)


class BottleDefectDataset(Dataset):
    """
    Синтетический датасет сигналов датчиков линии закупорки.

    Каждый образец — временной ряд длиной signal_length,
    имитирующий показания датчика давления/вибрации.
    """

    def __init__(
        self,
        num_samples: int = 2000,
        num_classes: int = 4,
        signal_length: int = 64,
        noise_level: float = 0.1,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.signal_length = signal_length

        rng = np.random.default_rng(seed)
        samples_per_class = num_samples // num_classes

        signals = []
        labels = []

        for class_id in range(num_classes):
            n = samples_per_class if class_id < num_classes - 1 else (
                num_samples - samples_per_class * (num_classes - 1)
            )
            for _ in range(n):
                sig = _generate_single_signal(class_id, signal_length, noise_level, rng)
                signals.append(sig)
                labels.append(class_id)

        # Перемешиваем
        idx = rng.permutation(len(signals))
        self.signals = torch.tensor(np.array(signals)[idx], dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels)[idx], dtype=torch.long)

        # Нормализация к [0, 1]
        max_val = self.signals.max()
        if max_val > 0:
            self.signals = self.signals / max_val

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]

    def get_class_distribution(self) -> Dict[int, int]:
        """Возвращает распределение классов."""
        return {
            i: (self.labels == i).sum().item()
            for i in range(self.num_classes)
        }


# ---------------------------------------------------------------------------
# Архитектура SNN
# ---------------------------------------------------------------------------

class SNNDefectClassifier(nn.Module):
    """
    Импульсная нейронная сеть для классификации дефектов закупорки.

    Использует rate coding: значение сигнала → вероятность спайка на каждом шаге.
    Финальная классификация — по сумме выходных спайков за все временные шаги.

    Args:
        input_size: длина входного сигнала
        hidden1: нейронов в первом скрытом слое
        hidden2: нейронов во втором скрытом слое
        num_classes: количество классов дефектов
        beta: коэффициент утечки LIF-нейронов
        num_timesteps: число временных шагов при кодировании
    """

    def __init__(
        self,
        input_size: int = 64,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 4,
        beta: float = 0.9,
        num_timesteps: int = 25,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # Линейные слои
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

        # LIF-нейроны (learnable beta)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=True)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True)
        self.lif3 = snn.Leaky(beta=beta, learn_beta=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход: rate encoding → SNN → суммирование спайков.

        Args:
            x: входной сигнал [batch, input_size]

        Returns:
            (spike_count_output [batch, num_classes], all_spikes [T, batch, num_classes])
        """
        # Инициализация мембранных потенциалов
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_list = []

        # Rate encoding: на каждом шаге генерируем бинарный вход с вероятностью = x
        for _ in range(self.num_timesteps):
            x_enc = (torch.rand_like(x) < x).float()

            cur1 = self.fc1(x_enc)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_list.append(spk3)

        all_spikes = torch.stack(spk3_list, dim=0)       # [T, batch, num_classes]
        spike_counts = all_spikes.sum(dim=0)              # [batch, num_classes]

        return spike_counts, all_spikes

    def count_parameters(self) -> int:
        """Возвращает количество обучаемых параметров."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Тренер SNN
# ---------------------------------------------------------------------------

class SNNTrainer:
    """
    Обёртка для обучения и оценки SNN-классификатора.

    Args:
        model: экземпляр SNNDefectClassifier
        device: устройство вычислений
        lr: скорость обучения
    """

    def __init__(
        self,
        model: SNNDefectClassifier,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
    ):
        self.model = model
        self.device = device or get_device()
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # CE по числу спайков — стандартная практика для SNN
        self.loss_fn = nn.CrossEntropyLoss()

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def _step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], train: bool
    ) -> Tuple[float, float]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        if train:
            self.optimizer.zero_grad()

        spike_counts, _ = self.model(x)
        loss = self.loss_fn(spike_counts, y)

        if train:
            loss.backward()
            self.optimizer.step()

        preds = spike_counts.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        return loss.item(), acc

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        losses, accs = [], []
        for batch in loader:
            l, a = self._step(batch, train=True)
            losses.append(l)
            accs.append(a)
        return float(np.mean(losses)), float(np.mean(accs))

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        losses, accs = [], []
        for batch in loader:
            l, a = self._step(batch, train=False)
            losses.append(l)
            accs.append(a)
        return float(np.mean(losses)), float(np.mean(accs))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Полный цикл обучения.

        Args:
            train_loader: загрузчик тренировочных данных
            val_loader: загрузчик валидационных данных
            num_epochs: количество эпох
            verbose: выводить ли прогресс

        Returns:
            Словарь с историей метрик
        """
        if verbose:
            print(f"\nОбучение SNN на устройстве: {self.device}")
            print(f"Параметров модели: {self.model.count_parameters():,}")
            print("-" * 55)

        for epoch in tqdm(range(1, num_epochs + 1), desc="Эпохи", disable=not verbose):
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

        if verbose:
            print("-" * 55)
            print(f"Финальная точность (val): {self.history['val_acc'][-1]:.3f}")

        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает предсказания и истинные метки."""
        self.model.eval()
        all_preds, all_labels = [], []
        for x, y in loader:
            x = x.to(self.device)
            spike_counts, _ = self.model(x)
            preds = spike_counts.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
        return np.array(all_preds), np.array(all_labels)

    def save(self, path: str) -> None:
        """Сохраняет веса модели."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Модель сохранена: {path}")

    def load(self, path: str) -> None:
        """Загружает веса модели."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Модель загружена: {path}")


# ---------------------------------------------------------------------------
# Утилита: визуализация истории обучения
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "История обучения SNN",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color="#3498db")
    ax1.plot(epochs, history["val_loss"], label="Val", color="#e74c3c", linestyle="--")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss (CE)")
    ax1.set_title("Функция потерь")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", color="#3498db")
    ax2.plot(epochs, history["val_acc"], label="Val", color="#e74c3c", linestyle="--")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Точность")
    ax2.set_title("Точность классификации")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEVICE = get_device()
    print(f"Устройство: {DEVICE}")

    # Датасет
    dataset = BottleDefectDataset(
        num_samples=2000, num_classes=4, signal_length=64, noise_level=0.1, seed=42
    )
    print(f"Датасет: {len(dataset)} образцов")
    print(f"Распределение классов: {dataset.get_class_distribution()}")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Модель
    model = SNNDefectClassifier(
        input_size=64, hidden1=128, hidden2=64, num_classes=4,
        beta=0.9, num_timesteps=25
    )
    trainer = SNNTrainer(model, device=DEVICE, lr=1e-3)

    # Обучение
    history = trainer.fit(train_loader, val_loader, num_epochs=20)

    # Оценка
    preds, labels = trainer.predict(test_loader)
    test_acc = (preds == labels).mean()
    print(f"\nТочность на тесте: {test_acc:.3f}")

    plot_training_history(history, save_path="snn_training.png")
    plt.show()
