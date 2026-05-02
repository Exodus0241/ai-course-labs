"""
lif_neuron.py — Реализация нейрона с утечкой и интеграцией (Leaky Integrate-and-Fire)

Модель LIF описывает динамику мембранного потенциала нейрона:
    dV/dt = -(V - V_rest) / tau + I(t)

В дискретном виде (Euler):
    V[t] = beta * V[t-1] + I[t]
    Если V[t] >= threshold → spike, V[t] = 0 (сброс)

Адаптировано для задачи: контроль дефектов закупорки молочной тары.
Входные сигналы — временные ряды датчиков давления/вибрации линии розлива.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

@dataclass
class LIFNeuronConfig:
    """Гиперпараметры одного LIF-нейрона."""

    beta: float = 0.9
    """Коэффициент утечки (decay factor). beta=1 — нет утечки, beta→0 — мгновенная утечка."""

    threshold: float = 1.0
    """Порог срабатывания (spike threshold)."""

    reset_mechanism: str = "zero"
    """Механизм сброса: 'zero' (V=0) или 'subtract' (V -= threshold)."""

    num_timesteps: int = 25
    """Количество временных шагов симуляции."""

    def __post_init__(self):
        assert 0.0 < self.beta <= 1.0, "beta должен быть в диапазоне (0, 1]"
        assert self.threshold > 0.0, "threshold должен быть положительным"
        assert self.reset_mechanism in ("zero", "subtract"), (
            "reset_mechanism: 'zero' или 'subtract'"
        )


# ---------------------------------------------------------------------------
# Класс LIF-нейрона
# ---------------------------------------------------------------------------

class LIFNeuron(nn.Module):
    """
    Одиночный LIF-нейрон на базе snntorch.Leaky.

    Демонстрирует базовую нейронную динамику:
    - интеграцию входного тока
    - экспоненциальную утечку мембранного потенциала
    - генерацию спайка при достижении порога
    - сброс потенциала после спайка

    Применение в контексте дипломной работы:
        Каждый нейрон может кодировать один тип аномалии:
        нарушение давления, вибрацию крышки, утечку и т.д.
    """

    def __init__(self, config: Optional[LIFNeuronConfig] = None):
        super().__init__()
        self.config = config or LIFNeuronConfig()

        # snntorch.Leaky реализует LIF с обратным распространением ошибки
        # через суррогатный градиент (fast sigmoid по умолчанию)
        self.lif = snn.Leaky(
            beta=self.config.beta,
            threshold=self.config.threshold,
            reset_mechanism=self.config.reset_mechanism,
        )

    def forward(
        self, input_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Один шаг симуляции нейрона.

        Args:
            input_current: входной ток формы [batch, neurons] или [neurons]

        Returns:
            (spike, mem): спайк (0/1) и мембранный потенциал
        """
        mem = self.lif.init_leaky()
        spikes_out = []
        mem_out = []

        for _ in range(self.config.num_timesteps):
            spike, mem = self.lif(input_current, mem)
            spikes_out.append(spike)
            mem_out.append(mem)

        return torch.stack(spikes_out), torch.stack(mem_out)

    def simulate(
        self,
        input_currents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Симуляция нейрона при изменяющемся входном токе.

        Args:
            input_currents: временной ряд токов формы [T] или [T, batch]

        Returns:
            (spikes [T], mem_potentials [T])
        """
        if input_currents.dim() == 1:
            input_currents = input_currents.unsqueeze(1)  # [T, 1]

        T = input_currents.shape[0]
        mem = self.lif.init_leaky()
        spikes_list = []
        mem_list = []

        for t in range(T):
            spike, mem = self.lif(input_currents[t], mem)
            spikes_list.append(spike.squeeze())
            mem_list.append(mem.squeeze())

        return torch.stack(spikes_list), torch.stack(mem_list)


# ---------------------------------------------------------------------------
# Генерация тестовых сигналов (аналог сигналов датчиков линии розлива)
# ---------------------------------------------------------------------------

def generate_sensor_signal(
    signal_type: str = "normal",
    duration: int = 100,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Генерирует синтетический сигнал датчика для демонстрации LIF-нейрона.

    Типы сигналов соответствуют состояниям линии закупорки:
        'normal'      — норма (низкий постоянный ток)
        'pressure'    — скачок давления (импульсный ток)
        'vibration'   — вибрация крышки (синусоидальный ток)
        'leak'        — утечка (нарастающий ток)

    Args:
        signal_type: тип сигнала
        duration: длина временного ряда
        noise_level: уровень гауссовского шума
        seed: seed для воспроизводимости

    Returns:
        Тензор формы [duration]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    t = np.linspace(0, 1, duration)

    if signal_type == "normal":
        signal = np.ones(duration) * 0.3

    elif signal_type == "pressure":
        # Резкий импульс — нарушение давления при закупорке
        signal = np.zeros(duration)
        pulse_start = duration // 3
        pulse_end = pulse_start + duration // 8
        signal[pulse_start:pulse_end] = 2.0

    elif signal_type == "vibration":
        # Синусоидальный сигнал — вибрация плохо закреплённой крышки
        signal = 0.8 * np.sin(2 * np.pi * 5 * t) + 0.5

    elif signal_type == "leak":
        # Линейно нарастающий сигнал — постепенная разгерметизация
        signal = np.linspace(0.1, 1.8, duration)

    else:
        raise ValueError(f"Неизвестный тип сигнала: {signal_type}")

    noise = np.random.normal(0, noise_level, duration)
    signal = np.clip(signal + noise, 0.0, 3.0)

    return torch.tensor(signal, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Визуализация динамики LIF-нейрона
# ---------------------------------------------------------------------------

def plot_lif_dynamics(
    neuron: LIFNeuron,
    signal_types: Optional[list] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Строит графики мембранного потенциала и спайков для разных входных сигналов.

    Args:
        neuron: экземпляр LIFNeuron
        signal_types: список типов сигналов для отображения
        figsize: размер фигуры
        save_path: путь для сохранения (опционально)

    Returns:
        matplotlib Figure
    """
    if signal_types is None:
        signal_types = ["normal", "pressure", "vibration", "leak"]

    colors = {
        "normal": "#2ecc71",
        "pressure": "#e74c3c",
        "vibration": "#3498db",
        "leak": "#f39c12",
    }
    labels = {
        "normal": "Норма",
        "pressure": "Скачок давления",
        "vibration": "Вибрация крышки",
        "leak": "Разгерметизация",
    }

    n = len(signal_types)
    fig, axes = plt.subplots(n, 3, figsize=figsize)
    fig.suptitle(
        "Динамика LIF-нейрона\nКонтроль дефектов закупорки молочной тары",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for row, sig_type in enumerate(signal_types):
        signal = generate_sensor_signal(sig_type, duration=100, seed=42)
        color = colors[sig_type]

        with torch.no_grad():
            spikes, mem_potentials = neuron.simulate(signal)

        spikes_np = spikes.numpy()
        mem_np = mem_potentials.numpy()
        signal_np = signal.numpy()
        t = np.arange(len(signal_np))

        # --- Входной сигнал ---
        ax0 = axes[row, 0]
        ax0.plot(t, signal_np, color=color, linewidth=1.5)
        ax0.axhline(y=neuron.config.threshold, color="red", linestyle="--",
                    linewidth=1, alpha=0.5, label=f"Порог={neuron.config.threshold}")
        ax0.set_ylabel(labels[sig_type], fontsize=9, fontweight="bold")
        ax0.set_title("Входной ток" if row == 0 else "", fontsize=10)
        ax0.set_ylim(-0.1, 2.5)
        ax0.grid(True, alpha=0.3)
        if row == 0:
            ax0.legend(fontsize=8)

        # --- Мембранный потенциал ---
        ax1 = axes[row, 1]
        ax1.plot(t, mem_np, color=color, linewidth=1.5, label="Потенциал")
        ax1.axhline(y=neuron.config.threshold, color="red", linestyle="--",
                    linewidth=1, alpha=0.7, label="Порог")
        ax1.set_title("Мембранный потенциал" if row == 0 else "", fontsize=10)
        ax1.set_ylim(-0.1, 1.5)
        ax1.grid(True, alpha=0.3)
        if row == 0:
            ax1.legend(fontsize=8)

        # --- Спайки ---
        ax2 = axes[row, 2]
        spike_times = np.where(spikes_np > 0)[0]
        spike_count = len(spike_times)
        ax2.vlines(spike_times, 0, 1, color=color, linewidth=2, alpha=0.9)
        ax2.set_title("Спайки" if row == 0 else "", fontsize=10)
        ax2.set_ylim(-0.1, 1.3)
        ax2.set_yticks([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.text(0.98, 0.95, f"Σ={spike_count}",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=9, fontweight="bold", color=color)

        for ax in [ax0, ax1, ax2]:
            ax.set_xlim(0, len(t))
            if row == n - 1:
                ax.set_xlabel("Временной шаг", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Точка входа — демонстрация
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LIF-нейрон: симуляция для задачи контроля закупорки")
    print("=" * 60)

    config = LIFNeuronConfig(beta=0.9, threshold=1.0, num_timesteps=25)
    neuron = LIFNeuron(config)
    print(f"\nКонфигурация нейрона:")
    print(f"  beta (утечка)    = {config.beta}")
    print(f"  threshold (порог) = {config.threshold}")
    print(f"  reset_mechanism  = {config.reset_mechanism}")

    print("\nСимуляция для разных типов сигналов датчиков:")
    for sig_type in ["normal", "pressure", "vibration", "leak"]:
        signal = generate_sensor_signal(sig_type, duration=100, seed=42)
        with torch.no_grad():
            spikes, mem = neuron.simulate(signal)
        spike_count = spikes.sum().item()
        avg_mem = mem.mean().item()
        print(f"  [{sig_type:12s}] спайков: {int(spike_count):3d}  |  "
              f"ср. потенциал: {avg_mem:.4f}")

    fig = plot_lif_dynamics(neuron, save_path="lif_dynamics.png")
    plt.show()
    print("\nГотово!")
