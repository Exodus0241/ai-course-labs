"""
test_snn.py — Тесты для компонентов SNN

Запуск:
    pytest tests/test_snn.py -v

Тесты покрывают:
    - Конфигурацию и инициализацию LIF-нейрона
    - Форму выходных тензоров (спайки, мембранный потенциал)
    - Синтетический датасет (размер, классы, формы)
    - Архитектуру SNN-классификатора
    - Один шаг прямого прохода
    - Детерминизм при фиксированном seed
    - Smoke-тест обучения (1 эпоха)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.snn.lif_neuron import (
    LIFNeuron,
    LIFNeuronConfig,
    generate_sensor_signal,
)
from src.snn.snn_classifier import (
    BottleDefectDataset,
    SNNDefectClassifier,
    SNNTrainer,
    get_device,
)
from src.snn.comparison import ANNDefectClassifier, estimate_ann_ops


# ===========================================================================
# Фикстуры
# ===========================================================================

@pytest.fixture(scope="module")
def default_config() -> LIFNeuronConfig:
    return LIFNeuronConfig(beta=0.9, threshold=1.0, num_timesteps=10)


@pytest.fixture(scope="module")
def lif_neuron(default_config) -> LIFNeuron:
    return LIFNeuron(default_config)


@pytest.fixture(scope="module")
def small_dataset() -> BottleDefectDataset:
    return BottleDefectDataset(
        num_samples=200, num_classes=4, signal_length=32,
        noise_level=0.1, seed=42
    )


@pytest.fixture(scope="module")
def snn_model() -> SNNDefectClassifier:
    return SNNDefectClassifier(
        input_size=32, hidden1=32, hidden2=16,
        num_classes=4, beta=0.9, num_timesteps=5,
    )


@pytest.fixture(scope="module")
def ann_model() -> ANNDefectClassifier:
    return ANNDefectClassifier(
        input_size=32, hidden1=32, hidden2=16, num_classes=4
    )


# ===========================================================================
# Тесты: LIFNeuronConfig
# ===========================================================================

class TestLIFNeuronConfig:
    def test_default_values(self):
        cfg = LIFNeuronConfig()
        assert cfg.beta == 0.9
        assert cfg.threshold == 1.0
        assert cfg.reset_mechanism == "zero"
        assert cfg.num_timesteps == 25

    def test_custom_values(self):
        cfg = LIFNeuronConfig(beta=0.5, threshold=0.8, num_timesteps=15)
        assert cfg.beta == 0.5
        assert cfg.threshold == 0.8
        assert cfg.num_timesteps == 15

    def test_invalid_beta_zero(self):
        with pytest.raises((AssertionError, ValueError)):
            LIFNeuronConfig(beta=0.0)

    def test_invalid_beta_negative(self):
        with pytest.raises((AssertionError, ValueError)):
            LIFNeuronConfig(beta=-0.1)

    def test_invalid_threshold(self):
        with pytest.raises((AssertionError, ValueError)):
            LIFNeuronConfig(threshold=-0.5)

    def test_invalid_reset_mechanism(self):
        with pytest.raises((AssertionError, ValueError)):
            LIFNeuronConfig(reset_mechanism="invalid")

    def test_valid_reset_mechanisms(self):
        for mech in ["zero", "subtract"]:
            cfg = LIFNeuronConfig(reset_mechanism=mech)
            assert cfg.reset_mechanism == mech


# ===========================================================================
# Тесты: LIFNeuron
# ===========================================================================

class TestLIFNeuron:
    def test_instantiation(self, lif_neuron):
        assert lif_neuron is not None

    def test_forward_output_shape(self, lif_neuron):
        """forward() должен возвращать (spikes, mem) c корректными формами."""
        x = torch.tensor([0.5])
        spikes, mem = lif_neuron(x)
        T = lif_neuron.config.num_timesteps
        assert spikes.shape[0] == T
        assert mem.shape[0] == T

    def test_simulate_output_shape(self, lif_neuron):
        """simulate() должен вернуть тензоры длиной duration."""
        duration = 50
        signal = generate_sensor_signal("normal", duration=duration, seed=0)
        spikes, mem = lif_neuron.simulate(signal)
        assert spikes.shape == (duration,)
        assert mem.shape == (duration,)

    def test_spikes_are_binary(self, lif_neuron):
        """Спайки должны быть 0 или 1."""
        signal = generate_sensor_signal("pressure", duration=50, seed=0)
        spikes, _ = lif_neuron.simulate(signal)
        unique = torch.unique(spikes)
        for v in unique:
            assert v.item() in (0.0, 1.0), f"Неожиданное значение спайка: {v}"

    def test_no_spikes_for_zero_input(self, default_config):
        """При нулевом входе спайков не должно быть."""
        neuron = LIFNeuron(default_config)
        signal = torch.zeros(30)
        spikes, _ = neuron.simulate(signal)
        assert spikes.sum().item() == 0.0

    def test_more_spikes_for_high_input(self, default_config):
        """Высокий вход → больше спайков, чем низкий."""
        neuron = LIFNeuron(default_config)
        low_signal = torch.ones(50) * 0.2
        high_signal = torch.ones(50) * 2.0
        spikes_low, _ = neuron.simulate(low_signal)
        spikes_high, _ = neuron.simulate(high_signal)
        assert spikes_high.sum() >= spikes_low.sum()

    def test_membrane_potential_range(self, lif_neuron):
        """Мембранный потенциал не должен превышать порог (при reset='zero')."""
        signal = generate_sensor_signal("vibration", duration=80, seed=1)
        _, mem = lif_neuron.simulate(signal)
        # Потенциал до сброса должен быть <= threshold + небольшой допуск
        assert mem.max().item() <= lif_neuron.config.threshold + 0.5


# ===========================================================================
# Тесты: generate_sensor_signal
# ===========================================================================

class TestGenerateSensorSignal:
    @pytest.mark.parametrize("sig_type", ["normal", "pressure", "vibration", "leak"])
    def test_output_shape(self, sig_type):
        signal = generate_sensor_signal(sig_type, duration=64)
        assert signal.shape == (64,)

    @pytest.mark.parametrize("sig_type", ["normal", "pressure", "vibration", "leak"])
    def test_output_range(self, sig_type):
        signal = generate_sensor_signal(sig_type, duration=64, noise_level=0.0)
        assert signal.min().item() >= 0.0
        assert signal.max().item() <= 3.0

    def test_dtype(self):
        signal = generate_sensor_signal("normal")
        assert signal.dtype == torch.float32

    def test_reproducibility_with_seed(self):
        s1 = generate_sensor_signal("pressure", seed=7)
        s2 = generate_sensor_signal("pressure", seed=7)
        assert torch.allclose(s1, s2)

    def test_invalid_signal_type(self):
        with pytest.raises(ValueError):
            generate_sensor_signal("unknown_type")


# ===========================================================================
# Тесты: BottleDefectDataset
# ===========================================================================

class TestBottleDefectDataset:
    def test_length(self, small_dataset):
        assert len(small_dataset) == 200

    def test_item_shapes(self, small_dataset):
        x, y = small_dataset[0]
        assert x.shape == (32,)
        assert y.shape == ()

    def test_label_range(self, small_dataset):
        labels = small_dataset.labels
        assert labels.min().item() >= 0
        assert labels.max().item() < 4

    def test_signal_range(self, small_dataset):
        assert small_dataset.signals.min().item() >= 0.0
        assert small_dataset.signals.max().item() <= 1.0

    def test_class_distribution_keys(self, small_dataset):
        dist = small_dataset.get_class_distribution()
        assert set(dist.keys()) == {0, 1, 2, 3}

    def test_class_distribution_total(self, small_dataset):
        dist = small_dataset.get_class_distribution()
        assert sum(dist.values()) == len(small_dataset)

    def test_reproducibility(self):
        ds1 = BottleDefectDataset(num_samples=100, signal_length=32, seed=99)
        ds2 = BottleDefectDataset(num_samples=100, signal_length=32, seed=99)
        assert torch.allclose(ds1.signals, ds2.signals)


# ===========================================================================
# Тесты: SNNDefectClassifier
# ===========================================================================

class TestSNNDefectClassifier:
    def test_parameter_count(self, snn_model):
        n = snn_model.count_parameters()
        assert n > 0
        print(f"\nSNN параметров: {n:,}")

    def test_forward_output_shapes(self, snn_model):
        batch = 8
        x = torch.rand(batch, 32)
        spike_counts, all_spikes = snn_model(x)
        assert spike_counts.shape == (batch, 4), f"Ожидалось ({batch}, 4), получено {spike_counts.shape}"
        assert all_spikes.shape == (5, batch, 4), f"Ожидалось (5, {batch}, 4), получено {all_spikes.shape}"

    def test_spike_counts_non_negative(self, snn_model):
        x = torch.rand(4, 32)
        spike_counts, _ = snn_model(x)
        assert (spike_counts >= 0).all()

    def test_spike_counts_leq_timesteps(self, snn_model):
        x = torch.ones(4, 32)
        spike_counts, _ = snn_model(x)
        assert (spike_counts <= snn_model.num_timesteps).all()

    def test_output_dtype(self, snn_model):
        x = torch.rand(2, 32)
        spike_counts, _ = snn_model(x)
        assert spike_counts.dtype == torch.float32

    def test_no_nan_in_output(self, snn_model):
        x = torch.rand(8, 32)
        spike_counts, all_spikes = snn_model(x)
        assert not torch.isnan(spike_counts).any()
        assert not torch.isnan(all_spikes).any()


# ===========================================================================
# Тесты: ANNDefectClassifier
# ===========================================================================

class TestANNDefectClassifier:
    def test_forward_shape(self, ann_model):
        x = torch.rand(8, 32)
        out = ann_model(x)
        assert out.shape == (8, 4)

    def test_parameter_count(self, ann_model):
        assert ann_model.count_parameters() > 0

    def test_estimate_ann_ops(self, ann_model):
        ops = estimate_ann_ops(ann_model, input_size=32)
        assert ops > 0


# ===========================================================================
# Тест: Smoke-обучение SNN (1 эпоха)
# ===========================================================================

class TestSNNTrainer:
    def test_smoke_training(self, small_dataset, snn_model):
        """Проверяет, что обучение проходит без ошибок за 1 эпоху."""
        train_size = int(0.8 * len(small_dataset))
        val_size = len(small_dataset) - train_size
        train_ds, val_ds = random_split(
            small_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(0)
        )
        train_loader = DataLoader(train_ds, batch_size=32)
        val_loader = DataLoader(val_ds, batch_size=32)

        model = SNNDefectClassifier(
            input_size=32, hidden1=32, hidden2=16,
            num_classes=4, beta=0.9, num_timesteps=5
        )
        trainer = SNNTrainer(model, device=torch.device("cpu"), lr=1e-3)
        history = trainer.fit(train_loader, val_loader, num_epochs=1, verbose=False)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert not np.isnan(history["train_loss"][0])
        assert 0.0 <= history["train_acc"][0] <= 1.0

    def test_predict_returns_correct_shapes(self, small_dataset, snn_model):
        """predict() должен вернуть корректные формы массивов."""
        loader = DataLoader(small_dataset, batch_size=32)
        trainer = SNNTrainer(snn_model, device=torch.device("cpu"))
        preds, labels = trainer.predict(loader)

        assert preds.shape == labels.shape
        assert len(preds) == len(small_dataset)
        assert set(np.unique(preds)).issubset({0, 1, 2, 3})


# ===========================================================================
# Тест: Устройство
# ===========================================================================

class TestDevice:
    def test_get_device_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "mps", "cuda")

    def test_model_moves_to_device(self, snn_model):
        device = torch.device("cpu")
        snn_model.to(device)
        for param in snn_model.parameters():
            assert param.device.type == "cpu"
