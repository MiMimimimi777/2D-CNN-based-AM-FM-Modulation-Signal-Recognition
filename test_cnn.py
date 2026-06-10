import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import spectrogram

class CNN2DModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(CNN2DModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def signal_to_spectrogram(signal, fs=1000, nperseg=32, noverlap=16):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx + 1e-10)
    Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx) + 1e-10)
    return Sxx

def am_modulation(carrier_freq, message_freq, amplitude=1, duration=1, fs=1000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    message = np.sin(2 * np.pi * message_freq * t)
    am_signal = (1 + amplitude * message) * carrier
    return t, am_signal

def fm_modulation(carrier_freq, message_freq, modulation_index=5, duration=1, fs=1000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    message = np.sin(2 * np.pi * message_freq * t)
    fm_signal = np.sin(2 * np.pi * carrier_freq * t + modulation_index * np.cos(2 * np.pi * message_freq * t))
    return t, fm_signal

def add_gaussian_noise(signal, snr_db=10):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise
    return noisy_signal

print("Testing CNN model...")

_, am_signal = am_modulation(100, 5)
_, fm_signal = fm_modulation(100, 5)

noisy_am = add_gaussian_noise(am_signal, 10)
noisy_fm = add_gaussian_noise(fm_signal, 10)

spec_am = signal_to_spectrogram(noisy_am)
spec_fm = signal_to_spectrogram(noisy_fm)

print(f"Spectrogram shape: {spec_am.shape}")

X = np.array([spec_am, spec_fm])
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
y = np.array([0, 1])

print(f"Input shape: {X.shape}")

model = CNN2DModel(input_channels=1, num_classes=2)
print("Model created successfully")

X_tensor = torch.tensor(X, dtype=torch.float32)
outputs = model(X_tensor)
print(f"Output shape: {outputs.shape}")
print(f"Output: {outputs}")

print("✓ CNN model test passed!")