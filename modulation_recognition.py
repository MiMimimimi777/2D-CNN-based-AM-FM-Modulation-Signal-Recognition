import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, spectrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USE_CNN = True
except ImportError:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    USE_CNN = False

def generate_signal(freq=1, duration=1, fs=1000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

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

def am_demodulation(am_signal, fs=1000):
    analytic_signal = hilbert(am_signal)
    envelope = np.abs(analytic_signal)
    b, a = butter(4, 10, btype='low', fs=fs)
    demodulated = filtfilt(b, a, envelope)
    demodulated = demodulated - np.mean(demodulated)
    return demodulated[:-1]

def fm_demodulation(fm_signal, fs=1000, message_freq=5):
    analytic_signal = hilbert(fm_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_freq = np.diff(instantaneous_phase) * fs / (2 * np.pi)
    
    cutoff_freq = message_freq * 2
    b, a = butter(6, cutoff_freq, btype='low', fs=fs)
    demodulated = filtfilt(b, a, instantaneous_freq)
    
    demodulated = demodulated - np.mean(demodulated)
    demodulated = demodulated / np.max(np.abs(demodulated))
    
    return demodulated

def signal_to_spectrogram(signal, fs=1000, nperseg=32, noverlap=16):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx + 1e-10)
    Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx) + 1e-10)
    return Sxx

def signal_to_gaf(signal, image_size=64):
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
    signal = 2 * signal - 1
    
    scaled_signal = np.outer(signal, signal)
    GAF_cos = np.arccos(scaled_signal)
    
    GAF_cos = (GAF_cos - np.min(GAF_cos)) / (np.max(GAF_cos) - np.min(GAF_cos) + 1e-10)
    
    if image_size != GAF_cos.shape[0]:
        from scipy.ndimage import zoom
        GAF_cos = zoom(GAF_cos, image_size / GAF_cos.shape[0])
    
    return GAF_cos

def generate_training_data_cnn(num_samples=1000, duration=1, fs=1000, image_type='spectrogram', image_size=64):
    X = []
    y = []
    
    for _ in range(num_samples // 2):
        carrier_freq = random.uniform(50, 150)
        message_freq = random.uniform(1, 10)
        snr = random.uniform(0, 20)
        
        _, am_signal = am_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
        noisy_am = add_gaussian_noise(am_signal, snr)
        
        if image_type == 'spectrogram':
            img = signal_to_spectrogram(noisy_am, fs=fs)
        else:
            img = signal_to_gaf(noisy_am, image_size=image_size)
        
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        X.append(img)
        y.append('AM')
        
        _, fm_signal = fm_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
        noisy_fm = add_gaussian_noise(fm_signal, snr)
        
        if image_type == 'spectrogram':
            img = signal_to_spectrogram(noisy_fm, fs=fs)
        else:
            img = signal_to_gaf(noisy_fm, image_size=image_size)
        
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        X.append(img)
        y.append('FM')
    
    X = np.array(X)
    X = np.transpose(X, (0, 3, 1, 2))
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le

def generate_training_data_svm(num_samples=1000, duration=1, fs=1000):
    X = []
    y = []
    
    for _ in range(num_samples // 2):
        carrier_freq = random.uniform(50, 150)
        message_freq = random.uniform(1, 10)
        snr = random.uniform(0, 20)
        
        _, am_signal = am_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
        noisy_am = add_gaussian_noise(am_signal, snr)
        X.append(extract_features(noisy_am, fs=fs))
        y.append('AM')
        
        _, fm_signal = fm_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
        noisy_fm = add_gaussian_noise(fm_signal, snr)
        X.append(extract_features(noisy_fm, fs=fs))
        y.append('FM')
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_features(signal, fs=1000):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_freq = np.diff(instantaneous_phase) * fs / (2 * np.pi)
    
    features = []
    features.append(np.std(amplitude_envelope))
    features.append(np.mean(np.abs(np.diff(amplitude_envelope))))
    features.append(np.std(instantaneous_freq))
    features.append(np.mean(np.abs(np.diff(instantaneous_freq))))
    features.append(np.max(amplitude_envelope) - np.min(amplitude_envelope))
    features.append(np.mean(amplitude_envelope))
    
    return np.array(features)

if USE_CNN:
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
                nn.MaxPool2d(2, 2)
            )
            
            self.fc_layers = nn.Sequential(
                nn.Linear(256 * 2 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x

if USE_CNN:
    def train_cnn_model(X_train, y_train, epochs=15, batch_size=32):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN2DModel(input_channels=1, num_classes=2).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            
            if (epoch + 1) % 3 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        return model

    def evaluate_cnn_model(model, X_test, y_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / len(y_test)
        
        return accuracy

    def predict_cnn_model(model, signal, image_type='spectrogram', fs=1000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        if image_type == 'spectrogram':
            img = signal_to_spectrogram(signal, fs=fs)
        else:
            img = signal_to_gaf(signal)
        
        if len(img.shape) == 2:
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
        
        img_tensor = torch.tensor(img, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)
        
        return predicted.item(), probabilities[0][predicted.item()].item()

def build_svm_model():
    scaler = StandardScaler()
    clf = SVC(kernel='rbf', probability=True)
    return scaler, clf

def plot_signals(t, original, am_mod, fm_mod, noisy_am, noisy_fm, random_signal=None):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    axes[0, 0].plot(t, original)
    axes[0, 0].set_title('Original Signal', fontsize=12)
    axes[0, 0].set_xlabel('Time (s)', fontsize=10)
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(t, am_mod)
    axes[0, 1].set_title('AM Modulated Signal', fontsize=12)
    axes[0, 1].set_xlabel('Time (s)', fontsize=10)
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(t, fm_mod)
    axes[1, 0].set_title('FM Modulated Signal', fontsize=12)
    axes[1, 0].set_xlabel('Time (s)', fontsize=10)
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(t, noisy_am)
    axes[1, 1].set_title('Noisy AM Signal', fontsize=12)
    axes[1, 1].set_xlabel('Time (s)', fontsize=10)
    axes[1, 1].grid(True)
    
    axes[2, 0].plot(t, noisy_fm)
    axes[2, 0].set_title('Noisy FM Signal', fontsize=12)
    axes[2, 0].set_xlabel('Time (s)', fontsize=10)
    axes[2, 0].grid(True)
    
    if random_signal is not None:
        axes[2, 1].plot(t, random_signal)
        axes[2, 1].set_title('Random Test Signal', fontsize=12)
        axes[2, 1].set_xlabel('Time (s)', fontsize=10)
        axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('signals_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("信号图像已保存到 signals_plot.png")

def plot_spectrograms(am_signal, fm_signal, noisy_am, noisy_fm, fs=1000):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    f, t_spec, Sxx = spectrogram(am_signal, fs=fs, nperseg=32, noverlap=16)
    im1 = axes[0, 0].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    axes[0, 0].set_title('AM Modulated Signal Spectrogram', fontsize=12)
    axes[0, 0].set_xlabel('Time (s)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(im1, ax=axes[0, 0])
    
    f, t_spec, Sxx = spectrogram(fm_signal, fs=fs, nperseg=32, noverlap=16)
    im2 = axes[0, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    axes[0, 1].set_title('FM Modulated Signal Spectrogram', fontsize=12)
    axes[0, 1].set_xlabel('Time (s)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(im2, ax=axes[0, 1])
    
    f, t_spec, Sxx = spectrogram(noisy_am, fs=fs, nperseg=32, noverlap=16)
    im3 = axes[1, 0].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    axes[1, 0].set_title('Noisy AM Signal Spectrogram', fontsize=12)
    axes[1, 0].set_xlabel('Time (s)', fontsize=10)
    axes[1, 0].set_ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(im3, ax=axes[1, 0])
    
    f, t_spec, Sxx = spectrogram(noisy_fm, fs=fs, nperseg=32, noverlap=16)
    im4 = axes[1, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    axes[1, 1].set_title('Noisy FM Signal Spectrogram', fontsize=12)
    axes[1, 1].set_xlabel('Time (s)', fontsize=10)
    axes[1, 1].set_ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('spectrograms_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("频谱图已保存到 spectrograms_plot.png")

def plot_gaf_images(am_signal, fm_signal, noisy_am, noisy_fm):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    gaf_am = signal_to_gaf(am_signal)
    im1 = axes[0, 0].imshow(gaf_am, cmap='viridis')
    axes[0, 0].set_title('AM Modulated GAF Image', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0])
    
    gaf_fm = signal_to_gaf(fm_signal)
    im2 = axes[0, 1].imshow(gaf_fm, cmap='viridis')
    axes[0, 1].set_title('FM Modulated GAF Image', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1])
    
    gaf_noisy_am = signal_to_gaf(noisy_am)
    im3 = axes[1, 0].imshow(gaf_noisy_am, cmap='viridis')
    axes[1, 0].set_title('Noisy AM GAF Image', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0])
    
    gaf_noisy_fm = signal_to_gaf(noisy_fm)
    im4 = axes[1, 1].imshow(gaf_noisy_fm, cmap='viridis')
    axes[1, 1].set_title('Noisy FM GAF Image', fontsize=12)
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('gaf_images_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("GAF图像已保存到 gaf_images_plot.png")

def plot_demodulation(t, am_demod, fm_demod, original):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(t[:-1], am_demod)
    axes[0].set_title('AM Demodulation', fontsize=12)
    axes[0].set_xlabel('Time (s)', fontsize=10)
    axes[0].grid(True)
    
    axes[1].plot(t[:-1], fm_demod)
    axes[1].set_title('FM Demodulation', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=10)
    axes[1].grid(True)
    
    axes[2].plot(t, original)
    axes[2].set_title('Original Signal', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=10)
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('demodulation_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("解调对比图已保存到 demodulation_plot.png")

def generate_random_signal(modulation_type='random', duration=1, fs=1000):
    carrier_freq = random.uniform(50, 150)
    message_freq = random.uniform(1, 10)
    snr = random.uniform(5, 15)
    
    if modulation_type == 'random':
        modulation_type = random.choice(['AM', 'FM'])
    
    if modulation_type == 'AM':
        _, signal = am_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
    else:
        _, signal = fm_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
    
    noisy_signal = add_gaussian_noise(signal, snr)
    return noisy_signal, modulation_type

def main():
    fs = 1000
    duration = 1
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    print("=" * 60)
    print("基于2D CNN的AM/FM调制方式识别系统")
    print("=" * 60)
    
    print("\n1. 生成原始信号...")
    _, original_signal = generate_signal(freq=5, duration=duration, fs=fs)
    
    print("\n2. 生成AM/FM调制信号...")
    _, am_signal = am_modulation(carrier_freq=100, message_freq=5, duration=duration, fs=fs)
    _, fm_signal = fm_modulation(carrier_freq=100, message_freq=5, duration=duration, fs=fs)
    
    print("\n3. 添加高斯噪声 (SNR=10dB)...")
    noisy_am = add_gaussian_noise(am_signal, snr_db=10)
    noisy_fm = add_gaussian_noise(fm_signal, snr_db=10)
    
    print("\n4. 解调信号...")
    message_freq = 5
    am_demod = am_demodulation(noisy_am, fs=fs)
    fm_demod = fm_demodulation(noisy_fm, fs=fs, message_freq=message_freq)
    
    print("\n5. 生成训练数据并训练2D CNN模型...")
    if USE_CNN:
        print("使用2D CNN模型进行训练...")
        X, y, le = generate_training_data_cnn(num_samples=2000, duration=duration, fs=fs, image_type='spectrogram')
        print(f"数据形状: {X.shape}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("训练2D CNN模型...")
        model = train_cnn_model(X_train, y_train, epochs=15, batch_size=32)
        
        test_acc = evaluate_cnn_model(model, X_test, y_test)
        print(f"\n测试准确率: {test_acc:.4f}")
    else:
        print("使用SVM模型进行训练...")
        X, y = generate_training_data_svm(num_samples=2000, duration=duration, fs=fs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler, clf = build_svm_model()
        X_train_scaled = scaler.fit_transform(X_train)
        clf.fit(X_train_scaled, y_train)
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"\n测试准确率: {test_acc:.4f}")
    
    print("\n6. 生成随机测试信号...")
    random_signal, true_type = generate_random_signal(modulation_type='random')
    
    print("\n7. AI识别调制方式...")
    if USE_CNN:
        predicted_idx, confidence = predict_cnn_model(model, random_signal, image_type='spectrogram', fs=fs)
        predicted_type = le.inverse_transform([predicted_idx])[0]
    else:
        random_signal_features = extract_features(random_signal, fs=fs)
        random_signal_features_scaled = scaler.transform(random_signal_features.reshape(1, -1))
        prediction = clf.predict(random_signal_features_scaled)
        confidence = np.max(clf.predict_proba(random_signal_features_scaled))
        predicted_type = prediction[0]
    
    print("\n" + "=" * 60)
    print("识别结果 (Recognition Result)")
    print("=" * 60)
    print(f"真实调制方式 (True Modulation): {true_type}")
    print(f"AI识别结果 (AI Prediction):     {predicted_type}")
    print(f"置信度 (Confidence):            {confidence:.4f}")
    print("=" * 60)
    
    print("\n8. 绘制信号图像...")
    plot_signals(t, original_signal, am_signal, fm_signal, noisy_am, noisy_fm, random_signal)
    
    print("\n9. 绘制频谱图（2D图像）...")
    plot_spectrograms(am_signal, fm_signal, noisy_am, noisy_fm, fs=fs)
    
    print("\n10. 绘制GAF图像（2D图像）...")
    plot_gaf_images(am_signal, fm_signal, noisy_am, noisy_fm)
    
    print("\n11. 绘制解调对比图...")
    plot_demodulation(t, am_demod, fm_demod, original_signal)
    
    print("\n" + "=" * 60)
    print("所有图像已生成完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()