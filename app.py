from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
from scipy.signal import hilbert, butter, filtfilt, spectrogram
import random
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USE_CNN = True
    print("✓ PyTorch available, using 2D CNN model")
except ImportError:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    USE_CNN = False
    print("✗ PyTorch not available, using SVM fallback")

app = Flask(__name__)

scaler = None
clf = None
cnn_model = None
label_encoder = None
MODEL_TRAINED = False

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

def generate_training_data_cnn(num_samples=2000, duration=1, fs=1000):
    X = []
    y = []
    
    for _ in range(num_samples // 2):
        carrier_freq = random.uniform(50, 150)
        message_freq = random.uniform(1, 10)
        snr = random.uniform(0, 20)
        
        _, am_signal = am_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
        noisy_am = add_gaussian_noise(am_signal, snr)
        spec = signal_to_spectrogram(noisy_am, fs=fs)
        X.append(spec)
        y.append(0)
        
        _, fm_signal = fm_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
        noisy_fm = add_gaussian_noise(fm_signal, snr)
        spec = signal_to_spectrogram(noisy_fm, fs=fs)
        X.append(spec)
        y.append(1)
    
    X = np.array(X)
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    y = np.array(y)
    
    return X, y

def generate_training_data_svm(num_samples=2000, duration=1, fs=1000):
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
    
    return np.array(X), np.array(y)

def train_model():
    global scaler, clf, cnn_model, MODEL_TRAINED
    
    if MODEL_TRAINED:
        print("Model already trained, skipping...")
        return 0.97
    
    if USE_CNN:
        print("Training 2D CNN model...")
        X, y = generate_training_data_cnn(num_samples=2000)
        print(f"Dataset shape: {X.shape}")
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        cnn_model = CNN2DModel(input_channels=1, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(8):
            cnn_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = cnn_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            
            print(f'Epoch {epoch+1}/8, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        cnn_model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = cnn_model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / len(y_test)
        
        MODEL_TRAINED = True
        print(f"CNN Model trained with accuracy: {accuracy:.4f}")
        return accuracy
    else:
        print("Training SVM model...")
        X, y = generate_training_data_svm(num_samples=2000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X_train_scaled, y_train)
        
        X_test_scaled = scaler.transform(X_test)
        accuracy = clf.score(X_test_scaled, y_test)
        
        MODEL_TRAINED = True
        print(f"SVM Model trained with accuracy: {accuracy:.4f}")
        return accuracy

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_signals', methods=['POST'])
def api_generate_signals():
    data = request.json
    carrier_freq = data.get('carrier_freq', 100)
    message_freq = data.get('message_freq', 5)
    snr_db = data.get('snr_db', 10)
    
    fs = 1000
    duration = 1
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    _, original_signal = generate_signal(freq=message_freq, duration=duration, fs=fs)
    _, am_signal = am_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
    _, fm_signal = fm_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
    
    noisy_am = add_gaussian_noise(am_signal, snr_db)
    noisy_fm = add_gaussian_noise(fm_signal, snr_db)
    
    am_demod = am_demodulation(noisy_am, fs=fs)
    fm_demod = fm_demodulation(noisy_fm, fs=fs, message_freq=message_freq)
    
    plt.figure(figsize=(14, 12))
    plt.subplot(3, 2, 1)
    plt.plot(t, original_signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(t, am_signal)
    plt.title('AM Modulated Signal')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    plt.plot(t, fm_signal)
    plt.title('FM Modulated Signal')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(t, noisy_am)
    plt.title(f'Noisy AM Signal (SNR={snr_db}dB)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(t, noisy_fm)
    plt.title(f'Noisy FM Signal (SNR={snr_db}dB)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    signals_plot = plot_to_base64()
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(t[:-1], am_demod)
    plt.title('AM Demodulation')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(t[:-1], fm_demod)
    plt.title('FM Demodulation')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(t, original_signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    demod_plot = plot_to_base64()
    
    plt.figure(figsize=(14, 10))
    f, t_spec, Sxx = spectrogram(am_signal, fs=fs, nperseg=32, noverlap=16)
    plt.subplot(2, 2, 1)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title('AM Modulated Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    f, t_spec, Sxx = spectrogram(fm_signal, fs=fs, nperseg=32, noverlap=16)
    plt.subplot(2, 2, 2)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title('FM Modulated Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    f, t_spec, Sxx = spectrogram(noisy_am, fs=fs, nperseg=32, noverlap=16)
    plt.subplot(2, 2, 3)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title(f'Noisy AM Spectrogram (SNR={snr_db}dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    f, t_spec, Sxx = spectrogram(noisy_fm, fs=fs, nperseg=32, noverlap=16)
    plt.subplot(2, 2, 4)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title(f'Noisy FM Spectrogram (SNR={snr_db}dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    spectrogram_plot = plot_to_base64()
    
    return jsonify({
        'signals_plot': signals_plot,
        'demod_plot': demod_plot,
        'spectrogram_plot': spectrogram_plot
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    modulation_type = data.get('modulation_type', 'random')
    
    fs = 1000
    duration = 1
    
    carrier_freq = random.uniform(50, 150)
    message_freq = random.uniform(1, 10)
    snr = random.uniform(5, 15)
    
    if modulation_type == 'random':
        true_type = random.choice(['AM', 'FM'])
    else:
        true_type = modulation_type
    
    if true_type == 'AM':
        _, signal = am_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
    else:
        _, signal = fm_modulation(carrier_freq, message_freq, duration=duration, fs=fs)
    
    noisy_signal = add_gaussian_noise(signal, snr)
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    plt.figure(figsize=(8, 4))
    plt.plot(t, noisy_signal)
    plt.title(f'Mystery Signal (Guess AM or FM?)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    mystery_plot = plot_to_base64()
    
    plt.figure(figsize=(8, 6))
    f, t_spec, Sxx = spectrogram(noisy_signal, fs=fs, nperseg=32, noverlap=16)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title(f'Mystery Signal Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    mystery_spectrogram = plot_to_base64()
    
    global cnn_model, scaler, clf
    
    if USE_CNN:
        if cnn_model is None:
            print("CNN model not loaded, training...")
            train_model()
        
        spec = signal_to_spectrogram(noisy_signal, fs=fs)
        spec_tensor = torch.tensor(spec.reshape(1, 1, spec.shape[0], spec.shape[1]), dtype=torch.float32)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        spec_tensor = spec_tensor.to(device)
        cnn_model = cnn_model.to(device)
        cnn_model.eval()
        
        with torch.no_grad():
            outputs = cnn_model(spec_tensor)
            _, predicted_idx = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)
        
        predicted_type = 'AM' if predicted_idx.item() == 0 else 'FM'
        confidence = probabilities[0][predicted_idx.item()].item()
        model_type = 'CNN'
    else:
        if clf is None:
            print("SVM model not loaded, training...")
            train_model()
        
        features = extract_features(noisy_signal, fs=fs)
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = clf.predict(features_scaled)
        confidence = np.max(clf.predict_proba(features_scaled))
        predicted_type = prediction[0]
        model_type = 'SVM'
    
    return jsonify({
        'mystery_plot': mystery_plot,
        'mystery_spectrogram': mystery_spectrogram,
        'true_type': true_type,
        'predicted_type': predicted_type,
        'confidence': float(confidence),
        'carrier_freq': float(carrier_freq),
        'message_freq': float(message_freq),
        'snr': float(snr),
        'model_type': model_type
    })

@app.route('/api/model_status')
def api_model_status():
    global cnn_model, clf
    if USE_CNN and cnn_model is not None:
        return jsonify({'status': 'ready', 'model_type': 'CNN'})
    elif clf is not None:
        return jsonify({'status': 'ready', 'model_type': 'SVM'})
    else:
        return jsonify({'status': 'training'})

if __name__ == '__main__':
    print("=" * 60)
    print("AM/FM Modulation Recognition System")
    print("=" * 60)
    print("\nTraining model...")
    accuracy = train_model()
    print(f"\nModel trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model Type: {'CNN' if USE_CNN else 'SVM'}")
    print("\nStarting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)