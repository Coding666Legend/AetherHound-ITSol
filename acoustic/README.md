# AetherHound Acoustic DSP Module

## 1. Module Overview

- **Module Name:** AetherHound Acoustic DSP Module
- **Owner:** Member 03 — DSP & Acoustic Engineer
- **Domain:** Hardware Security / Signals Intelligence (SIGINT)
- **Purpose:** Analyze acoustic emissions from computer hardware and server-room environments to detect suspicious periodic, modulated, or structured acoustic patterns that may indicate covert acoustic communication originating from air-gapped computers.

Air-gapped computers isolated from external networks may be manipulated by targeted malware to modulate physical components—such as CPU cooling fans, hard drive actuators, or internal power circuitry—to emit structured acoustic signals. The AetherHound Acoustic DSP Module provides the signal processing foundation to capture, filter, analyze, and extract features from these acoustic emissions.

---

## 2. Module Responsibilities

As defined in the team responsibility matrix, Member 03 owns:

- **Synthetic Acoustic Signal Generation:** Generating simulated server-room ambient noise, baseline fan noise, and covert structured acoustic transmissions for hardware-free development and testing.
- **Audio Acquisition & Validation:** Loading, validating, and saving standard WAV audio files, along with live microphone capture.
- **Signal Filtering & Noise Reduction:** Isolating target frequency bands and attenuating background acoustic noise.
- **Frequency-Domain Analysis:** Performing Fast Fourier Transform (FFT) computations to identify dominant tones, harmonics, and anomalous periodicities.
- **Time-Frequency Spectrogram Generation:** Generating 2D time-frequency representations (STFT spectrograms) to visualize transient signals and structured frequency shifts over time.
- **Acoustic Feature Extraction:** Computing numerical acoustic descriptors to feed into anomaly detection algorithms.

---

## 3. Module Architecture & Status

The acoustic package is structured into modular Python components:

1. **`generator.py`** *(Implemented in Commit 03)*  
   Responsible for synthetic audio generation, baseline fan acoustic profiles, and controlled synthetic acoustic anomalies (e.g., frequency-shift or amplitude-modulated covert patterns) to enable testing without physical hardware. Currently implements `generate_sine_wave()`.

2. **`audio_io.py`** *(Implemented in Commit 06)*  
   Responsible for loading, validating, and saving standard uncompressed WAV audio files with strict sample rate and bit-depth validation. Provides `read_wav()` and `write_wav()`.

3. **`recorder.py`** *(Planned)*  
   Responsible for real-time microphone recording and audio stream acquisition using PyAudio when physical microphone hardware is attached.

4. **`filters.py`** *(Implemented in Commit 07)*  
   Responsible for digital signal filtering using SciPy, including zero-phase Butterworth bandpass, lowpass, highpass filtering, and noise suppression techniques. Provides `butter_lowpass_filter()`, `butter_highpass_filter()`, and `butter_bandpass_filter()`.

5. **`fft_analysis.py`** *(Planned)*  
   Responsible for frequency-domain analysis using FFT, calculating magnitude spectra, power spectral density (PSD), and detecting dominant spectral peaks.

6. **`spectrogram.py`** *(Planned)*  
   Responsible for time-frequency analysis and Short-Time Fourier Transform (STFT) spectrogram computation, supporting both numerical matrix output and visualization.

7. **`features.py`** *(Planned)*  
   Responsible for extracting numerical acoustic features (such as spectral centroid, spectral rolloff, spectral flux, energy variance, and band ratios) for downstream machine learning analysis.

---

## 4. Synthetic Audio Generator (`generator.py`)

The `acoustic.generator` module provides offline synthetic audio generation using NumPy without hardware or network requirements.

### 4.1 `generate_sine_wave()`

Generates a continuous sinusoidal audio waveform:
$$y(t) = A \cdot \sin(2\pi f t + \phi)$$

#### Parameters

| Parameter | Type | Default | Description | Constraints |
|---|---|---|---|---|
| `frequency` | `float` | *Required* | Tone frequency in Hertz (Hz) | Must be positive ($f > 0$) and strictly below Nyquist limit ($f < f_s / 2$) |
| `duration` | `float` | *Required* | Signal length in seconds | Must be positive ($\text{duration} > 0$) |
| `sample_rate` | `int` | `44100` | Sampling frequency in Hz | Must be a positive integer ($f_s > 0$) |
| `amplitude` | `float` | `1.0` | Peak normalized amplitude | Must be within range $[0.0, 1.0]$ |
| `phase` | `float` | `0.0` | Initial phase angle in radians | Numeric value |
| `dtype` | `np.dtype` | `np.float64` | Output NumPy data type | e.g. `np.float64`, `np.float32` |

#### Return Value
Returns a 1D `numpy.ndarray` containing the computed discrete time-domain audio samples with length $N = \text{round}(\text{duration} \times \text{sample\_rate})$.

#### Validation and Error Handling
- **`TypeError`**: Raised if non-numeric inputs are provided for frequency, duration, sample rate, amplitude, or phase.
- **`ValueError`**:
  - `frequency <= 0`: Non-positive frequencies are rejected.
  - `frequency >= sample_rate / 2`: Frequencies violating the Nyquist-Shannon sampling theorem are rejected.
  - `duration <= 0` or `sample_rate <= 0`: Non-positive duration or sampling rates are rejected.
  - `amplitude < 0.0` or `amplitude > 1.0`: Out-of-bounds normalized amplitudes are rejected.

### 4.2 `generate_fan_noise()`

Synthesizes multi-harmonic rotational cooling fan tones accompanied by broadband aerodynamic turbulence noise:
$$y(t) = \sum_{k=1}^{M} w_k \cdot \sin(2\pi \cdot k f_0 \cdot t) + \alpha \cdot \mathcal{N}(0, 1)$$

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_frequency` | `float` | *Required* | Fundamental Blade Pass Frequency in Hz ($f_0 > 0$) |
| `duration` | `float` | *Required* | Signal length in seconds ($> 0$) |
| `sample_rate` | `int` | `44100` | Sampling rate in Hz |
| `harmonics` | `Sequence[float]` | `(1.0, 0.5, 0.25, 0.125)` | Relative amplitude weights of harmonic overtones |
| `amplitude` | `float` | `0.8` | Peak normalized output amplitude in range $[0.0, 1.0]$ |
| `noise_level` | `float` | `0.1` | Broadband airflow turbulence noise level $[0.0, 1.0]$ |
| `seed` | `Optional[int]` | `None` | Random seed for deterministic reproducibility |
| `dtype` | `Any` | `np.float64` | Output floating-point NumPy data type |

### 4.3 `generate_acoustic_anomaly()`

Simulates covert acoustic communication channels originating from compromised computers (e.g., Fansmitter-style malware modulating fan speeds to exfiltrate data).

#### Supported Anomaly Types
- **`frequency_shift`**: Frequency Shift Keying (FSK) where fan rotational frequency shifts by $\Delta f$ during active pulses.
- **`pulsed_burst`**: Periodic burst bursts of anomalous acoustic energy (OOK / amplitude bursts).
- **`frequency_modulation`**: Continuous sinusoidal RPM frequency modulation.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_frequency` | `float` | *Required* | Normal baseline fan frequency in Hz ($f_0 > 0$) |
| `duration` | `float` | *Required* | Total signal length in seconds |
| `sample_rate` | `int` | `44100` | Sampling rate in Hz |
| `anomaly_type` | `str` | `"frequency_shift"` | Type of covert anomaly (`frequency_shift`, `pulsed_burst`, `frequency_modulation`) |
| `shift_hz` | `float` | `40.0` | Frequency deviation during anomaly pulse |
| `pulse_duration` | `float` | `0.5` | Duration of each pulse in seconds |
| `pulse_interval` | `float` | `1.0` | Repetition period between pulses |
| `harmonics` | `Sequence[float]` | `(1.0, 0.5, 0.25)` | Harmonic overtone amplitude weights |
| `amplitude` | `float` | `0.8` | Peak normalized output amplitude |
| `noise_level` | `float` | `0.05` | Background turbulence noise level |
| `seed` | `Optional[int]` | `None` | Deterministic random seed |

### 4.4 Usage Example

```python
from acoustic import generate_fan_noise, generate_sine_wave
import numpy as np

# 1. Generate a 1-second 440 Hz test tone at 44.1 kHz
audio = generate_sine_wave(
    frequency=440.0,
    duration=1.0,
    sample_rate=44100,
    amplitude=0.8
)

# 2. Generate a 150 Hz synthetic server cooling fan hum with 4 harmonics
fan_audio = generate_fan_noise(
    base_frequency=150.0,
    duration=2.0,
    sample_rate=44100,
    harmonics=(1.0, 0.6, 0.3, 0.15),
    amplitude=0.75,
    noise_level=0.15,
    seed=42
)
```

---

## 5. Audio Input and Output (`audio_io.py`)

The `acoustic.audio_io` module handles loading, saving, and verifying uncompressed WAV audio files:

- **`write_wav(file_path, audio, sample_rate=44100, subtype='PCM_16')`**: Saves 1D or 2D audio arrays into standard 16-bit PCM or 32-bit float WAV files with automatic directory creation and range normalization.
- **`read_wav(file_path, target_dtype=np.float64, mono=True)`**: Ingests WAV files, automatically normalizes integer PCM formats (`uint8`, `int16`, `int32`) to normalized floating-point range $[-1.0, 1.0]$, and optionally downmixes multi-channel files to mono.

### 5.1 Usage Example

```python
from acoustic import read_wav, write_wav, generate_fan_noise

# Generate and save a synthetic fan recording
audio = generate_fan_noise(base_frequency=120.0, duration=2.0, sample_rate=44100)
write_wav("data/synthetic/fan_test.wav", audio, sample_rate=44100)

# Load the WAV back into normalized numpy float array
loaded_audio, sr = read_wav("data/synthetic/fan_test.wav")
print(f"Loaded {len(loaded_audio)} samples at {sr} Hz")
```

---

## 6. Digital Audio Filtering (`filters.py`)

The `acoustic.filters` module provides zero-phase digital filtering using SciPy (`scipy.signal.butter` and `scipy.signal.filtfilt`):

- **`butter_lowpass_filter(data, cutoff_freq, sample_rate=44100, order=4)`**: Attenuates high-frequency hiss and transient spikes above `cutoff_freq`.
- **`butter_highpass_filter(data, cutoff_freq, sample_rate=44100, order=4)`**: Removes DC offsets, sub-audible mechanical rumble, and low-frequency drift below `cutoff_freq`.
- **`butter_bandpass_filter(data, low_cutoff, high_cutoff, sample_rate=44100, order=4)`**: Isolates specific acoustic frequency bands of interest (such as fan blade-pass harmonics or ultrasonic communication channels).

---

## 7. Acoustic Processing Pipeline

The acoustic signal flow operates according to the following conceptual pipeline:

```
+-------------------------------------------------------------+
|                        Audio Source                         |
|      (Synthetic Generator or Physical Microphone Input)      |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                      Audio Acquisition                      |
|           (WAV File Ingestion or PyAudio Stream)            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                      Audio Validation                       |
|   (Format Verification, Sample Rate & Integrity Checking)   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                      Signal Filtering                       |
|    (Bandpass/Lowpass/Highpass Filtering & Noise Reduction)  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                        FFT Analysis                         |
|     (Frequency Spectrum, Peak Detection & Harmonics)        |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   Spectrogram Generation                    |
|        (Short-Time Fourier Transform - STFT Matrices)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Feature Extraction                      |
|       (Spectral Centroid, Energy, Rolloff, Bandwidth)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|               Future ML Integration (Member 04)             |
|          (Autoencoder Baseline & Anomaly Detection)         |
+-------------------------------------------------------------+
```

---

## 6. Technology Stack

The module relies strictly on standard local scientific Python libraries:

- **Python 3:** Core programming language ensuring cross-platform, local standalone execution.
- **NumPy:** High-performance vectorized numerical operations, multidimensional array manipulations, and buffer management.
- **SciPy:** Digital signal processing algorithms (`scipy.signal`), digital filter design, window functions, and frequency transforms.
- **PyAudio:** Low-latency hardware microphone audio capture via PortAudio bindings.
- **SoundFile:** Uncompressed WAV file I/O ensuring precise sample-rate control and lossless floating-point representations.
- **Matplotlib:** Static rendering and generation of spectrogram plots and spectral diagnostic figures.

---

## 7. Offline-First Architecture

AetherHound is designed for high-security, air-gapped environments. Consequently, the acoustic DSP module adheres strictly to an offline-first architecture:

- **Zero Cloud Dependencies:** No remote servers, external APIs, telemetry, or network-bound services are used.
- **Standalone Local Execution:** All signal generation, filtering, transformations, and feature extractions execute entirely in local memory on the host system (e.g., standard Linux workstation or Raspberry Pi).
- **No Web Frameworks:** The module is a pure Python library designed to be invoked via CLI or local background services without web servers, browsers, HTML, or CSS.

---

## 8. Future Integration

The acoustic DSP package is designed with clear modular boundaries:

- **Member 01 (System Architecture & CLI Integration):** The acoustic module exposes standardized, reusable Python interfaces (such as audio frame containers and feature frames) allowing Member 01 to cleanly integrate acoustic acquisition and analysis into the unified AetherHound CLI application.
- **Member 04 (Machine Learning & Anomaly Detection):** Extracted numerical acoustic feature vectors will adhere to stable, documented schemas consumable by Member 04's autoencoder and reconstruction-error anomaly models.
- **Member 03 Ownership:** Member 03 maintains exclusive ownership of synthetic audio generation, signal acquisition, DSP filtering, spectral transformation, and acoustic feature engineering.

---

## 9. Dependencies & Linux Installation

### 9.1 Dependency Classification

To maintain a clean offline-first footprint, dependencies are divided into **Required** (core DSP, synthetic signal generation, analysis, visualization) and **Optional** (hardware-dependent real-time microphone capture):

| Package | Status | Purpose | Cross-Project Usage |
|---|---|---|---|
| **`numpy`** | **Required** | High-performance array buffers, signal vectors, math | Shared with Member 01 & Member 02 |
| **`scipy`** | **Required** | Digital filter design (`scipy.signal`), windowing, STFT | Shared with Member 02 (RF) |
| **`soundfile`** | **Required** | Lossless uncompressed WAV file I/O & validation | Acoustic module specific |
| **`matplotlib`** | **Required** | Spectrogram generation, frequency spectrum plotting | Shared across project |
| **`pyaudio`** | **Optional** | Real-time live microphone stream acquisition | Optional (hardware capture only) |

> **Note on Hardware Independence:** Synthetic audio generation, digital filtering, FFT, spectrogram calculation, and feature extraction operate completely without `pyaudio` or physical microphone hardware.

### 9.2 Repository Dependency Strategy

To avoid duplicate configuration files and conflicting version constraints across team members, all project dependencies are managed centrally in the root [`requirements.txt`](../requirements.txt).

### 9.3 Linux Installation Guide

#### Option A: Python Virtual Environment (Recommended)
Modern Linux distributions (Debian 12+, Ubuntu 24.04+, Arch Linux) enforce [PEP 668](https://peps.python.org/pep-0668/) externally managed environments. A virtual environment is the recommended installation path:

```bash
# 1. Create and activate a local virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install core dependencies (offline synthetic testing & DSP)
pip install -r requirements.txt
```

If physical microphone recording is not required or PortAudio system libraries are unavailable, core dependencies can be installed standalone:
```bash
pip install numpy scipy soundfile matplotlib
```

#### Option B: Linux System Prerequisites (for PyAudio & SoundFile)
If compiling `pyaudio` from source or linking against native audio libraries, install the PortAudio and libsndfile development headers:

- **Debian / Ubuntu:**
  ```bash
  sudo apt update
  sudo apt install libsndfile1 portaudio19-dev python3-pyaudio
  ```

- **Arch Linux:**
  ```bash
  sudo pacman -S libsndfile portaudio
  # Or via pacman system Python packages:
  sudo pacman -S python-numpy python-scipy python-matplotlib python-soundfile python-pyaudio
  ```

- **Fedora / RHEL:**
  ```bash
  sudo dnf install libsndfile portaudio-devel
  ```

### 9.4 Offline / Air-Gapped Deployment
For air-gapped computers with no internet access:
1. On an internet-connected staging machine, download pre-built wheels:
   ```bash
   pip download -r requirements.txt -d ./vendor/
   ```
2. Transfer `./vendor/` to the air-gapped target machine via secure physical media.
3. Install wheels entirely offline:
   ```bash
   pip install --no-index --find-links=./vendor/ -r requirements.txt
   ```


