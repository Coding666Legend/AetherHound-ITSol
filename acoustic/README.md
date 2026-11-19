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

## 3. Planned Modules

The acoustic package will be developed across modular Python components in subsequent commits. 

> **Notice:** The following components represent planned modules. They will be implemented iteratively in designated commits; no placeholder or dummy code is introduced prematurely.

1. **`generator.py`**  
   Responsible for synthetic server-room audio generation, baseline fan acoustic profiles, and controlled synthetic acoustic anomalies (e.g., frequency-shift or amplitude-modulated covert patterns) to enable testing without physical hardware.

2. **`audio_io.py`**  
   Responsible for loading, validating, and saving standard uncompressed WAV audio files with strict sample rate and bit-depth validation.

3. **`recorder.py`**  
   Responsible for real-time microphone recording and audio stream acquisition using PyAudio when physical microphone hardware is attached.

4. **`filters.py`**  
   Responsible for digital signal filtering using SciPy, including Butterworth bandpass, lowpass, highpass filtering, and noise suppression techniques.

5. **`fft_analysis.py`**  
   Responsible for frequency-domain analysis using FFT, calculating magnitude spectra, power spectral density (PSD), and detecting dominant spectral peaks.

6. **`spectrogram.py`**  
   Responsible for time-frequency analysis and Short-Time Fourier Transform (STFT) spectrogram computation, supporting both numerical matrix output and visualization.

7. **`features.py`**  
   Responsible for extracting numerical acoustic features (such as spectral centroid, spectral rolloff, spectral flux, energy variance, and band ratios) for downstream machine learning analysis.

---

## 4. Acoustic Processing Pipeline

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

## 5. Technology Stack

The module relies strictly on standard local scientific Python libraries:

- **Python 3:** Core programming language ensuring cross-platform, local standalone execution.
- **NumPy:** High-performance vectorized numerical operations, multidimensional array manipulations, and buffer management.
- **SciPy:** Digital signal processing algorithms (`scipy.signal`), digital filter design, window functions, and frequency transforms.
- **PyAudio:** Low-latency hardware microphone audio capture via PortAudio bindings.
- **SoundFile:** Uncompressed WAV file I/O ensuring precise sample-rate control and lossless floating-point representations.
- **Matplotlib:** Static rendering and generation of spectrogram plots and spectral diagnostic figures.

---

## 6. Offline-First Architecture

AetherHound is designed for high-security, air-gapped environments. Consequently, the acoustic DSP module adheres strictly to an offline-first architecture:

- **Zero Cloud Dependencies:** No remote servers, external APIs, telemetry, or network-bound services are used.
- **Standalone Local Execution:** All signal generation, filtering, transformations, and feature extractions execute entirely in local memory on the host system (e.g., standard Linux workstation or Raspberry Pi).
- **No Web Frameworks:** The module is a pure Python library designed to be invoked via CLI or local background services without web servers, browsers, HTML, or CSS.

---

## 7. Future Integration

The acoustic DSP package is designed with clear modular boundaries:

- **Member 01 (System Architecture & CLI Integration):** The acoustic module exposes standardized, reusable Python interfaces (such as audio frame containers and feature frames) allowing Member 01 to cleanly integrate acoustic acquisition and analysis into the unified AetherHound CLI application.
- **Member 04 (Machine Learning & Anomaly Detection):** Extracted numerical acoustic feature vectors will adhere to stable, documented schemas consumable by Member 04's autoencoder and reconstruction-error anomaly models.
- **Member 03 Ownership:** Member 03 maintains exclusive ownership of synthetic audio generation, signal acquisition, DSP filtering, spectral transformation, and acoustic feature engineering.

---

## 8. Dependencies & Linux Installation

### 8.1 Dependency Classification

To maintain a clean offline-first footprint, dependencies are divided into **Required** (core DSP, synthetic signal generation, analysis, visualization) and **Optional** (hardware-dependent real-time microphone capture):

| Package | Status | Purpose | Cross-Project Usage |
|---|---|---|---|
| **`numpy`** | **Required** | High-performance array buffers, signal vectors, math | Shared with Member 01 & Member 02 |
| **`scipy`** | **Required** | Digital filter design (`scipy.signal`), windowing, STFT | Shared with Member 02 (RF) |
| **`soundfile`** | **Required** | Lossless uncompressed WAV file I/O & validation | Acoustic module specific |
| **`matplotlib`** | **Required** | Spectrogram generation, frequency spectrum plotting | Shared across project |
| **`pyaudio`** | **Optional** | Real-time live microphone stream acquisition | Optional (hardware capture only) |

> **Note on Hardware Independence:** Synthetic audio generation, digital filtering, FFT, spectrogram calculation, and feature extraction operate completely without `pyaudio` or physical microphone hardware.

### 8.2 Repository Dependency Strategy

To avoid duplicate configuration files and conflicting version constraints across team members, all project dependencies are managed centrally in the root [`requirements.txt`](../requirements.txt).

### 8.3 Linux Installation Guide

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

### 8.4 Offline / Air-Gapped Deployment
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


