"""Acoustic DSP processing pipeline for AetherHound.

Orchestrates acoustic signal processing:
- Input ingestion (numpy array or WAV file via audio_io) and validation
- Optional zero-phase digital filtering (low-pass, high-pass, band-pass)
- Frequency-domain spectral analysis (FFT, peak detection, harmonic analysis)
- Time-frequency spectrogram generation (STFT matrices)
- Acoustic feature extraction and FeatureFrame formatting

Owner: Member 03 – DSP & Acoustic Engineer
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from acoustic.audio_io import read_wav
from acoustic.features import (
    FeatureFrame,
    create_feature_frame,
    extract_acoustic_features,
)
from acoustic.fft_analysis import (
    analyze_harmonics,
    compute_fft,
    detect_spectral_peaks,
)
from acoustic.filters import (
    butter_bandpass_filter,
    butter_highpass_filter,
    butter_lowpass_filter,
)
from acoustic.spectrogram import compute_spectrogram


@dataclass
class FilterConfig:
    """Configuration for optional digital filtering stage.

    Attributes
    ----------
    filter_type : str
        Type of filter to apply: 'lowpass', 'highpass', or 'bandpass'.
    cutoff_freq : Optional[float]
        Cutoff frequency in Hz for 'lowpass' or 'highpass'.
    low_cutoff : Optional[float]
        Lower cutoff frequency in Hz for 'bandpass'.
    high_cutoff : Optional[float]
        Upper cutoff frequency in Hz for 'bandpass'.
    order : int
        Filter order (default 4, range [1, 12]).
    enabled : bool
        Whether this filtering stage is active (default True).
    """

    filter_type: str = "bandpass"
    cutoff_freq: Optional[float] = None
    low_cutoff: Optional[float] = None
    high_cutoff: Optional[float] = None
    order: int = 4
    enabled: bool = True

    def validate(self, sample_rate: int) -> None:
        """Validate filter parameters against sample rate and filter type."""
        if not isinstance(self.enabled, bool):
            raise TypeError(f"FilterConfig.enabled must be a bool, got {type(self.enabled).__name__}.")

        if not self.enabled:
            return

        valid_types = ("lowpass", "highpass", "bandpass")
        if not isinstance(self.filter_type, str) or self.filter_type not in valid_types:
            raise ValueError(
                f"FilterConfig.filter_type must be one of {valid_types}, got '{self.filter_type}'."
            )

        if isinstance(self.order, bool) or not isinstance(self.order, (int, np.integer)):
            raise TypeError(f"FilterConfig.order must be an integer, got {type(self.order).__name__}.")
        if self.order < 1 or self.order > 12:
            raise ValueError(f"FilterConfig.order must be in range [1, 12], got {self.order}.")

        nyquist = float(sample_rate) / 2.0

        if self.filter_type in ("lowpass", "highpass"):
            if self.cutoff_freq is None:
                raise ValueError(f"FilterConfig.cutoff_freq is required for '{self.filter_type}'.")
            if isinstance(self.cutoff_freq, bool) or not isinstance(self.cutoff_freq, (int, float)):
                raise TypeError(
                    f"FilterConfig.cutoff_freq must be numeric, got {type(self.cutoff_freq).__name__}."
                )
            if not math.isfinite(self.cutoff_freq) or self.cutoff_freq <= 0.0 or self.cutoff_freq >= nyquist:
                raise ValueError(
                    f"FilterConfig.cutoff_freq ({self.cutoff_freq} Hz) must be strictly between 0 and Nyquist ({nyquist} Hz)."
                )

        elif self.filter_type == "bandpass":
            if self.low_cutoff is None or self.high_cutoff is None:
                raise ValueError(
                    "FilterConfig requires both low_cutoff and high_cutoff for 'bandpass'."
                )
            for name, val in [("low_cutoff", self.low_cutoff), ("high_cutoff", self.high_cutoff)]:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(f"FilterConfig.{name} must be numeric, got {type(val).__name__}.")
                if not math.isfinite(val):
                    raise ValueError(f"FilterConfig.{name} must be a finite number.")

            if self.low_cutoff <= 0.0:
                raise ValueError(f"FilterConfig.low_cutoff must be positive, got {self.low_cutoff} Hz.")
            if self.high_cutoff >= nyquist:
                raise ValueError(
                    f"FilterConfig.high_cutoff ({self.high_cutoff} Hz) must be strictly below Nyquist ({nyquist} Hz)."
                )
            if self.low_cutoff >= self.high_cutoff:
                raise ValueError(
                    f"FilterConfig.low_cutoff ({self.low_cutoff} Hz) must be strictly less than high_cutoff ({self.high_cutoff} Hz)."
                )


@dataclass
class SpectrogramConfig:
    """Configuration for time-frequency spectrogram analysis.

    Attributes
    ----------
    nperseg : int
        Length of each FFT segment (default 1024).
    noverlap : Optional[int]
        Number of overlapping points between segments (default None -> nperseg // 2).
    window : str
        Window function to apply (default 'hann').
    """

    nperseg: int = 1024
    noverlap: Optional[int] = None
    window: str = "hann"

    def validate(self) -> None:
        """Validate spectrogram configuration parameters."""
        if isinstance(self.nperseg, bool) or not isinstance(self.nperseg, (int, np.integer)):
            raise TypeError(f"SpectrogramConfig.nperseg must be an integer, got {type(self.nperseg).__name__}.")
        if self.nperseg <= 0:
            raise ValueError(f"SpectrogramConfig.nperseg must be positive, got {self.nperseg}.")

        if self.noverlap is not None:
            if isinstance(self.noverlap, bool) or not isinstance(self.noverlap, (int, np.integer)):
                raise TypeError(
                    f"SpectrogramConfig.noverlap must be an integer, got {type(self.noverlap).__name__}."
                )
            if self.noverlap < 0 or self.noverlap >= self.nperseg:
                raise ValueError(
                    f"SpectrogramConfig.noverlap must satisfy 0 <= noverlap < nperseg ({self.nperseg}), got {self.noverlap}."
                )

        valid_windows = ("hann", "hamming", "blackman", "boxcar")
        if not isinstance(self.window, str) or self.window not in valid_windows:
            raise ValueError(
                f"SpectrogramConfig.window must be one of {valid_windows}, got '{self.window}'."
            )


@dataclass
class ProcessingResult:
    """Structured container holding all acoustic pipeline outputs.

    Attributes
    ----------
    raw_signal : np.ndarray
        Input signal (normalized float array).
    processed_signal : np.ndarray
        Signal after optional filtering (or raw signal if filtering was disabled).
    sample_rate : int
        Audio sampling rate in Hz.
    source_id : str
        Source identifier.
    timestamp : datetime
        Processing timestamp in UTC.
    filtered : bool
        Whether a filter was applied.
    filter_type : Optional[str]
        Filter type applied ('lowpass', 'highpass', 'bandpass', or None).
    frequencies : np.ndarray
        FFT frequency bins (0 to Nyquist) in Hz.
    magnitudes : np.ndarray
        FFT normalized magnitude spectrum.
    spectral_peaks : List[Tuple[float, float]]
        Detected spectral peaks sorted descending by magnitude: [(freq_hz, magnitude), ...].
    harmonics : Optional[Dict[int, Optional[float]]]
        Harmonic overtone mapping if base_frequency was specified, otherwise None.
    spectrogram_freqs : np.ndarray
        Spectrogram frequency axis in Hz.
    spectrogram_times : np.ndarray
        Spectrogram time axis in seconds.
    spectrogram_matrix : np.ndarray
        2D STFT spectrogram magnitude matrix Sxx.
    features : Dict[str, float]
        Dictionary of extracted acoustic scalar descriptors.
    feature_frame : FeatureFrame
        Standardized FeatureFrame object ready for downstream anomaly detection.
    """

    raw_signal: np.ndarray
    processed_signal: np.ndarray
    sample_rate: int
    source_id: str
    timestamp: datetime
    filtered: bool
    filter_type: Optional[str]
    frequencies: np.ndarray
    magnitudes: np.ndarray
    spectral_peaks: List[Tuple[float, float]]
    harmonics: Optional[Dict[int, Optional[float]]]
    spectrogram_freqs: np.ndarray
    spectrogram_times: np.ndarray
    spectrogram_matrix: np.ndarray
    features: Dict[str, float]
    feature_frame: FeatureFrame

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a lightweight serializable summary of the processing results."""
        return {
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat(),
            "sample_rate": self.sample_rate,
            "signal_length": len(self.processed_signal),
            "duration_s": float(len(self.processed_signal) / self.sample_rate),
            "filtered": self.filtered,
            "filter_type": self.filter_type,
            "dominant_frequency_hz": float(self.features.get("dominant_frequency", 0.0)),
            "rms_energy": float(self.features.get("rms_energy", 0.0)),
            "peak_count": len(self.spectral_peaks),
            "spectrogram_shape": list(self.spectrogram_matrix.shape),
            "features": dict(self.features),
        }


class AcousticPipeline:
    """Lightweight reusable DSP processing pipeline for acoustic signals in AetherHound.

    Integrates input validation, optional filtering, FFT frequency analysis,
    STFT spectrogram generation, and acoustic feature extraction into a unified workflow.

    Parameters
    ----------
    sample_rate : int, optional
        Default sampling frequency in Hz (default 44100).
    filter_config : Optional[FilterConfig], optional
        Digital filtering configuration. If None or enabled=False, filtering is bypassed.
    spectrogram_config : Optional[SpectrogramConfig], optional
        STFT spectrogram configuration. If None, default SpectrogramConfig is used.
    fft_window : str, optional
        Windowing function for FFT spectral analysis (default 'hann').
    max_peaks : int, optional
        Maximum number of spectral peaks to detect (default 10).
    peak_threshold_rel : float, optional
        Relative threshold for peak detection in range [0.0, 1.0] (default 0.05).
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        filter_config: Optional[FilterConfig] = None,
        spectrogram_config: Optional[SpectrogramConfig] = None,
        fft_window: str = "hann",
        max_peaks: int = 10,
        peak_threshold_rel: float = 0.05,
    ) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
            raise TypeError(f"sample_rate must be an integer, got {type(sample_rate).__name__}.")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate} Hz.")
        self.sample_rate = int(sample_rate)

        if filter_config is not None:
            if not isinstance(filter_config, FilterConfig):
                raise TypeError(f"filter_config must be a FilterConfig instance, got {type(filter_config).__name__}.")
            filter_config.validate(self.sample_rate)
        self.filter_config = filter_config

        if spectrogram_config is None:
            spectrogram_config = SpectrogramConfig()
        elif not isinstance(spectrogram_config, SpectrogramConfig):
            raise TypeError(
                f"spectrogram_config must be a SpectrogramConfig instance, got {type(spectrogram_config).__name__}."
            )
        spectrogram_config.validate()
        self.spectrogram_config = spectrogram_config

        valid_windows = ("hann", "hamming", "blackman", "boxcar")
        if not isinstance(fft_window, str) or fft_window not in valid_windows:
            raise ValueError(f"fft_window must be one of {valid_windows}, got '{fft_window}'.")
        self.fft_window = fft_window

        if isinstance(max_peaks, bool) or not isinstance(max_peaks, (int, np.integer)):
            raise TypeError(f"max_peaks must be an integer, got {type(max_peaks).__name__}.")
        if max_peaks < 1:
            raise ValueError(f"max_peaks must be at least 1, got {max_peaks}.")
        self.max_peaks = int(max_peaks)

        if isinstance(peak_threshold_rel, bool) or not isinstance(peak_threshold_rel, (int, float)):
            raise TypeError(f"peak_threshold_rel must be numeric, got {type(peak_threshold_rel).__name__}.")
        if not math.isfinite(peak_threshold_rel) or peak_threshold_rel < 0.0 or peak_threshold_rel > 1.0:
            raise ValueError(f"peak_threshold_rel must be in range [0.0, 1.0], got {peak_threshold_rel}.")
        self.peak_threshold_rel = float(peak_threshold_rel)

    def _sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Validate input audio array and ensure 1D float64 representation."""
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Audio input must be a NumPy ndarray, got {type(audio).__name__}.")

        if not np.issubdtype(audio.dtype, np.floating):
            raise ValueError(f"Audio input must have a floating-point dtype, got {audio.dtype}.")

        if audio.ndim == 2:
            # Multi-channel array: average channels to mono
            audio = np.mean(audio, axis=-1)
        elif audio.ndim != 1:
            raise ValueError(f"Audio array must be 1D or 2D, got shape {audio.shape}.")

        if audio.size == 0:
            raise ValueError("Audio array must not be empty.")

        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio array contains non-finite values (NaN or Inf).")

        return audio.astype(np.float64)

    def _apply_filter(self, signal_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, bool, Optional[str]]:
        """Apply configured digital filter if enabled."""
        if self.filter_config is None or not self.filter_config.enabled:
            return signal_data, False, None

        cfg = self.filter_config
        ftype = cfg.filter_type

        if ftype == "lowpass":
            filtered = butter_lowpass_filter(
                signal_data,
                cutoff_freq=cfg.cutoff_freq,  # type: ignore[arg-type]
                sample_rate=sample_rate,
                order=cfg.order,
            )
        elif ftype == "highpass":
            filtered = butter_highpass_filter(
                signal_data,
                cutoff_freq=cfg.cutoff_freq,  # type: ignore[arg-type]
                sample_rate=sample_rate,
                order=cfg.order,
            )
        elif ftype == "bandpass":
            filtered = butter_bandpass_filter(
                signal_data,
                low_cutoff=cfg.low_cutoff,  # type: ignore[arg-type]
                high_cutoff=cfg.high_cutoff,  # type: ignore[arg-type]
                sample_rate=sample_rate,
                order=cfg.order,
            )
        else:
            raise ValueError(f"Unknown filter type '{ftype}'.")

        return filtered, True, ftype

    def process(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        source_id: str = "acoustic_stream",
        timestamp: Optional[datetime] = None,
        base_frequency: Optional[float] = None,
    ) -> ProcessingResult:
        """Process an in-memory NumPy audio signal through the acoustic DSP pipeline.

        Parameters
        ----------
        audio : np.ndarray
            1D or 2D floating-point NumPy array of audio samples.
        sample_rate : Optional[int], optional
            Sampling rate in Hz. If None, uses pipeline's default sample_rate.
        source_id : str, optional
            Source device or stream identifier (default 'acoustic_stream').
        timestamp : Optional[datetime], optional
            Capture timestamp (default UTC now).
        base_frequency : Optional[float], optional
            Fundamental frequency in Hz to analyze harmonic overtones. If None, harmonic analysis is skipped.

        Returns
        -------
        ProcessingResult
            Structured container with processed signals, FFT, spectrogram, and features.
        """
        sr = self.sample_rate if sample_rate is None else int(sample_rate)
        if sr <= 0:
            raise ValueError(f"sample_rate must be positive, got {sr} Hz.")

        if not isinstance(source_id, str) or len(source_id.strip()) == 0:
            raise ValueError(f"source_id must be a non-empty string, got '{source_id}'.")

        if timestamp is None:
            ts = datetime.now(timezone.utc)
        else:
            if not isinstance(timestamp, datetime):
                raise TypeError(f"timestamp must be a datetime object, got {type(timestamp).__name__}.")
            ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)

        # 1. Input sanitization
        clean_raw = self._sanitize_audio(audio)

        # Validate minimum length for spectrogram
        if len(clean_raw) < self.spectrogram_config.nperseg:
            raise ValueError(
                f"Audio length ({len(clean_raw)} samples) is shorter than spectrogram nperseg "
                f"({self.spectrogram_config.nperseg}). Provide at least {self.spectrogram_config.nperseg} samples."
            )

        # 2. Digital filtering
        processed, was_filtered, ftype = self._apply_filter(clean_raw, sr)

        # 3. FFT Frequency analysis
        freqs, mags = compute_fft(processed, sample_rate=sr, window=self.fft_window)
        peaks = detect_spectral_peaks(
            freqs,
            mags,
            threshold_rel=self.peak_threshold_rel,
            max_peaks=self.max_peaks,
        )

        harmonics: Optional[Dict[int, Optional[float]]] = None
        if base_frequency is not None:
            if isinstance(base_frequency, bool) or not isinstance(base_frequency, (int, float)):
                raise TypeError(f"base_frequency must be numeric, got {type(base_frequency).__name__}.")
            if not math.isfinite(base_frequency) or base_frequency <= 0:
                raise ValueError(f"base_frequency must be positive and finite, got {base_frequency}.")
            peak_freqs = [f for f, _ in peaks]
            harmonics = analyze_harmonics(float(base_frequency), peak_freqs)

        # 4. Spectrogram generation
        spec_freqs, spec_times, sxx = compute_spectrogram(
            processed,
            sample_rate=sr,
            nperseg=self.spectrogram_config.nperseg,
            noverlap=self.spectrogram_config.noverlap,
            window=self.spectrogram_config.window,
        )

        # 5. Acoustic feature extraction
        features = extract_acoustic_features(processed, sample_rate=sr)

        # 6. FeatureFrame contract
        frame = create_feature_frame(
            processed,
            sample_rate=sr,
            source_id=source_id.strip(),
            timestamp=ts,
        )

        return ProcessingResult(
            raw_signal=clean_raw,
            processed_signal=processed,
            sample_rate=sr,
            source_id=source_id.strip(),
            timestamp=ts,
            filtered=was_filtered,
            filter_type=ftype,
            frequencies=freqs,
            magnitudes=mags,
            spectral_peaks=peaks,
            harmonics=harmonics,
            spectrogram_freqs=spec_freqs,
            spectrogram_times=spec_times,
            spectrogram_matrix=sxx,
            features=features,
            feature_frame=frame,
        )

    def process_file(
        self,
        file_path: Union[str, Path],
        source_id: Optional[str] = None,
        base_frequency: Optional[float] = None,
    ) -> ProcessingResult:
        """Load a WAV audio file from disk and execute the acoustic DSP pipeline.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the WAV audio file.
        source_id : Optional[str], optional
            Source identifier. If None, the filename stem is used.
        base_frequency : Optional[float], optional
            Target base frequency for harmonic overtone analysis.

        Returns
        -------
        ProcessingResult
            Structured result containing processed audio, spectrum, spectrogram, and features.
        """
        path_obj = Path(file_path)
        audio_data, file_sr = read_wav(path_obj, mono=True)
        sid = path_obj.stem if source_id is None else source_id

        return self.process(
            audio=audio_data,
            sample_rate=file_sr,
            source_id=sid,
            base_frequency=base_frequency,
        )
