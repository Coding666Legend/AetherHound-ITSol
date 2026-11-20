"""Time-frequency spectrogram analysis and visualization module for AetherHound.

Computes Short-Time Fourier Transform (STFT) spectrogram matrices using SciPy and
exports static diagnostic spectrogram figures using Matplotlib in headless/offline
mode for inspecting acoustic emissions and transient covert channels.

Owner: Member 03 – DSP & Acoustic Engineer
"""

import math
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal

# Use non-interactive Agg backend to ensure offline, headless operation without X11
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int = 44100,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Short-Time Fourier Transform (STFT) spectrogram of audio data.

    Parameters
    ----------
    data : np.ndarray
        1D array of audio samples.
    sample_rate : int, optional
        Sampling frequency in Hz. Default 44100.
    nperseg : int, optional
        Length of each FFT segment. Default 1024.
    noverlap : Optional[int], optional
        Number of overlapping points between segments. If None, defaults to nperseg // 2.
    window : str, optional
        Desired window function (e.g. 'hann', 'hamming', 'blackman'). Default 'hann'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (frequencies, times, Sxx) where:
        - frequencies: 1D array of sample frequencies (Hz).
        - times: 1D array of segment times (seconds).
        - Sxx: 2D spectrogram magnitude matrix of shape (len(frequencies), len(times)).

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If parameters are invalid, non-finite, or if data is shorter than nperseg.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a NumPy ndarray, got {type(data).__name__}.")

    if data.ndim != 1:
        raise ValueError(f"data must be a 1D array, got {data.ndim}D.")

    if data.size == 0:
        raise ValueError("data array must not be empty.")

    if not np.all(np.isfinite(data)):
        raise ValueError("data array contains non-finite values (NaN or Inf).")

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError(f"sample_rate must be an integer, got {type(sample_rate).__name__}.")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate} Hz.")

    if isinstance(nperseg, bool) or not isinstance(nperseg, (int, np.integer)):
        raise TypeError(f"nperseg must be an integer, got {type(nperseg).__name__}.")

    if nperseg <= 0:
        raise ValueError(f"nperseg must be positive, got {nperseg}.")

    if len(data) < nperseg:
        raise ValueError(f"data length ({len(data)}) must be at least nperseg ({nperseg}).")

    if noverlap is None:
        noverlap_int = nperseg // 2
    else:
        if isinstance(noverlap, bool) or not isinstance(noverlap, (int, np.integer)):
            raise TypeError(f"noverlap must be an integer, got {type(noverlap).__name__}.")
        if noverlap < 0 or noverlap >= nperseg:
            raise ValueError(f"noverlap must satisfy 0 <= noverlap < nperseg ({nperseg}), got {noverlap}.")
        noverlap_int = int(noverlap)

    valid_windows = ("hann", "hamming", "blackman", "boxcar")
    if window not in valid_windows:
        raise ValueError(f"Unsupported window '{window}'. Must be one of {valid_windows}.")

    frequencies, times, Sxx = signal.spectrogram(
        data,
        fs=int(sample_rate),
        window=window,
        nperseg=int(nperseg),
        noverlap=noverlap_int,
        scaling="density",
        mode="magnitude",
    )

    return frequencies, times, Sxx


def export_spectrogram_plot(
    frequencies: np.ndarray,
    times: np.ndarray,
    Sxx: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Acoustic Spectrogram",
    cmap: str = "viridis",
    db_scale: bool = True,
) -> None:
    """Render and save an offline static spectrogram figure to disk.

    Parameters
    ----------
    frequencies : np.ndarray
        1D array of frequency bins in Hz.
    times : np.ndarray
        1D array of time bin centers in seconds.
    Sxx : np.ndarray
        2D spectrogram matrix of shape (len(frequencies), len(times)).
    output_path : Union[str, Path]
        Destination image file path (e.g., .png or .jpg).
    title : str, optional
        Plot title. Default 'Acoustic Spectrogram'.
    cmap : str, optional
        Matplotlib colormap name. Default 'viridis'.
    db_scale : bool, optional
        If True, renders magnitudes on a logarithmic decibel (dB) scale. Default True.
    """
    if not isinstance(output_path, (str, Path)):
        raise TypeError(f"output_path must be a string or Path, got {type(output_path).__name__}.")
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if Sxx.ndim != 2:
        raise ValueError(f"Sxx must be a 2D matrix, got shape {Sxx.shape}.")

    if Sxx.shape[0] != len(frequencies) or Sxx.shape[1] != len(times):
        raise ValueError(
            f"Sxx shape {Sxx.shape} must match (len(frequencies), len(times)) = "
            f"({len(frequencies)}, {len(times)})."
        )

    # Compute intensity values
    if db_scale:
        eps = 1e-12
        display_data = 20.0 * np.log10(np.maximum(Sxx, eps))
        cb_label = "Magnitude (dB)"
    else:
        display_data = Sxx
        cb_label = "Linear Magnitude"

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(times, frequencies, display_data, shading="auto", cmap=cmap)
    fig.colorbar(mesh, ax=ax, label=cb_label)

    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim([0, frequencies[-1]])

    fig.tight_layout()
    fig.savefig(str(path_obj), dpi=150)
    plt.close(fig)
