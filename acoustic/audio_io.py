"""Audio I/O module for loading and saving WAV files in AetherHound.

Provides functions to read and write uncompressed WAV audio files with sample rate
verification, bit-depth normalization, and channel validation.

Owner: Member 03 – DSP & Acoustic Engineer
"""

import math
import os
from pathlib import Path
from typing import Any, Tuple, Union
import numpy as np
import scipy.io.wavfile as wavfile


def write_wav(
    file_path: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int = 44100,
    subtype: str = "PCM_16",
) -> None:
    """Save an audio array to a standard WAV audio file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Target file path for the output WAV file.
    audio : np.ndarray
        Audio samples. Can be 1D (mono) or 2D (samples, channels).
        Must be a floating-point array with values normalized in range [-1.0, 1.0].
    sample_rate : int, optional
        Sampling frequency in Hz. Must be a positive integer. Default 44100.
    subtype : str, optional
        WAV encoding format: 'PCM_16' (16-bit integer PCM) or 'FLOAT_32' (32-bit float).
        Default is 'PCM_16'.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If sample_rate <= 0, audio contains non-finite values, audio is not 1D or 2D,
        or subtype is unsupported.
    """
    if not isinstance(file_path, (str, Path)):
        raise TypeError(f"file_path must be a string or Path, got {type(file_path).__name__}.")
    path_str = str(file_path).strip()
    if len(path_str) == 0:
        raise ValueError("file_path must not be empty.")
    path_obj = Path(path_str)

    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be a NumPy ndarray, got {type(audio).__name__}.")

    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError(f"audio must have a floating-point dtype, got {audio.dtype}.")

    if audio.ndim not in (1, 2):
        raise ValueError(f"audio array must be 1D or 2D, got shape {audio.shape}.")

    if audio.size == 0:
        raise ValueError("audio array must not be empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("audio array contains non-finite values (NaN or Inf).")

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError(f"sample_rate must be an integer, got {type(sample_rate).__name__}.")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate} Hz.")

    valid_subtypes = ("PCM_16", "FLOAT_32")
    if subtype not in valid_subtypes:
        raise ValueError(f"Unsupported subtype '{subtype}'. Must be one of {valid_subtypes}.")

    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Format audio according to requested subtype
    if subtype == "PCM_16":
        # Clip to [-1.0, 1.0] and scale to signed 16-bit integers
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16_data = np.round(clipped * 32767.0).astype(np.int16)
        wavfile.write(str(path_obj), int(sample_rate), pcm16_data)
    elif subtype == "FLOAT_32":
        float32_data = audio.astype(np.float32)
        wavfile.write(str(path_obj), int(sample_rate), float32_data)


def read_wav(
    file_path: Union[str, Path],
    target_dtype: Any = np.float64,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Read a WAV audio file and return normalized audio samples and sample rate.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the WAV audio file to read.
    target_dtype : Any, optional
        NumPy floating-point dtype for output samples (default np.float64).
    mono : bool, optional
        If True, downmixes multi-channel audio to mono by averaging channels.
        Default is True.

    Returns
    -------
    Tuple[np.ndarray, int]
        Tuple of (audio_samples, sample_rate). Audio samples are normalized
        to range [-1.0, 1.0].

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    TypeError
        If file_path is invalid or target_dtype is not a valid floating-point dtype.
    ValueError
        If the file is not a valid WAV or has invalid parameters.
    """
    if not isinstance(file_path, (str, Path)):
        raise TypeError(f"file_path must be a string or Path, got {type(file_path).__name__}.")
    path_str = str(file_path).strip()
    if len(path_str) == 0:
        raise ValueError("file_path must not be empty.")
    path_obj = Path(path_str)

    if not path_obj.is_file():
        raise FileNotFoundError(f"WAV file not found at: {path_obj}")

    try:
        out_dtype = np.dtype(target_dtype)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Invalid NumPy dtype: {target_dtype}.") from err

    if not np.issubdtype(out_dtype, np.floating):
        raise ValueError(f"target_dtype must be floating-point, got {out_dtype}.")

    sr, raw_data = wavfile.read(str(path_obj))

    # Normalize samples to [-1.0, 1.0] based on source dtype
    if np.issubdtype(raw_data.dtype, np.integer):
        info = np.iinfo(raw_data.dtype)
        if raw_data.dtype == np.uint8:
            # 8-bit unsigned PCM: [0, 255], center at 128
            normalized = (raw_data.astype(np.float64) - 128.0) / 128.0
        else:
            # Signed PCM (int16, int32)
            max_val = max(abs(info.min), abs(info.max))
            normalized = raw_data.astype(np.float64) / max_val
    elif np.issubdtype(raw_data.dtype, np.floating):
        normalized = raw_data.astype(np.float64)
    else:
        raise ValueError(f"Unsupported WAV sample format: {raw_data.dtype}")

    # Convert to mono if requested
    if mono and normalized.ndim > 1:
        normalized = np.mean(normalized, axis=-1)

    return normalized.astype(out_dtype), int(sr)
