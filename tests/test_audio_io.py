"""Unit tests for the audio I/O module in AetherHound."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from acoustic.audio_io import read_wav, write_wav
from acoustic.generator import generate_sine_wave


class TestAudioIO(unittest.TestCase):
    """Test suite for WAV reading and writing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_round_trip_pcm16(self):
        """Verify round-trip save and load with PCM_16."""
        sr = 44100
        dur = 0.5
        orig = generate_sine_wave(frequency=440.0, duration=dur, sample_rate=sr, amplitude=0.75)

        wav_file = self.temp_path / "test_pcm16.wav"
        write_wav(wav_file, orig, sample_rate=sr, subtype="PCM_16")

        loaded, loaded_sr = read_wav(wav_file)
        self.assertEqual(loaded_sr, sr)
        self.assertEqual(len(loaded), len(orig))
        # 16-bit quantization noise is bounded by 1 / 32768 approx 3.1e-5
        np.testing.assert_allclose(loaded, orig, atol=1e-4)

    def test_round_trip_float32(self):
        """Verify round-trip save and load with FLOAT_32."""
        sr = 48000
        orig = generate_sine_wave(frequency=1000.0, duration=0.2, sample_rate=sr, amplitude=0.5)

        wav_file = self.temp_path / "test_float32.wav"
        write_wav(wav_file, orig, sample_rate=sr, subtype="FLOAT_32")

        loaded, loaded_sr = read_wav(wav_file)
        self.assertEqual(loaded_sr, sr)
        np.testing.assert_allclose(loaded, orig, atol=1e-6)

    def test_stereo_downmix(self):
        """Verify stereo multi-channel audio downmixes to mono."""
        sr = 22050
        n_samples = 1000
        stereo_data = np.zeros((n_samples, 2), dtype=np.float64)
        stereo_data[:, 0] = 0.5
        stereo_data[:, 1] = -0.5

        wav_file = self.temp_path / "stereo.wav"
        write_wav(wav_file, stereo_data, sample_rate=sr, subtype="FLOAT_32")

        loaded_mono, _ = read_wav(wav_file, mono=True)
        self.assertEqual(loaded_mono.shape, (n_samples,))
        # Mean of 0.5 and -0.5 is 0.0
        np.testing.assert_allclose(loaded_mono, np.zeros(n_samples), atol=1e-6)

        loaded_stereo, _ = read_wav(wav_file, mono=False)
        self.assertEqual(loaded_stereo.shape, (n_samples, 2))

    def test_invalid_arguments(self):
        """Verify proper exceptions for invalid parameters."""
        wav_file = self.temp_path / "dummy.wav"

        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            read_wav(self.temp_path / "nonexistent.wav")

        # Invalid audio array
        with self.assertRaises(ValueError):
            write_wav(wav_file, np.array([], dtype=np.float64), sample_rate=44100)

        # Audio containing NaN
        with self.assertRaises(ValueError):
            write_wav(wav_file, np.array([0.1, np.nan, 0.5], dtype=np.float64), sample_rate=44100)

        # Invalid sample rate
        with self.assertRaises(ValueError):
            write_wav(wav_file, np.array([0.1, 0.2], dtype=np.float64), sample_rate=0)

        # Boolean sample rate rejection
        with self.assertRaises(TypeError):
            write_wav(wav_file, np.array([0.1, 0.2], dtype=np.float64), sample_rate=True)

        # Non-floating audio array rejection
        with self.assertRaises(ValueError):
            write_wav(wav_file, np.array([1, 2, 3], dtype=np.int32), sample_rate=44100)

        # 3D audio array rejection
        with self.assertRaises(ValueError):
            write_wav(wav_file, np.zeros((10, 2, 2), dtype=np.float64), sample_rate=44100)

        # Non-array audio argument
        with self.assertRaises(TypeError):
            write_wav(wav_file, [0.1, 0.2], sample_rate=44100)  # type: ignore

        # Empty file path
        with self.assertRaises(ValueError):
            write_wav("   ", np.array([0.1, 0.2], dtype=np.float64), sample_rate=44100)
        with self.assertRaises(ValueError):
            read_wav("   ")

        # Invalid file path type
        with self.assertRaises(TypeError):
            write_wav(12345, np.array([0.1, 0.2], dtype=np.float64))  # type: ignore
        with self.assertRaises(TypeError):
            read_wav(12345)  # type: ignore

        # Unsupported subtype
        with self.assertRaises(ValueError):
            write_wav(wav_file, np.array([0.1, 0.2], dtype=np.float64), sample_rate=44100, subtype="MP3")

        # Write valid file to test invalid read parameters
        write_wav(wav_file, np.array([0.1, 0.2], dtype=np.float64), sample_rate=44100)

        # Non-floating target_dtype in read_wav
        with self.assertRaises(ValueError):
            read_wav(wav_file, target_dtype=np.int32)

        # Completely invalid target_dtype
        with self.assertRaises(TypeError):
            read_wav(wav_file, target_dtype="not_a_valid_dtype_xyz")


if __name__ == "__main__":
    unittest.main()
