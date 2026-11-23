"""Unit tests for the spectrogram module in AetherHound."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from acoustic.generator import generate_acoustic_anomaly, generate_sine_wave
from acoustic.spectrogram import compute_spectrogram, export_spectrogram_plot


class TestSpectrogram(unittest.TestCase):
    """Test suite for STFT spectrogram generation and offline export."""

    def setUp(self):
        self.sr = 16000
        self.duration = 1.0
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_spectrogram_shapes(self):
        """Verify output shapes and frequency range of computed spectrogram."""
        sig = generate_sine_wave(400.0, self.duration, self.sr)
        nperseg = 512
        noverlap = 256

        freqs, times, Sxx = compute_spectrogram(
            sig, sample_rate=self.sr, nperseg=nperseg, noverlap=noverlap
        )

        expected_freq_bins = nperseg // 2 + 1
        self.assertEqual(len(freqs), expected_freq_bins)
        self.assertEqual(freqs[0], 0.0)
        self.assertAlmostEqual(freqs[-1], self.sr / 2.0, delta=1.0)

        self.assertEqual(Sxx.shape[0], len(freqs))
        self.assertEqual(Sxx.shape[1], len(times))
        self.assertGreater(len(times), 1)

    def test_spectrogram_detects_time_varying_frequencies(self):
        """Verify spectrogram captures frequency shift during active pulse windows."""
        sig = generate_acoustic_anomaly(
            base_frequency=200.0,
            duration=1.0,
            sample_rate=self.sr,
            anomaly_type="frequency_shift",
            shift_hz=100.0,
            pulse_duration=0.5,
            pulse_interval=1.0,
            harmonics=(1.0,),
            noise_level=0.0,
        )

        freqs, times, Sxx = compute_spectrogram(sig, sample_rate=self.sr, nperseg=512)

        idx_base = np.argmin(np.abs(freqs - 200.0))
        idx_shifted = np.argmin(np.abs(freqs - 300.0))

        # First half of second (0 to 0.5s): shifted frequency (300 Hz) active
        first_half_cols = np.where(times < 0.45)[0]
        # Second half of second (0.5 to 1.0s): base frequency (200 Hz) active
        second_half_cols = np.where(times > 0.55)[0]

        mean_first_half_shifted = np.mean(Sxx[idx_shifted, first_half_cols])
        mean_first_half_base = np.mean(Sxx[idx_base, first_half_cols])
        self.assertGreater(mean_first_half_shifted, mean_first_half_base)

        mean_second_half_base = np.mean(Sxx[idx_base, second_half_cols])
        mean_second_half_shifted = np.mean(Sxx[idx_shifted, second_half_cols])
        self.assertGreater(mean_second_half_base, mean_second_half_shifted)

    def test_zero_overlap(self):
        """Verify noverlap=0 operates correctly."""
        sig = generate_sine_wave(300.0, 0.5, self.sr)
        freqs, times, Sxx = compute_spectrogram(sig, sample_rate=self.sr, nperseg=256, noverlap=0)
        self.assertEqual(Sxx.shape[0], 256 // 2 + 1)
        self.assertGreater(Sxx.shape[1], 1)

    def test_full_length_nperseg_boundary(self):
        """Verify nperseg == len(data) boundary operates safely."""
        sig = generate_sine_wave(300.0, 0.05, self.sr)  # 800 samples
        n = len(sig)
        freqs, times, Sxx = compute_spectrogram(sig, sample_rate=self.sr, nperseg=n, noverlap=0)
        self.assertEqual(Sxx.shape, (n // 2 + 1, 1))

    def test_non_floating_dtype_rejection(self):
        """Verify integer arrays are rejected in compute_spectrogram."""
        int_data = np.arange(1000, dtype=np.int32)
        with self.assertRaises(ValueError):
            compute_spectrogram(int_data, sample_rate=self.sr, nperseg=256)

    def test_export_spectrogram_plot(self):
        """Verify offline image export produces a valid image file."""
        sig = generate_sine_wave(300.0, 0.5, self.sr)
        freqs, times, Sxx = compute_spectrogram(sig, sample_rate=self.sr)

        plot_file = self.temp_path / "spectrogram.png"
        export_spectrogram_plot(freqs, times, Sxx, output_path=plot_file, title="Test Plot")

        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 1000)

    def test_export_spectrogram_plot_validation(self):
        """Verify error checking on mismatched dimensions in export_spectrogram_plot."""
        freqs = np.linspace(0, 8000, 129)
        times = np.linspace(0, 1.0, 30)
        # Mismatched Sxx shape
        with self.assertRaises(ValueError):
            export_spectrogram_plot(freqs, times, np.zeros((100, 30)), self.temp_path / "out.png")

        # Non-finite values in Sxx
        bad_sxx = np.zeros((129, 30))
        bad_sxx[0, 0] = np.nan
        with self.assertRaises(ValueError):
            export_spectrogram_plot(freqs, times, bad_sxx, self.temp_path / "out.png")

    def test_invalid_parameters(self):
        """Verify error checking on invalid spectrogram parameters."""
        sig = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        # Data shorter than nperseg
        with self.assertRaises(ValueError):
            compute_spectrogram(sig, nperseg=512)

        # Negative nperseg
        with self.assertRaises(ValueError):
            compute_spectrogram(sig, nperseg=-10)

        # noverlap >= nperseg
        with self.assertRaises(ValueError):
            compute_spectrogram(np.ones(1000, dtype=np.float64), nperseg=256, noverlap=256)

        # 2D array
        with self.assertRaises(ValueError):
            compute_spectrogram(np.ones((1000, 2), dtype=np.float64), nperseg=256)

        # Boolean parameters
        with self.assertRaises(TypeError):
            compute_spectrogram(np.ones(1000, dtype=np.float64), sample_rate=True)
        with self.assertRaises(TypeError):
            compute_spectrogram(np.ones(1000, dtype=np.float64), nperseg=True)


if __name__ == "__main__":
    unittest.main()
