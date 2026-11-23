"""Unit tests for the digital filtering module in AetherHound."""

import unittest
import numpy as np

from acoustic.filters import (
    butter_bandpass_filter,
    butter_highpass_filter,
    butter_lowpass_filter,
)
from acoustic.generator import generate_sine_wave


class TestFilters(unittest.TestCase):
    """Test suite for Butterworth digital filters."""

    def setUp(self):
        self.sr = 44100
        self.duration = 1.0
        # Create composite signal: 100 Hz (low), 1000 Hz (mid), 8000 Hz (high)
        self.sig_low = generate_sine_wave(100.0, self.duration, self.sr, amplitude=1.0)
        self.sig_mid = generate_sine_wave(1000.0, self.duration, self.sr, amplitude=1.0)
        self.sig_high = generate_sine_wave(8000.0, self.duration, self.sr, amplitude=1.0)
        self.composite = self.sig_low + self.sig_mid + self.sig_high

    def test_lowpass_filter_frequency_response(self):
        """Verify low-pass filter attenuates high frequencies and preserves low frequency."""
        filtered = butter_lowpass_filter(self.composite, cutoff_freq=300.0, sample_rate=self.sr, order=4)
        self.assertEqual(filtered.shape, self.composite.shape)
        self.assertEqual(filtered.dtype, self.composite.dtype)

        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0 / self.sr)

        idx_100 = np.argmin(np.abs(fft_freqs - 100.0))
        idx_1000 = np.argmin(np.abs(fft_freqs - 1000.0))
        idx_8000 = np.argmin(np.abs(fft_freqs - 8000.0))

        self.assertGreater(fft_vals[idx_100], 100.0 * fft_vals[idx_1000])
        self.assertGreater(fft_vals[idx_100], 1000.0 * fft_vals[idx_8000])

    def test_highpass_filter_frequency_response(self):
        """Verify high-pass filter attenuates low frequencies and preserves high frequency."""
        filtered = butter_highpass_filter(self.composite, cutoff_freq=4000.0, sample_rate=self.sr, order=4)

        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0 / self.sr)

        idx_100 = np.argmin(np.abs(fft_freqs - 100.0))
        idx_8000 = np.argmin(np.abs(fft_freqs - 8000.0))

        self.assertGreater(fft_vals[idx_8000], 100.0 * fft_vals[idx_100])

    def test_bandpass_filter_frequency_response(self):
        """Verify band-pass filter isolates middle frequency."""
        filtered = butter_bandpass_filter(
            self.composite, low_cutoff=800.0, high_cutoff=1200.0, sample_rate=self.sr, order=4
        )

        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0 / self.sr)

        idx_100 = np.argmin(np.abs(fft_freqs - 100.0))
        idx_1000 = np.argmin(np.abs(fft_freqs - 1000.0))
        idx_8000 = np.argmin(np.abs(fft_freqs - 8000.0))

        self.assertGreater(fft_vals[idx_1000], 50.0 * fft_vals[idx_100])
        self.assertGreater(fft_vals[idx_1000], 50.0 * fft_vals[idx_8000])

    def test_filter_zero_phase_alignment(self):
        """Verify filtfilt zero-phase filter preserves peak positions without delay."""
        pure_sine = generate_sine_wave(100.0, 0.1, self.sr)
        filtered = butter_lowpass_filter(pure_sine, cutoff_freq=500.0, sample_rate=self.sr)

        peak_orig = np.argmax(pure_sine[:200])
        peak_filt = np.argmax(filtered[:200])
        self.assertEqual(peak_orig, peak_filt)

    def test_multichannel_2d_filtering(self):
        """Verify filtering functions correctly process 2D multi-channel signals."""
        stereo = np.column_stack([self.sig_low, self.sig_high])
        filtered_stereo = butter_lowpass_filter(stereo, cutoff_freq=500.0, sample_rate=self.sr)

        self.assertEqual(filtered_stereo.shape, stereo.shape)
        # Channel 0 (100 Hz) should maintain energy; Channel 1 (8000 Hz) should be attenuated
        rms_ch0 = np.sqrt(np.mean(filtered_stereo[:, 0] ** 2))
        rms_ch1 = np.sqrt(np.mean(filtered_stereo[:, 1] ** 2))
        self.assertGreater(rms_ch0, 100.0 * rms_ch1)

    def test_order_boundaries(self):
        """Verify minimum order 1 and maximum order 12 boundaries."""
        # Order 1 (valid minimum)
        f1 = butter_lowpass_filter(self.composite, cutoff_freq=1000.0, sample_rate=self.sr, order=1)
        self.assertEqual(f1.shape, self.composite.shape)

        # Order 12 (valid maximum)
        f12 = butter_lowpass_filter(self.composite, cutoff_freq=1000.0, sample_rate=self.sr, order=12)
        self.assertEqual(f12.shape, self.composite.shape)

        # Order 13 (unstable, should be rejected)
        with self.assertRaises(ValueError):
            butter_lowpass_filter(self.composite, cutoff_freq=1000.0, sample_rate=self.sr, order=13)

        # Order 0 (rejected)
        with self.assertRaises(ValueError):
            butter_lowpass_filter(self.composite, cutoff_freq=1000.0, sample_rate=self.sr, order=0)

    def test_non_floating_dtype_rejection(self):
        """Verify non-floating NumPy arrays are rejected."""
        int_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5, dtype=np.int16)
        with self.assertRaises(ValueError):
            butter_lowpass_filter(int_data, cutoff_freq=500.0, sample_rate=self.sr)

        bool_data = np.ones(50, dtype=bool)
        with self.assertRaises(ValueError):
            butter_lowpass_filter(bool_data, cutoff_freq=500.0, sample_rate=self.sr)

    def test_invalid_filter_parameters(self):
        """Verify input validation and edge cases."""
        # Cutoff >= Nyquist
        with self.assertRaises(ValueError):
            butter_lowpass_filter(self.composite, cutoff_freq=22050.0, sample_rate=self.sr)

        # Negative cutoff
        with self.assertRaises(ValueError):
            butter_highpass_filter(self.composite, cutoff_freq=-10.0, sample_rate=self.sr)

        # low_cutoff >= high_cutoff in bandpass
        with self.assertRaises(ValueError):
            butter_bandpass_filter(self.composite, low_cutoff=1000.0, high_cutoff=500.0, sample_rate=self.sr)

        # Data too short
        with self.assertRaises(ValueError):
            butter_lowpass_filter(np.array([1.0, 2.0]), cutoff_freq=500.0, sample_rate=self.sr, order=4)

        # 3D array
        with self.assertRaises(ValueError):
            butter_lowpass_filter(np.ones((100, 2, 2)), cutoff_freq=500.0, sample_rate=self.sr)

        # Non-finite values
        with self.assertRaises(ValueError):
            butter_lowpass_filter(np.array([1.0, np.nan, 3.0] * 20), cutoff_freq=500.0, sample_rate=self.sr)

        # Boolean parameters
        with self.assertRaises(TypeError):
            butter_lowpass_filter(self.composite, cutoff_freq=500.0, sample_rate=True)
        with self.assertRaises(TypeError):
            butter_lowpass_filter(self.composite, cutoff_freq=True, sample_rate=self.sr)
        with self.assertRaises(TypeError):
            butter_lowpass_filter(self.composite, cutoff_freq=500.0, order=True)


if __name__ == "__main__":
    unittest.main()
