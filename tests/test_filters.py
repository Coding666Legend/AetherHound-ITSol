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
        # Cutoff at 300 Hz: keeps 100 Hz, eliminates 1000 Hz and 8000 Hz
        filtered = butter_lowpass_filter(self.composite, cutoff_freq=300.0, sample_rate=self.sr, order=4)
        self.assertEqual(filtered.shape, self.composite.shape)
        self.assertEqual(filtered.dtype, self.composite.dtype)

        # Measure energy in frequency bins via FFT
        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0 / self.sr)

        idx_100 = np.argmin(np.abs(fft_freqs - 100.0))
        idx_1000 = np.argmin(np.abs(fft_freqs - 1000.0))
        idx_8000 = np.argmin(np.abs(fft_freqs - 8000.0))

        # 100 Hz should have massive energy compared to 1000 Hz and 8000 Hz
        self.assertGreater(fft_vals[idx_100], 100.0 * fft_vals[idx_1000])
        self.assertGreater(fft_vals[idx_100], 1000.0 * fft_vals[idx_8000])

    def test_highpass_filter_frequency_response(self):
        """Verify high-pass filter attenuates low frequencies and preserves high frequency."""
        # Cutoff at 4000 Hz: keeps 8000 Hz, eliminates 100 Hz and 1000 Hz
        filtered = butter_highpass_filter(self.composite, cutoff_freq=4000.0, sample_rate=self.sr, order=4)

        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0 / self.sr)

        idx_100 = np.argmin(np.abs(fft_freqs - 100.0))
        idx_8000 = np.argmin(np.abs(fft_freqs - 8000.0))

        self.assertGreater(fft_vals[idx_8000], 100.0 * fft_vals[idx_100])

    def test_bandpass_filter_frequency_response(self):
        """Verify band-pass filter isolates middle frequency."""
        # Passband 800 Hz to 1200 Hz: isolates 1000 Hz, eliminates 100 Hz and 8000 Hz
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


if __name__ == "__main__":
    unittest.main()
