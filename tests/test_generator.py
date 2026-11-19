"""Unit tests for the synthetic audio generator in the acoustic DSP module."""

import math
import unittest
import numpy as np

from acoustic.generator import generate_sine_wave


class TestSineWaveGenerator(unittest.TestCase):
    """Test suite for generate_sine_wave function."""

    def test_basic_generation(self):
        """Test default generation of a 440 Hz standard concert pitch tone."""
        freq = 440.0
        duration = 1.0
        sample_rate = 44100
        signal = generate_sine_wave(frequency=freq, duration=duration, sample_rate=sample_rate)

        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.shape, (44100,))
        self.assertEqual(signal.dtype, np.float64)
        self.assertLessEqual(np.max(signal), 1.0)
        self.assertGreaterEqual(np.min(signal), -1.0)

    def test_output_shape_and_duration(self):
        """Verify output sample counts for diverse durations and sample rates."""
        test_cases = [
            (0.01, 8000),
            (0.25, 16000),
            (0.5, 44100),
            (1.5, 48000),
            (2.0, 22050),
        ]
        for duration, sr in test_cases:
            with self.subTest(duration=duration, sample_rate=sr):
                signal = generate_sine_wave(frequency=200.0, duration=duration, sample_rate=sr)
                expected_samples = int(round(duration * sr))
                self.assertEqual(len(signal), expected_samples)

    def test_output_dtype(self):
        """Verify output array data types for float64 and float32."""
        sig64 = generate_sine_wave(frequency=300.0, duration=0.1, dtype=np.float64)
        self.assertEqual(sig64.dtype, np.float64)

        sig32 = generate_sine_wave(frequency=300.0, duration=0.1, dtype=np.float32)
        self.assertEqual(sig32.dtype, np.float32)

    def test_different_frequencies_spectral_accuracy(self):
        """Verify frequency accuracy using FFT for low, mid, and high frequencies."""
        sr = 48000
        duration = 1.0
        test_frequencies = [60.0, 440.0, 1000.0, 5000.0, 18000.0]

        for freq in test_frequencies:
            with self.subTest(target_frequency=freq):
                signal = generate_sine_wave(
                    frequency=freq, duration=duration, sample_rate=sr, amplitude=1.0
                )
                # Compute single-sided magnitude spectrum
                fft_vals = np.abs(np.fft.rfft(signal))
                fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
                peak_freq = fft_freqs[np.argmax(fft_vals)]

                # Peak frequency should match target within FFT bin resolution (sr / N = 1 Hz)
                self.assertAlmostEqual(peak_freq, freq, delta=1.0)

    def test_amplitude_behavior(self):
        """Verify signal scaling for various amplitude settings."""
        # Silence (amplitude = 0.0)
        silent = generate_sine_wave(frequency=500.0, duration=0.5, amplitude=0.0)
        np.testing.assert_array_equal(silent, np.zeros_like(silent))

        # Scaled amplitudes
        for amp in [0.1, 0.5, 0.8, 1.0]:
            with self.subTest(amplitude=amp):
                sig = generate_sine_wave(
                    frequency=1000.0, duration=0.5, sample_rate=44100, amplitude=amp
                )
                self.assertAlmostEqual(np.max(sig), amp, delta=1e-3)
                self.assertAlmostEqual(np.min(sig), -amp, delta=1e-3)

    def test_phase_offset(self):
        """Verify initial phase shifts."""
        sr = 44100
        # Phase 0 radians: sin(0) = 0
        sig_sin = generate_sine_wave(frequency=100.0, duration=0.1, sample_rate=sr, phase=0.0)
        self.assertAlmostEqual(sig_sin[0], 0.0, places=6)

        # Phase pi/2 radians: sin(pi/2) = 1.0 (cosine)
        sig_cos = generate_sine_wave(
            frequency=100.0, duration=0.1, sample_rate=sr, phase=np.pi / 2.0
        )
        self.assertAlmostEqual(sig_cos[0], 1.0, places=6)

    def test_invalid_frequency(self):
        """Ensure negative, zero, and above-Nyquist frequencies raise ValueError."""
        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=0.0, duration=1.0)

        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=-100.0, duration=1.0)

        # Nyquist frequency is sample_rate / 2 = 22050 Hz for 44100 Hz
        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=22050.0, duration=1.0, sample_rate=44100)

        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=25000.0, duration=1.0, sample_rate=44100)

    def test_invalid_duration(self):
        """Ensure negative or zero duration raises ValueError."""
        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=440.0, duration=0.0)

        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=440.0, duration=-1.5)

    def test_invalid_sample_rate(self):
        """Ensure negative or zero sample rate raises ValueError."""
        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=440.0, duration=1.0, sample_rate=0)

        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=440.0, duration=1.0, sample_rate=-44100)

    def test_invalid_amplitude(self):
        """Ensure amplitudes outside [0.0, 1.0] raise ValueError."""
        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=440.0, duration=1.0, amplitude=-0.1)

        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=440.0, duration=1.0, amplitude=1.05)

    def test_invalid_types(self):
        """Ensure non-numeric types raise TypeError."""
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency="440", duration=1.0)

        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration="1.0")

        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, sample_rate="44100")

        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, amplitude="loud")

        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, phase=None)

    # Robustness Review Tests

    def test_boolean_rejection(self):
        """Verify boolean values are rejected for all numeric parameters."""
        # Booleans inherit from int in Python; they must be explicitly rejected
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=True, duration=1.0)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=False, duration=1.0)

        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=True)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=False)

        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, sample_rate=True)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, sample_rate=False)

        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, amplitude=True)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, amplitude=False)

        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, phase=True)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, phase=False)

    def test_nan_and_infinity_rejection(self):
        """Verify NaN and infinity are rejected across all numeric parameters."""
        non_finite_values = [float("nan"), float("inf"), float("-inf")]

        for val in non_finite_values:
            with self.subTest(param="frequency", val=val):
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=val, duration=1.0)

            with self.subTest(param="duration", val=val):
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=440.0, duration=val)

            with self.subTest(param="sample_rate", val=val):
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=440.0, duration=1.0, sample_rate=val)

            with self.subTest(param="amplitude", val=val):
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=440.0, duration=1.0, amplitude=val)

            with self.subTest(param="phase", val=val):
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=440.0, duration=1.0, phase=val)

    def test_invalid_and_non_floating_dtypes(self):
        """Verify non-floating NumPy dtypes and invalid dtype specs are rejected."""
        # Non-floating dtypes should raise ValueError
        non_floating = [int, np.int16, np.int32, np.int64, bool, np.bool_, object, str]
        for dt in non_floating:
            with self.subTest(dtype=dt):
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=440.0, duration=0.1, dtype=dt)

        # Invalid dtype specifications should raise TypeError
        invalid_specs = ["nonexistent_dtype", [1, 2, 3]]
        for spec in invalid_specs:
            with self.subTest(spec=spec):
                with self.assertRaises(TypeError):
                    generate_sine_wave(frequency=440.0, duration=0.1, dtype=spec)

    def test_extremely_small_duration_handling(self):
        """Verify extremely small durations producing zero samples raise ValueError."""
        # 1 nanosecond or 1 picosecond at 44.1 kHz cannot produce even 1 discrete sample
        tiny_durations = [1e-9, 1e-12, 1e-15]
        for dur in tiny_durations:
            with self.subTest(duration=dur):
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=440.0, duration=dur, sample_rate=44100)

    def test_phase_finiteness_and_offsets(self):
        """Verify phase finiteness validation and arbitrary valid phase angles."""
        # Valid phases
        for ph in [0.0, math.pi / 4, -math.pi / 2, 2 * math.pi, -10.5]:
            with self.subTest(phase=ph):
                sig = generate_sine_wave(frequency=100.0, duration=0.1, phase=ph)
                self.assertEqual(len(sig), 4410)
                self.assertFalse(np.any(np.isnan(sig)))


if __name__ == "__main__":
    unittest.main()
