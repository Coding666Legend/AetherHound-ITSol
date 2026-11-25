"""Unit tests for the synthetic audio generator in the acoustic DSP module."""

import math
import unittest
import numpy as np

from acoustic.generator import generate_acoustic_anomaly, generate_fan_noise, generate_sine_wave


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
                fft_vals = np.abs(np.fft.rfft(signal))
                fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
                peak_freq = fft_freqs[np.argmax(fft_vals)]
                self.assertAlmostEqual(peak_freq, freq, delta=1.0)

    def test_amplitude_behavior(self):
        """Verify signal scaling for various amplitude settings."""
        silent = generate_sine_wave(frequency=500.0, duration=0.5, amplitude=0.0)
        np.testing.assert_array_equal(silent, np.zeros_like(silent))

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
        sig_sin = generate_sine_wave(frequency=100.0, duration=0.1, sample_rate=sr, phase=0.0)
        self.assertAlmostEqual(sig_sin[0], 0.0, places=6)

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

        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=22050.0, duration=1.0, sample_rate=44100)

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
            generate_sine_wave(frequency=440.0, duration=1.0, phase=None)

    def test_boolean_rejection(self):
        """Verify boolean values are rejected for all numeric parameters."""
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=True, duration=1.0)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=True)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, sample_rate=True)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, amplitude=True)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=1.0, phase=True)

    def test_nan_and_infinity_rejection(self):
        """Verify NaN and infinity are rejected across all numeric parameters."""
        non_finite = [float("nan"), float("inf"), float("-inf")]
        for val in non_finite:
            with self.subTest(val=val):
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=val, duration=1.0)
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=440.0, duration=val)
                with self.assertRaises(ValueError):
                    generate_sine_wave(frequency=440.0, duration=1.0, amplitude=val)

    def test_invalid_and_non_floating_dtypes(self):
        """Verify non-floating NumPy dtypes and invalid dtype specs are rejected."""
        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=440.0, duration=0.1, dtype=int)
        with self.assertRaises(TypeError):
            generate_sine_wave(frequency=440.0, duration=0.1, dtype="bad_dtype")

    def test_extremely_small_duration_handling(self):
        """Verify extremely small durations producing zero samples raise ValueError."""
        with self.assertRaises(ValueError):
            generate_sine_wave(frequency=440.0, duration=1e-12, sample_rate=44100)


class TestFanNoiseGenerator(unittest.TestCase):
    """Test suite for generate_fan_noise function."""

    def test_basic_fan_noise_generation(self):
        """Test default generation of fan noise."""
        signal = generate_fan_noise(base_frequency=120.0, duration=0.5, sample_rate=44100, seed=42)
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(len(signal), 22050)
        self.assertLessEqual(np.max(signal), 0.8 + 1e-4)
        self.assertGreaterEqual(np.min(signal), -0.8 - 1e-4)

    def test_reproducible_seed(self):
        """Verify that identical seeds produce identical audio samples."""
        sig1 = generate_fan_noise(base_frequency=100.0, duration=0.2, seed=1234)
        sig2 = generate_fan_noise(base_frequency=100.0, duration=0.2, seed=1234)
        np.testing.assert_array_equal(sig1, sig2)

        sig3 = generate_fan_noise(base_frequency=100.0, duration=0.2, seed=9999)
        self.assertFalse(np.array_equal(sig1, sig3))

    def test_harmonic_peaks_presence(self):
        """Verify presence of fundamental and harmonic frequency peaks using FFT."""
        sr = 48000
        dur = 1.0
        f0 = 150.0
        # No noise, pure harmonics 1.0, 0.5, 0.25
        sig = generate_fan_noise(
            base_frequency=f0,
            duration=dur,
            sample_rate=sr,
            harmonics=(1.0, 0.5, 0.25),
            noise_level=0.0,
            amplitude=1.0,
        )
        fft_mag = np.abs(np.fft.rfft(sig))
        fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)

        for harmonic_order in [1, 2, 3]:
            target_f = f0 * harmonic_order
            idx = int(round(target_f))
            # Harmonic bin should be a local peak
            self.assertGreater(fft_mag[idx], np.mean(fft_mag))

    def test_noise_level_scaling(self):
        """Verify zero noise produces deterministic signal and higher noise adds variance."""
        tonal = generate_fan_noise(base_frequency=200.0, duration=0.1, noise_level=0.0)
        noisy = generate_fan_noise(base_frequency=200.0, duration=0.1, noise_level=0.5, seed=42)
        self.assertNotEqual(np.std(tonal), np.std(noisy))

    def test_invalid_parameters(self):
        """Verify parameter range checking for fan noise generator."""
        with self.assertRaises(ValueError):
            generate_fan_noise(base_frequency=0.0, duration=1.0)
        with self.assertRaises(ValueError):
            generate_fan_noise(base_frequency=100.0, duration=0.0)
        with self.assertRaises(ValueError):
            generate_fan_noise(base_frequency=100.0, duration=1.0, amplitude=1.5)
        with self.assertRaises(ValueError):
            generate_fan_noise(base_frequency=100.0, duration=1.0, noise_level=-0.1)
        with self.assertRaises(ValueError):
            generate_fan_noise(base_frequency=100.0, duration=1.0, seed=-5)
        with self.assertRaises(TypeError):
            generate_fan_noise(base_frequency=100.0, duration=1.0, seed=True)

    def test_harmonic_exceeding_nyquist(self):
        """Verify that harmonics exceeding Nyquist limit raise ValueError."""
        # 10000 Hz with 3 harmonics = 30000 Hz, exceeding 22050 Hz Nyquist for 44.1 kHz
        with self.assertRaises(ValueError):
            generate_fan_noise(base_frequency=10000.0, duration=0.1, harmonics=(1.0, 0.5, 0.25))


class TestAcousticAnomalyGenerator(unittest.TestCase):
    """Test suite for generate_acoustic_anomaly function."""

    def test_frequency_shift_anomaly(self):
        """Verify frequency shift anomaly generation."""
        sig = generate_acoustic_anomaly(
            base_frequency=120.0,
            duration=1.0,
            sample_rate=44100,
            anomaly_type="frequency_shift",
            shift_hz=30.0,
            pulse_duration=0.5,
            pulse_interval=1.0,
            seed=42,
        )
        self.assertIsInstance(sig, np.ndarray)
        self.assertEqual(len(sig), 44100)
        self.assertLessEqual(np.max(sig), 0.8 + 1e-4)

    def test_pulsed_burst_anomaly(self):
        """Verify pulsed burst anomaly generation."""
        sig = generate_acoustic_anomaly(
            base_frequency=100.0,
            duration=0.5,
            anomaly_type="pulsed_burst",
            shift_hz=50.0,
            pulse_duration=0.2,
            pulse_interval=0.5,
            seed=42,
        )
        self.assertEqual(len(sig), int(0.5 * 44100))

    def test_frequency_modulation_anomaly(self):
        """Verify continuous frequency modulation anomaly."""
        sig = generate_acoustic_anomaly(
            base_frequency=200.0,
            duration=0.5,
            anomaly_type="frequency_modulation",
            shift_hz=20.0,
            pulse_interval=0.25,
            seed=42,
        )
        self.assertEqual(len(sig), int(0.5 * 44100))

    def test_invalid_anomaly_type(self):
        """Verify unknown anomaly type raises ValueError."""
        with self.assertRaises(ValueError):
            generate_acoustic_anomaly(base_frequency=100.0, duration=1.0, anomaly_type="unknown")

    def test_pulse_duration_exceeding_interval(self):
        """Verify pulse_duration > pulse_interval raises ValueError."""
        with self.assertRaises(ValueError):
            generate_acoustic_anomaly(
                base_frequency=100.0,
                duration=1.0,
                pulse_duration=1.5,
                pulse_interval=1.0,
            )

    def test_anomaly_exceeding_nyquist(self):
        """Verify anomaly with high shifted frequency exceeding Nyquist raises ValueError."""
        with self.assertRaises(ValueError):
            generate_acoustic_anomaly(
                base_frequency=20000.0,
                duration=0.1,
                sample_rate=44100,
                shift_hz=5000.0,
                harmonics=(1.0,),
            )

    def test_zero_sample_duration_regression(self):
        """Verify that micro-durations yielding 0 samples raise clear ValueError with guidance."""
        with self.assertRaises(ValueError) as ctx:
            generate_fan_noise(base_frequency=200.0, duration=1e-8, sample_rate=44100)
        self.assertIn("produces 0 samples", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            generate_acoustic_anomaly(base_frequency=200.0, duration=1e-8, sample_rate=44100)
        self.assertIn("produces 0 samples", str(ctx2.exception))

    def test_generator_dtype_validation_regression(self):
        """Verify integer and boolean dtypes are rejected consistently across generators."""
        with self.assertRaises(ValueError):
            generate_fan_noise(base_frequency=200.0, duration=0.1, dtype=np.int32)
        with self.assertRaises(ValueError):
            generate_acoustic_anomaly(base_frequency=200.0, duration=0.1, dtype=np.int16)
        with self.assertRaises(ValueError):
            generate_fan_noise(base_frequency=200.0, duration=0.1, dtype=bool)

    def test_generator_deterministic_seed_regression(self):
        """Verify reproducible signal generation across runs with identical seeds."""
        fan_a = generate_fan_noise(base_frequency=150.0, duration=0.2, noise_level=0.2, seed=99)
        fan_b = generate_fan_noise(base_frequency=150.0, duration=0.2, noise_level=0.2, seed=99)
        np.testing.assert_array_equal(fan_a, fan_b)

        anomaly_a = generate_acoustic_anomaly(
            base_frequency=150.0, duration=0.2, noise_level=0.1, seed=123
        )
        anomaly_b = generate_acoustic_anomaly(
            base_frequency=150.0, duration=0.2, noise_level=0.1, seed=123
        )
        np.testing.assert_array_equal(anomaly_a, anomaly_b)


if __name__ == "__main__":
    unittest.main()

