"""Unit tests for the acoustic feature extraction module in AetherHound."""

from datetime import datetime, timezone
import unittest
import numpy as np

from acoustic.features import (
    FeatureFrame,
    compute_dominant_frequency,
    compute_rms_energy,
    compute_spectral_bandwidth,
    compute_spectral_centroid,
    compute_spectral_rolloff,
    compute_zero_crossing_rate,
    create_feature_frame,
    extract_acoustic_features,
)
from acoustic.generator import generate_sine_wave


class TestFeatures(unittest.TestCase):
    """Test suite for feature extraction and FeatureFrame creation."""

    def setUp(self):
        self.sr = 44100
        self.duration = 1.0
        # 1000 Hz pure sine wave of amplitude 1.0
        self.tone_1000 = generate_sine_wave(
            1000.0, self.duration, self.sr, amplitude=1.0, phase=0.0
        )

    def test_rms_energy(self):
        """Verify RMS energy of a sine wave equals peak / sqrt(2)."""
        rms = compute_rms_energy(self.tone_1000)
        expected_rms = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(rms, expected_rms, delta=0.01)

    def test_zero_crossing_rate(self):
        """Verify zero-crossing rate of a 1000 Hz wave at 44.1 kHz."""
        zcr = compute_zero_crossing_rate(self.tone_1000)
        expected_zcr = 2.0 * 1000.0 / self.sr
        self.assertAlmostEqual(zcr, expected_zcr, delta=0.005)

    def test_spectral_features_on_pure_tone(self):
        """Verify spectral centroid, bandwidth, and dominant frequency on pure tone."""
        freqs = np.fft.rfftfreq(len(self.tone_1000), d=1.0 / self.sr)
        mags = np.abs(np.fft.rfft(self.tone_1000))

        dom_f = compute_dominant_frequency(freqs, mags)
        self.assertAlmostEqual(dom_f, 1000.0, delta=1.0)

        centroid = compute_spectral_centroid(freqs, mags)
        self.assertAlmostEqual(centroid, 1000.0, delta=5.0)

        bandwidth = compute_spectral_bandwidth(freqs, mags, centroid=centroid)
        self.assertLess(bandwidth, 50.0)

        rolloff = compute_spectral_rolloff(freqs, mags, roll_percent=0.85)
        self.assertAlmostEqual(rolloff, 1000.0, delta=5.0)

    def test_zero_silence_handling(self):
        """Verify silence (all zeros) returns 0.0 without division by zero errors."""
        silent = np.zeros(1000, dtype=np.float64)
        features = extract_acoustic_features(silent, sample_rate=self.sr)

        for name, val in features.items():
            self.assertEqual(val, 0.0, f"Feature {name} should be 0.0 for silence")

    def test_centroid_frequency_shift(self):
        """Verify high-frequency tone yields higher centroid than low-frequency tone."""
        low_tone = generate_sine_wave(200.0, 0.5, self.sr)
        high_tone = generate_sine_wave(5000.0, 0.5, self.sr)

        feats_low = extract_acoustic_features(low_tone, sample_rate=self.sr)
        feats_high = extract_acoustic_features(high_tone, sample_rate=self.sr)

        self.assertGreater(feats_high["spectral_centroid"], feats_low["spectral_centroid"] + 4000.0)

    def test_dc_signal_features(self):
        """Verify constant DC signal features (zero crossings and dominant frequency)."""
        dc_data = np.full(1000, 0.5, dtype=np.float64)
        self.assertEqual(compute_zero_crossing_rate(dc_data), 0.0)
        self.assertAlmostEqual(compute_rms_energy(dc_data), 0.5, delta=1e-5)

    def test_rolloff_percent_boundaries(self):
        """Verify rolloff percentages and boundary validation."""
        freqs = np.linspace(0, 1000, 101)
        mags = np.ones(101)

        r50 = compute_spectral_rolloff(freqs, mags, roll_percent=0.5)
        self.assertAlmostEqual(r50, 500.0, delta=15.0)

        # roll_percent <= 0 or > 1
        with self.assertRaises(ValueError):
            compute_spectral_rolloff(freqs, mags, roll_percent=0.0)
        with self.assertRaises(ValueError):
            compute_spectral_rolloff(freqs, mags, roll_percent=-0.5)
        with self.assertRaises(ValueError):
            compute_spectral_rolloff(freqs, mags, roll_percent=1.1)
        with self.assertRaises(TypeError):
            compute_spectral_rolloff(freqs, mags, roll_percent=True)

    def test_spectral_vector_validation(self):
        """Verify validation of frequencies and magnitudes in spectral feature functions."""
        # Mismatched lengths
        with self.assertRaises(ValueError):
            compute_spectral_centroid(np.ones(10), np.ones(20))

        # 2D arrays
        with self.assertRaises(ValueError):
            compute_spectral_centroid(np.ones((5, 2)), np.ones((5, 2)))

        # Non-finite values
        bad_mag = np.ones(10)
        bad_mag[2] = np.nan
        with self.assertRaises(ValueError):
            compute_spectral_centroid(np.ones(10), bad_mag)

    def test_extract_acoustic_features_keys_and_values(self):
        """Verify extract_acoustic_features returns all expected keys with finite floats."""
        features = extract_acoustic_features(self.tone_1000, sample_rate=self.sr)
        expected_keys = {
            "rms_energy",
            "zero_crossing_rate",
            "spectral_centroid",
            "spectral_bandwidth",
            "dominant_frequency",
            "spectral_rolloff",
        }
        self.assertEqual(set(features.keys()), expected_keys)
        for k, v in features.items():
            self.assertIsInstance(v, float)
            self.assertTrue(np.isfinite(v))

    def test_create_feature_frame(self):
        """Verify FeatureFrame container matches architectural expectations."""
        now = datetime.now(timezone.utc)
        frame = create_feature_frame(
            self.tone_1000,
            sample_rate=self.sr,
            source_id="mic_sensor_alpha",
            timestamp=now,
        )

        self.assertIsInstance(frame, FeatureFrame)
        self.assertEqual(frame.modality, "acoustic")
        self.assertEqual(frame.source_id, "mic_sensor_alpha")
        self.assertEqual(frame.timestamp, now)
        self.assertIsInstance(frame.features, np.ndarray)
        self.assertEqual(len(frame.features), len(frame.feature_names))
        self.assertTrue(np.all(np.isfinite(frame.features)))

    def test_create_feature_frame_validation(self):
        """Verify source_id and timestamp validation in create_feature_frame."""
        # Empty source_id
        with self.assertRaises(ValueError):
            create_feature_frame(self.tone_1000, source_id="")
        with self.assertRaises(ValueError):
            create_feature_frame(self.tone_1000, source_id="   ")

        # Non-string source_id
        with self.assertRaises(ValueError):
            create_feature_frame(self.tone_1000, source_id=123)

        # Invalid timestamp type
        with self.assertRaises(TypeError):
            create_feature_frame(self.tone_1000, timestamp="2026-11-23")

    def test_invalid_parameters(self):
        """Verify error checking on invalid parameters."""
        with self.assertRaises(ValueError):
            extract_acoustic_features(np.array([], dtype=np.float64))
        with self.assertRaises(ValueError):
            extract_acoustic_features(np.array([1.0, np.nan]))
        with self.assertRaises(ValueError):
            extract_acoustic_features(np.array([1.0, 2.0]), sample_rate=-44100)
        # Non-floating dtype
        with self.assertRaises(ValueError):
            extract_acoustic_features(np.array([1, 2, 3], dtype=np.int32))
        # 2D array
        with self.assertRaises(ValueError):
            extract_acoustic_features(np.ones((100, 2), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
