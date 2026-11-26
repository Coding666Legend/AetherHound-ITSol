"""Unit and integration tests for the acoustic DSP processing pipeline in AetherHound."""

import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from acoustic.audio_io import write_wav
from acoustic.features import FeatureFrame
from acoustic.generator import generate_fan_noise, generate_sine_wave
from acoustic.pipeline import (
    AcousticPipeline,
    FilterConfig,
    ProcessingResult,
    SpectrogramConfig,
)


class TestAcousticPipeline(unittest.TestCase):
    """Test suite for AcousticPipeline orchestration."""

    def setUp(self):
        self.sr = 44100
        self.duration = 0.5  # 22050 samples (> 1024 nperseg)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 440 Hz test tone
        self.tone_440 = generate_sine_wave(
            frequency=440.0,
            duration=self.duration,
            sample_rate=self.sr,
            amplitude=0.8,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_pipeline_execution(self):
        """Verify normal pipeline execution with default configurations (no filtering)."""
        pipeline = AcousticPipeline(sample_rate=self.sr)
        result = pipeline.process(self.tone_440, source_id="sensor_01")

        self.assertIsInstance(result, ProcessingResult)
        self.assertEqual(result.source_id, "sensor_01")
        self.assertEqual(result.sample_rate, self.sr)
        self.assertFalse(result.filtered)
        self.assertIsNone(result.filter_type)

        # Signal arrays
        self.assertEqual(len(result.raw_signal), len(self.tone_440))
        self.assertEqual(len(result.processed_signal), len(self.tone_440))
        np.testing.assert_array_equal(result.raw_signal, result.processed_signal)

        # Spectral analysis
        self.assertGreater(len(result.frequencies), 0)
        self.assertEqual(len(result.frequencies), len(result.magnitudes))
        self.assertGreater(len(result.spectral_peaks), 0)
        peak_f, peak_mag = result.spectral_peaks[0]
        self.assertAlmostEqual(peak_f, 440.0, delta=2.0)

        # Spectrogram output
        self.assertEqual(result.spectrogram_matrix.shape[0], len(result.spectrogram_freqs))
        self.assertEqual(result.spectrogram_matrix.shape[1], len(result.spectrogram_times))
        self.assertGreater(result.spectrogram_matrix.shape[1], 1)

        # Features output
        self.assertIn("dominant_frequency", result.features)
        self.assertIn("rms_energy", result.features)
        self.assertIn("spectral_centroid", result.features)
        self.assertAlmostEqual(result.features["dominant_frequency"], 440.0, delta=2.0)

        # FeatureFrame
        self.assertIsInstance(result.feature_frame, FeatureFrame)
        self.assertEqual(result.feature_frame.modality, "acoustic")
        self.assertEqual(result.feature_frame.source_id, "sensor_01")

        # Summary dict
        summary = result.to_summary_dict()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["source_id"], "sensor_01")
        self.assertAlmostEqual(summary["dominant_frequency_hz"], 440.0, delta=2.0)

    def test_pipeline_with_bandpass_filter(self):
        """Verify pipeline execution with active band-pass filtering."""
        f_cfg = FilterConfig(
            filter_type="bandpass",
            low_cutoff=300.0,
            high_cutoff=600.0,
            order=4,
            enabled=True,
        )
        pipeline = AcousticPipeline(sample_rate=self.sr, filter_config=f_cfg)
        result = pipeline.process(self.tone_440, source_id="filtered_test")

        self.assertTrue(result.filtered)
        self.assertEqual(result.filter_type, "bandpass")
        # 440 Hz is inside [300, 600] passband, so dominant frequency should be preserved
        self.assertAlmostEqual(result.features["dominant_frequency"], 440.0, delta=2.0)

    def test_pipeline_with_filter_disabled(self):
        """Verify filter is bypassed when FilterConfig.enabled is False."""
        f_cfg = FilterConfig(
            filter_type="lowpass",
            cutoff_freq=200.0,
            enabled=False,
        )
        pipeline = AcousticPipeline(sample_rate=self.sr, filter_config=f_cfg)
        result = pipeline.process(self.tone_440)

        self.assertFalse(result.filtered)
        self.assertIsNone(result.filter_type)
        np.testing.assert_array_equal(result.raw_signal, result.processed_signal)

    def test_pipeline_with_lowpass_filter(self):
        """Verify low-pass filtering stage attenuates high-frequency tone."""
        # 4000 Hz high tone
        high_tone = generate_sine_wave(4000.0, duration=self.duration, sample_rate=self.sr, amplitude=0.8)
        f_cfg = FilterConfig(filter_type="lowpass", cutoff_freq=1000.0, order=4)
        pipeline = AcousticPipeline(sample_rate=self.sr, filter_config=f_cfg)
        result = pipeline.process(high_tone)

        self.assertTrue(result.filtered)
        self.assertEqual(result.filter_type, "lowpass")
        # RMS energy should be drastically attenuated
        raw_rms = np.sqrt(np.mean(result.raw_signal**2))
        proc_rms = np.sqrt(np.mean(result.processed_signal**2))
        self.assertLess(proc_rms, 0.05 * raw_rms)

    def test_pipeline_with_highpass_filter(self):
        """Verify high-pass filtering stage attenuates low-frequency tone."""
        # 100 Hz low rumble
        low_tone = generate_sine_wave(100.0, duration=self.duration, sample_rate=self.sr, amplitude=0.8)
        f_cfg = FilterConfig(filter_type="highpass", cutoff_freq=500.0, order=4)
        pipeline = AcousticPipeline(sample_rate=self.sr, filter_config=f_cfg)
        result = pipeline.process(low_tone)

        self.assertTrue(result.filtered)
        self.assertEqual(result.filter_type, "highpass")
        raw_rms = np.sqrt(np.mean(result.raw_signal**2))
        proc_rms = np.sqrt(np.mean(result.processed_signal**2))
        self.assertLess(proc_rms, 0.05 * raw_rms)

    def test_pipeline_harmonic_analysis_integration(self):
        """Verify harmonic analysis when base_frequency is provided to pipeline."""
        fan_audio = generate_fan_noise(
            base_frequency=120.0,
            duration=self.duration,
            sample_rate=self.sr,
            harmonics=(1.0, 0.6, 0.3),
            noise_level=0.02,
            seed=42,
        )
        pipeline = AcousticPipeline(sample_rate=self.sr)
        result = pipeline.process(fan_audio, base_frequency=120.0)

        self.assertIsNotNone(result.harmonics)
        self.assertIsInstance(result.harmonics, dict)
        self.assertIn(1, result.harmonics)
        self.assertIsNotNone(result.harmonics[1])
        self.assertAlmostEqual(result.harmonics[1], 120.0, delta=5.0)

    def test_stereo_input_channel_downmix(self):
        """Verify 2D multi-channel inputs are automatically averaged to mono."""
        left = generate_sine_wave(440.0, duration=self.duration, sample_rate=self.sr, amplitude=0.5)
        right = generate_sine_wave(440.0, duration=self.duration, sample_rate=self.sr, amplitude=0.5)
        stereo = np.column_stack([left, right])

        pipeline = AcousticPipeline(sample_rate=self.sr)
        result = pipeline.process(stereo)

        self.assertEqual(result.raw_signal.ndim, 1)
        self.assertEqual(len(result.raw_signal), len(left))

    def test_process_file_round_trip(self):
        """Verify process_file loads a WAV file and runs the full pipeline."""
        wav_path = self.temp_path / "test_fan.wav"
        write_wav(wav_path, self.tone_440, sample_rate=self.sr, subtype="PCM_16")

        pipeline = AcousticPipeline(sample_rate=self.sr)
        result = pipeline.process_file(wav_path)

        self.assertEqual(result.source_id, "test_fan")
        self.assertEqual(result.sample_rate, self.sr)
        self.assertAlmostEqual(result.features["dominant_frequency"], 440.0, delta=2.0)

    def test_custom_spectrogram_config(self):
        """Verify custom SpectrogramConfig modifies spectrogram matrix shape."""
        spec_cfg = SpectrogramConfig(nperseg=512, noverlap=128, window="hamming")
        pipeline = AcousticPipeline(sample_rate=self.sr, spectrogram_config=spec_cfg)
        result = pipeline.process(self.tone_440)

        expected_bins = 512 // 2 + 1
        self.assertEqual(result.spectrogram_matrix.shape[0], expected_bins)

    def test_invalid_pipeline_init_parameters(self):
        """Verify validation of pipeline constructor arguments."""
        with self.assertRaises(ValueError):
            AcousticPipeline(sample_rate=-44100)
        with self.assertRaises(TypeError):
            AcousticPipeline(sample_rate=True)
        with self.assertRaises(TypeError):
            AcousticPipeline(filter_config="invalid")
        with self.assertRaises(TypeError):
            AcousticPipeline(spectrogram_config="invalid")
        with self.assertRaises(ValueError):
            AcousticPipeline(fft_window="unsupported")
        with self.assertRaises(ValueError):
            AcousticPipeline(max_peaks=0)
        with self.assertRaises(ValueError):
            AcousticPipeline(peak_threshold_rel=-0.1)

    def test_filter_config_validation(self):
        """Verify FilterConfig parameter validation."""
        f_cfg = FilterConfig(filter_type="invalid_type")
        with self.assertRaises(ValueError):
            f_cfg.validate(sample_rate=44100)

        f_cfg2 = FilterConfig(filter_type="bandpass", low_cutoff=None, high_cutoff=1000.0)
        with self.assertRaises(ValueError):
            f_cfg2.validate(sample_rate=44100)

        f_cfg3 = FilterConfig(filter_type="bandpass", low_cutoff=1000.0, high_cutoff=500.0)
        with self.assertRaises(ValueError):
            f_cfg3.validate(sample_rate=44100)

        f_cfg4 = FilterConfig(filter_type="lowpass", cutoff_freq=-10.0)
        with self.assertRaises(ValueError):
            f_cfg4.validate(sample_rate=44100)

    def test_spectrogram_config_validation(self):
        """Verify SpectrogramConfig parameter validation."""
        with self.assertRaises(ValueError):
            SpectrogramConfig(nperseg=0).validate()
        with self.assertRaises(ValueError):
            SpectrogramConfig(nperseg=512, noverlap=512).validate()
        with self.assertRaises(ValueError):
            SpectrogramConfig(window="invalid").validate()

    def test_invalid_audio_inputs(self):
        """Verify pipeline rejects empty, non-finite, non-floating, or too-short inputs."""
        pipeline = AcousticPipeline(sample_rate=self.sr)

        # Empty array
        with self.assertRaises(ValueError):
            pipeline.process(np.array([], dtype=np.float64))

        # Non-floating dtype
        with self.assertRaises(ValueError):
            pipeline.process(np.ones(2000, dtype=np.int32))

        # Non-finite values
        with self.assertRaises(ValueError):
            pipeline.process(np.array([1.0, np.nan, 3.0] * 1000, dtype=np.float64))

        # 3D array
        with self.assertRaises(ValueError):
            pipeline.process(np.ones((100, 2, 2), dtype=np.float64))

        # Audio shorter than nperseg
        with self.assertRaises(ValueError):
            pipeline.process(np.ones(500, dtype=np.float64))  # default nperseg is 1024

        # Non-string or empty source_id
        with self.assertRaises(ValueError):
            pipeline.process(self.tone_440, source_id="")

        # Non-datetime timestamp
        with self.assertRaises(TypeError):
            pipeline.process(self.tone_440, timestamp="2026-11-26")


if __name__ == "__main__":
    unittest.main()
