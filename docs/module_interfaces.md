# AetherHound — Module Interfaces

**Purpose:** Define the initial Python contracts between acquisition, processing, anomaly detection, event handling, and the CLI.

**Status:** Proposed integration contract. Field names and basic responsibilities are defined here; signal parameters and feature schemas must be agreed by the relevant module owners before implementation.

## 1. Interface principles

- Keep RF and acoustic processing separate.
- Use explicit modality identifiers: `"rf"` and `"acoustic"`.
- Keep raw samples distinct from processed features.
- Keep RF and acoustic feature dimensions independent.
- Keep model-specific thresholds inside the relevant detector.
- Emit a common `DetectionEvent` from both detectors.
- Do not make the CLI import or depend on model internals.
- Use timestamps consistently. The initial implementation should use timezone-aware UTC timestamps internally; the CLI may display local time.

## 2. Data contracts

The following Python definitions are a starting point. They are not a claim that these files or classes already exist in the repository.

### 2.1 `SignalFrame`

Represents one captured or simulated block of raw signal data.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
import numpy as np
from numpy.typing import NDArray

Modality = Literal["rf", "acoustic"]

@dataclass
class SignalFrame:
    modality: Modality
    timestamp: datetime
    source_id: str
    sample_rate: float
    samples: NDArray
```

| Field | Meaning |
|---|---|
| `modality` | `"rf"` for RF or `"acoustic"` for audio |
| `timestamp` | Time associated with the captured block |
| `source_id` | Stable identifier for the simulator or physical source |
| `sample_rate` | Samples per second |
| `samples` | Raw sample array; complex IQ for RF and audio samples for acoustic |

**Validation expectations**
- `sample_rate` must be positive.
- `source_id` must be non-empty.
- `timestamp` should be timezone-aware.
- RF samples should preserve complex IQ values.
- Acoustic samples should use a documented numeric representation and amplitude convention.

### 2.2 `FeatureFrame`

Represents modality-specific features produced by a processing module.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
import numpy as np
from numpy.typing import NDArray

Modality = Literal["rf", "acoustic"]

@dataclass
class FeatureFrame:
    modality: Modality
    timestamp: datetime
    source_id: str
    features: NDArray
    feature_names: list[str]
```

| Field | Meaning |
|---|---|
| `modality` | Signal modality |
| `timestamp` | Timestamp associated with the source frame |
| `source_id` | Source identifier carried forward from acquisition |
| `features` | Numeric feature array for the relevant model |
| `feature_names` | Names describing the feature columns/dimensions |

**Validation expectations**
- `features` must contain finite numeric values.
- `feature_names` must describe the feature dimensions according to a documented shape convention.
- RF and acoustic feature schemas are allowed to differ.
- The same modality must use a stable feature ordering between training and inference.
- The preprocessing version and feature schema should be documented alongside each trained model.

The exact feature representation is not decided here. The RF and acoustic owners must document their final feature shape and names, and the ML owner must train each model against that exact representation.

### 2.3 `DetectionEvent`

Represents the output of either anomaly detector.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Modality = Literal["rf", "acoustic"]

@dataclass
class DetectionEvent:
    timestamp: datetime
    modality: Modality
    source_id: str
    anomaly_score: float
    is_anomaly: bool
    frequency: float | None = None
```

| Field | Meaning |
|---|---|
| `timestamp` | Time associated with the detection |
| `modality` | `"rf"` or `"acoustic"` |
| `source_id` | Originating signal source |
| `anomaly_score` | Model-specific reconstruction error or anomaly score |
| `is_anomaly` | Decision made using that model's own threshold |
| `frequency` | Relevant frequency in Hz when available; otherwise `None` |

**Important:** `anomaly_score` is not a universal probability. The RF and acoustic models may produce scores on different scales. Do not compare them directly or use one shared threshold.

## 3. Module boundaries

### 3.1 Acquisition modules

Suggested module locations:
- `src/acquisition/rf_source.py`
- `src/acquisition/acoustic_source.py`

Initial interface shape:

```python
class RFSource:
    def read_frame(self) -> SignalFrame:
        """Return the next RF frame from a simulator or SDR adapter."""
        raise NotImplementedError


class AcousticSource:
    def read_frame(self) -> SignalFrame:
        """Return the next audio frame from a simulator or microphone adapter."""
        raise NotImplementedError
```

Both adapters return `SignalFrame`, but the `samples` representation depends on modality.

### 3.2 Processing modules

Suggested module locations:
- `src/processing/rf_processing.py`
- `src/processing/acoustic_processing.py`

Initial interface shape:

```python
def process_rf(frame: SignalFrame) -> FeatureFrame:
    """Validate and process an RF frame into RF model features."""
    raise NotImplementedError


def process_acoustic(frame: SignalFrame) -> FeatureFrame:
    """Validate and process an acoustic frame into audio model features."""
    raise NotImplementedError
```

Each function must reject a frame with the wrong modality. Processing modules own their modality-specific filtering, FFT/spectrogram work, and feature extraction.

### 3.3 Anomaly detector modules

Suggested module locations:
- `src/detection/rf_detector.py`
- `src/detection/acoustic_detector.py`

Initial interface shape:

```python
class RFAnomalyDetector:
    def predict(self, frame: FeatureFrame) -> DetectionEvent:
        """Score RF features using the RF model and RF threshold."""
        raise NotImplementedError


class AcousticAnomalyDetector:
    def predict(self, frame: FeatureFrame) -> DetectionEvent:
        """Score acoustic features using the acoustic model and acoustic threshold."""
        raise NotImplementedError
```

Both detectors return `DetectionEvent`. Each detector owns:
- Its own trained model.
- Its own expected feature schema.
- Its own reconstruction-error calculation.
- Its own anomaly threshold.
- Its own model evaluation and metadata.

The detector must reject a `FeatureFrame` for the wrong modality or an incompatible feature schema.

### 3.4 Event handling

Suggested module location:
- `src/detection/event_manager.py`

Initial interface shape:

```python
def handle_event(event: DetectionEvent) -> None:
    """Forward an event to logging, dashboard state, and alert handling."""
    raise NotImplementedError
```

The event manager should route events without needing to know the internal structure of either autoencoder.

### 3.5 CLI

Suggested module location:
- `src/dashboard/cli.py`

The CLI consumes event data and display state. It should not load or execute the RF/acoustic models directly. Model inference belongs to the detector modules; event routing belongs to the event layer.

## 4. End-to-end data flow

```text
RF source
  └── SignalFrame(modality="rf")
        └── RF processing
              └── FeatureFrame(modality="rf")
                    └── RF anomaly detector
                          └── DetectionEvent(modality="rf")
                                └── Event manager
                                      ├── CLI
                                      ├── Event log
                                      └── Alert output

Acoustic source
  └── SignalFrame(modality="acoustic")
        └── Acoustic processing
              └── FeatureFrame(modality="acoustic")
                    └── Acoustic anomaly detector
                          └── DetectionEvent(modality="acoustic")
                                └── Event manager
                                      ├── CLI
                                      ├── Event log
                                      └── Alert output
```

## 5. Ownership and interface coordination

| Module / contract | Primary owner | Coordination |
|---|---|---|
| RF source and RF `SignalFrame` production | M02 — Sarang | M01 |
| Acoustic source and acoustic `SignalFrame` production | M03 — Piyush | M01 |
| RF processing and RF feature schema | M02 — Sarang | M04 and M01 |
| Acoustic processing and acoustic feature schema | M03 — Piyush | M04 and M01 |
| RF and acoustic model training/inference | M04 — Karthik | M02, M03, M01 |
| Common data contracts and event routing | M01 — Aditya | All members |
| CLI integration | M01 — Aditya | All members |

## 6. Integration acceptance checks

### Acquisition
- Produces repeatable frames.
- Includes modality, timestamp, source ID, sample rate, and samples.
- Simulator and future hardware adapter follow the same interface.

### Processing
- Accepts only the correct modality.
- Produces finite numeric features.
- Documents feature names, shape, ordering, and preprocessing settings.
- Handles invalid or empty input explicitly.

### Anomaly detection
- Loads the correct modality-specific model.
- Rejects incompatible feature schemas.
- Returns a valid `DetectionEvent`.
- Uses its own calibrated threshold.
- Documents training data and evaluation results.

### Event handling and CLI
- Accepts events from either modality.
- Displays modality, source, timestamp, score, and anomaly status.
- Logs timestamp and frequency when available.
- Handles missing frequency values.
- Does not assume RF and acoustic scores are directly comparable.

## 7. Decisions still required from module owners

Do not guess these values in the shared interface:
- Sample rates and frame/window sizes.
- RF IQ data type and scaling convention.
- Acoustic sample format and amplitude convention.
- FFT size, overlap, filters, and spectrogram settings.
- Feature names, dimensions, and ordering.
- Model architecture, training parameters, and checkpoint format.
- Threshold calibration method and values.
- Frequency-estimation method and units.

Each owner should propose and test the values for their module, then update this document before integration.

## 8. Source basis and design notes

The project brief specifies RF and acoustic capture, GNU Radio/Python DSP, FFT/spectrogram processing, an autoencoder trained on normal server emissions, a local CLI, anomaly alerting, and timestamp/frequency logging.

The responsibility matrix assigns RF acquisition to M02, acoustic DSP to M03, ML/anomaly detection to M04, and architecture/interfaces/integration/CLI to M01.

The separate RF and acoustic model arrangement and the Python dataclass contracts in this document are the team's proposed implementation design. They are not quoted requirements from the project brief.
