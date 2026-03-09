# EV-Telemetry-Event-Analytics


## Overview
Pipeline to extract **IgnitionEvents** and **ChargingEvents** from EV telematics and trigger data.

## Quick Start
```bash
pip install pandas numpy
python main.py
```

Outputs are written to `output/`:
- `IgnitionEvents.csv` — every ignition ON/OFF transition
- `ChargingEvents.csv` — detected battery-charging sessions
- `ChargingStatusEvents.csv` — charge-plug status events
- `data_quality_report.txt` — all anomalies discovered

---

## Data Sources & Discovered Schemas

| Tag | File | Key Columns | Notes |
|-----|------|-------------|-------|
| **TLM** | `telemetry_data.csv` | `VEHICLE_ID, TIMESTAMP, SPEED, IGNITION_STATUS, EV_BATTERY_LEVEL, ODOMETER` | 1.8M rows, 16 vehicles. Sparse — many columns are >50% null. |
| **TRG** | `triggers_soc.csv` | `CTS, PNID, NAME, VAL` | 68K rows, 20 PNIDs. NAME ∈ {IGN_CYL, CHARGE_STATE, EV_CHARGE_STATE}. Timestamps carry IST+0530. |
| **MAP** | `vehicle_pnid_mapping.csv` | `ID, IDS` | Maps 16 vehicle UUIDs to PNID lists. Some vehicles have no PNIDs. |
| **SYN** | `artificial_ign_off_data.json` | `vehicleId, timestamp, type` | 411 synthetic ignition-off events for 11 vehicles. |

## Data Issues Found

1. **TLM: `IGNITION_STATUS = 'Unknown'`** — treated as missing.
2. **TLM: Battery > 100%** — capped at 100.
3. **TLM: High sparsity** — SPEED (80%), IGNITION_STATUS (87%), BATTERY (78%), ODOMETER (73%) null.
4. **TRG: Timestamp format** — contains `IST+0530` timezone suffix; parsed and normalised.
5. **TRG: Duplicate rows** — deduplicated on (CTS, PNID, NAME, VAL).
6. **TRG: `EV_CHARGE_STATE` values** — `Aborted` normalised to `Abort`.
7. **MAP: Missing PNIDs** — some vehicles have `NaN` in IDS column (no TRG linkage).
8. **MAP: Duplicate vehicle entries** — deduplicated keeping last.
9. **SYN: Vehicle scope mismatch** — 7 of 11 SYN vehicles are not in TLM.

## Design Choices

### Ignition Event Extraction
- **TLM**: Detect status *transitions* (`on→off` or `off→on`) by comparing consecutive rows per vehicle.
- **TRG**: Same transition logic on `IGN_CYL` values, mapped to vehicle IDs via MAP.
- **SYN**: All 411 records treated as `ignitionOff` events directly.
- **Fusion**: Merge all three, then deduplicate events within 10 s for the same vehicle+event type. Priority: TLM > TRG > SYN.

### Battery Association
- Within **±300 seconds** of each ignition/charging-status event.
- **Tie-breaker**: prefer the reading *after* the event (most up-to-date state).
- If no reading within the window → `unknown`.

### Charging-Event Detection
- Walk through consecutive candidate events per vehicle.
- A "real" charge = battery rise exceeding a threshold:
  - **Ignition OFF**: ≥ 2% rise (plug-in charging is expected).
  - **Ignition ON**: ≥ 5% rise (must be stricter — regenerative braking can cause small rises, so we need a higher bar to distinguish real charging).
- **Debouncing**:
  - Sessions separated by < 5 min are merged into one continuous charge.
  - Sessions < 60 s total duration are discarded as noise.
## Dataset

The dataset is hosted externally.

Dataset link:  
https://drive.google.com/file/d/1Fap7vvtLjEDkh2lNIOZKV76jpf6Gtl9T/view?usp=drive_link

Download and extract the dataset before running the pipeline.
