# Microgrid PPO Edge Firmware

PlatformIO project that runs the quantised PPO policy on an ESP32-WROOM-32
development board. The firmware reads four analog channels (battery SOC,
demand, PV, tariff) and writes a binary charge/discharge GPIO decision at
1 Hz.

## Build

```bash
pio run -e esp32dev          # compile
pio run -e esp32dev -t upload  # flash
pio device monitor -b 115200  # serial monitor
```

## Files

| Path | Role |
|------|------|
| `platformio.ini` | Board / toolchain / library configuration. |
| `src/main.cpp` | Edge loop: sensor read + policy invocation + GPIO write. |
| `src/policy_inference.cpp` | TFLite Micro interpreter wrapper. |
| `include/policy_inference.h` | Inference API header. |
| `include/ppo_model_data.h` | Auto-generated declaration of the model byte array (produced by `src/quantize.py`). |
| `include/ppo_model_data.cc` | Auto-generated model byte array (produced by `src/quantize.py`). |

## Bench measurements

The serial monitor emits one CSV-like line per inference. The reference
benchmark captured on the bench harness is preserved at
[`../output/logs/esp32_serial.log`](../output/logs/esp32_serial.log) and
summarised in [`../output/results.csv`](../output/results.csv).
