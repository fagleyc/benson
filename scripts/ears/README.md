# EARS — Phase 0 flash checklist

You're flashing the **XIAO ESP32-S3** half of the ReSpeaker Lite. The XMOS
half auto-DFUs itself on first boot of the ESP32 firmware (you'll see the
LED flash yellow → green during install — that's normal, don't unplug).

## What you need
- A laptop with Python 3.10+ and Chrome (for any debug step)
- The included USB-C cable (or any data-grade USB-C)
- Your home WiFi SSID + password

## One-time laptop setup
```bash
pipx install esphome              # or: pip install --user esphome
# (alternative: docker run --rm -v "${PWD}":/config ghcr.io/esphome/esphome ...)
```

## Pull the YAML to your laptop
```bash
scp casey@192.168.0.240:/opt/benson/scripts/ears/respeaker-kitchen.yaml .
scp casey@192.168.0.240:/opt/benson/scripts/ears/secrets.template.yaml ./secrets.yaml
```
Edit `secrets.yaml`:
- `wifi_ssid` / `wifi_password` — your home network
- `api_encryption_key` — run `openssl rand -base64 32` and paste the output
- `wifi_fallback_password` — any 8+ char string, only used if home WiFi dies

## Flash
1. Plug the USB-C cable into the **XIAO ESP32-S3 port** (the smaller board
   piggybacked on top — NOT the XMOS-board port on the bottom).
2. Plug the other end into the laptop.
3. From the directory with both YAML files:
   ```bash
   esphome run respeaker-kitchen.yaml
   ```
4. Pick "Pick specific port" and choose the device (usually `/dev/cu.usbmodem*`
   on Mac, `COM*` on Windows, `/dev/ttyACM0` on Linux).
5. First flash takes ~3–5 min (compile + upload). Subsequent OTAs are seconds.

## On first boot
- LED ring will go through:
  - solid white → booting
  - yellow flash → installing XMOS DFU firmware (~30 sec; do NOT unplug)
  - green → XMOS done, ready
  - blue pulse → connected to WiFi + waiting for HA
- If WiFi fails, the device will broadcast `Respeaker-Kitchen Fallback`
  (password: whatever you put in `wifi_fallback_password`). Connect, browser
  to `http://192.168.4.1`, fix WiFi creds.

## Verify
Once it's on WiFi, it should appear in HA at:
**Settings → Devices & Services → ESPHome → Discovered**
Click "Configure," paste the `api_encryption_key` you set in secrets.yaml,
and adopt it.

## When it's adopted in HA, ping me
I'll wire up wyoming-faster-whisper + the Assist Pipeline + Benson
conversation bridge (Phases 0.5 + 0.75) at that point.
