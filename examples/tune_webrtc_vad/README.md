# Tuning Google's WebRTC VAD

`score.c` calls Google's unmodified `vad_core.c`. `chunk_error` returns the share of 30 ms frames that the VAD gets
wrong in one 30 s chunk, and `learnc.tune` minimises it over LibriParty dev. `tune.py` then compares Google's
constants with the tuned ones on LibriParty eval.

```bash
git clone https://github.com/wiseman/py-webrtcvad
pip install soundfile
python tune.py py-webrtcvad/cbits /path/to/LibriParty/dataset          # 8 named constants
python tune.py py-webrtcvad/cbits /path/to/LibriParty/dataset --all    # every constant of vad_core.c
```

The tuned `vad_core.c` and `TUNING_REPORT.txt` are written to `tuned/`. Keep the data at 300 chunks (the default)
to stay under 2 GB of memory.
