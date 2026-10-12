"""Tune Google's unmodified WebRTC VAD with learnc.tune: tune on LibriParty dev, test on LibriParty eval.

Usage:
    python tune.py WEBRTC_CBITS LIBRIPARTY_DATASET [--steps N] [--all] [--out DIR]

WEBRTC_CBITS: the cbits folder of https://github.com/wiseman/py-webrtcvad (Google's C sources).
LIBRIPARTY_DATASET: the LibriParty `dataset` folder (with dev/, eval/ and metadata/).
--all tunes every constant of vad_core.c; the default tunes 8 named ones.
"""
import argparse, ctypes, glob, json, os, subprocess, tempfile, time
import numpy as np
import soundfile as sf
import torch
import learnc

HERE = os.path.dirname(os.path.abspath(__file__))
F30, FRAMES = 480, 1000                                    # 30 ms frames at 16 kHz, 30 s chunks
NAMED = ["kLocalThresholdQ", "kGlobalThresholdQ", "kSpectrumWeight", "kNoiseUpdateConst", "kSpeechUpdateConst",
         "kBackEta", "kOverHangMax1Q", "kOverHangMax2Q"]


def session_labels(meta, n_frames, step):
    """1 for every frame whose centre falls inside an utterance."""
    centre = (np.arange(n_frames) * step + step // 2) / 16000
    y = np.zeros(n_frames, np.int8)
    for sub, entries in meta.items():
        if sub in ("noises", "background"): continue
        for e in entries: y[(centre >= e["start"]) & (centre < e["stop"])] = 1
    return y


def read_pcm(root, split, name):
    x, _ = sf.read(f"{root}/{split}/{name}/{name}_mixture.wav", dtype="float32")
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def dev_chunks(root, limit=300):
    """30 s chunks of the dev sessions with 30 ms labels (300 chunks = 2.5 h keeps memory under 2 GB)."""
    meta = json.load(open(f"{root}/metadata/dev.json")); X, Y = [], []
    for name in sorted(meta):
        pcm = read_pcm(root, "dev", name)
        y = session_labels(meta[name], len(pcm) // F30, F30)
        for c in range(len(y) // FRAMES):
            X.append(pcm[c * FRAMES * F30:(c + 1) * FRAMES * F30]); Y.append(y[c * FRAMES:(c + 1) * FRAMES])
    keep = np.random.default_rng(0).permutation(len(X))[:limit]
    return torch.from_numpy(np.stack([X[i] for i in keep])), torch.from_numpy(np.stack([Y[i] for i in keep]))


def build(core, others, flags, out):
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", *flags, os.path.join(HERE, "score.c"), core, *others,
                    "-o", out, "-lm"], check=True, capture_output=True)
    return ctypes.CDLL(out)


def evaluate(lib, root):
    """F1 and accuracy on LibriParty eval, scored on 32 ms frames as in Silero's benchmark."""
    meta = json.load(open(f"{root}/metadata/eval.json")); P, Y = [], []
    for name in sorted(meta):
        pcm = read_pcm(root, "eval", name)
        frames = len(pcm) // F30; d = np.zeros(frames, np.int8)
        lib.decide_stream(np.ascontiguousarray(pcm[:frames * F30]).ctypes.data_as(ctypes.c_void_p), frames,
                          d.ctypes.data_as(ctypes.c_void_p))
        y = session_labels(meta[name], len(pcm) // 512, 512)
        P.append(d[np.minimum((np.arange(len(y)) * 512 + 256) // F30, frames - 1)]); Y.append(y)
    p, y = np.concatenate(P), np.concatenate(Y)
    tp, fp, fn = ((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum()
    return 2 * tp / (2 * tp + fp + fn), (p == y).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cbits"); ap.add_argument("libriparty")
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "tuned"))
    a = ap.parse_args()
    vad = os.path.join(a.cbits, "webrtc/common_audio")
    core = os.path.join(vad, "vad/vad_core.c")
    others = [f for f in glob.glob(f"{vad}/signal_processing/*.c") + glob.glob(f"{vad}/vad/*.c")
              + glob.glob(f"{vad}/third_party/spl_sqrt_floor/*.c") if not f.endswith(("vad_core.c", "webrtc_vad.c"))]
    others.append(os.path.join(HERE, "checks_stub.c"))
    flags = ["-I", a.cbits, "-DWEBRTC_POSIX"]

    x, y = dev_chunks(a.libriparty)
    print(f"{len(x)} dev chunks of 30 s, speech share {y.float().mean():.2f}", flush=True)
    t0 = time.time()
    result = learnc.tune(os.path.join(HERE, "score.c"), [x, y], entry="chunk_error", sources=[core] + others,
                         flags=flags, constants={core: "all" if a.all else NAMED}, steps=a.steps, batch=48,
                         workers=4, verbose=True)
    print(f"tuned in {time.time() - t0:.0f}s\n{result.report()}", flush=True)
    tuned_core = result.write(a.out)[0]

    with tempfile.TemporaryDirectory() as tmp:
        for name, c in (("Google's constants", core), ("tuned on dev", tuned_core)):
            f1, acc = evaluate(build(c, others, flags, os.path.join(tmp, f"{len(name)}.so")), a.libriparty)
            print(f"{name:20s} LibriParty eval: F1 {f1:.3f}  accuracy {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
