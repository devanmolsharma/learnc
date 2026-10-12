"""Tune the JPEG quantization tables of stb_image_write.h (unmodified) for smaller files at the same quality.

Usage:
    python tune.py STB_DIR KODAK_DIR [--steps N] [--pairs P] [--start stock|mozjpeg] [--out DIR]

STB_DIR: folder with stb_image_write.h and stb_image.h (https://github.com/nothings/stb).
KODAK_DIR: the 24 Kodak images kodim01.png .. kodim24.png (https://r0k.us/graphics/kodak/).

Tuning uses 256x256 crops of the even-numbered images; the odd-numbered images are only used for the test.
--start mozjpeg begins from mozjpeg's default tables instead of stb's own.
The test reports BD-rate: the average change in file size at equal quality over JPEG quality 20 to 90 (negative
is smaller), measured by SSIM (Y, Cb, Cr weighted 6:1:1) and by PSNR of Y.
"""
import argparse, ctypes, glob, json, os, subprocess, tempfile, time
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import PchipInterpolator
import learnc
from learnc.lift import apply

HERE = os.path.dirname(os.path.abspath(__file__))
CROP, QUALITIES = 256, list(range(20, 91, 10))
DEFINES = ["-DSTB_IMAGE_WRITE_IMPLEMENTATION", "-DSTB_IMAGE_IMPLEMENTATION"]
MOZJPEG = {   # mozjpeg's tables (jcparam.c): its default (index 3) and its MS-SSIM table tuned on Kodak (index 2)
    "mozjpeg default tables": json.load(open(os.path.join(HERE, "mozjpeg_tables.json")))["mozjpeg_default (Robidoux)"],
    "mozjpeg MS-SSIM tables": json.load(open(os.path.join(HERE, "mozjpeg_tables.json")))["mozjpeg MS-SSIM (tuned on Kodak)"],
}


def load(path): return np.ascontiguousarray(np.array(Image.open(path).convert("RGB")))


class Encoder:
    """score.c built against one version of stb_image_write.h."""
    def __init__(self, header_dir, stb_dir, tmp):
        out = os.path.join(tmp, f"enc{abs(hash(header_dir))}.so")
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", *DEFINES, "-I", header_dir, "-I", stb_dir,
                        os.path.join(HERE, "score.c"), "-o", out, "-lm"], check=True)
        self.lib = ctypes.CDLL(out)

    def measure(self, img, quality):
        out = (ctypes.c_double * 3)()
        if self.lib.measure(img.ctypes.data_as(ctypes.c_void_p), img.shape[1], img.shape[0], quality, out):
            raise RuntimeError("encoding failed")
        return tuple(out)


def bd_rate(rate_a, q_a, rate_b, q_b):
    """Bjontegaard delta rate of b against a, in percent: average log-rate difference over the shared quality range."""
    ia, ib = np.argsort(q_a), np.argsort(q_b)
    lo, hi = max(min(q_a), min(q_b)), min(max(q_a), max(q_b))
    grid = np.linspace(lo, hi, 100)
    la = PchipInterpolator(np.asarray(q_a)[ia], np.log(np.asarray(rate_a))[ia])(grid)
    lb = PchipInterpolator(np.asarray(q_b)[ib], np.log(np.asarray(rate_b))[ib])(grid)
    return (np.exp(np.mean(lb - la)) - 1) * 100


def curves(enc, images):
    """per image: (bpp, SSIM in dB, PSNR) for every quality."""
    out = []
    for img in images:
        m = np.array([enc.measure(img, q) for q in QUALITIES])
        out.append((m[:, 0], -10 * np.log10(m[:, 1]), m[:, 2]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stb"); ap.add_argument("kodak")
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--start", choices=["stock", "mozjpeg"], default="stock")
    ap.add_argument("--out", default=os.path.join(HERE, "tuned"))
    a = ap.parse_args()
    header = os.path.join(os.path.abspath(a.stb), "stb_image_write.h")
    paths = sorted(glob.glob(os.path.join(a.kodak, "kodim*.png")))
    train = [load(p) for p in paths[1::2]]                                # kodim02, 04, ... 24
    test = [load(p) for p in paths[0::2]]                                 # kodim01, 03, ... 23

    with tempfile.TemporaryDirectory() as tmp:
        stock = Encoder(a.stb, a.stb, tmp)
        rng = np.random.default_rng(0); crops, qs, bpp, dist = [], [], [], []
        for img in train:                                                 # 12 crops per image, random qualities
            for _ in range(12):
                y, x = rng.integers(0, img.shape[0] - CROP + 1), rng.integers(0, img.shape[1] - CROP + 1)
                c = np.ascontiguousarray(img[y:y + CROP, x:x + CROP])
                curve = np.array([stock.measure(c, q) for q in QUALITIES])  # the stock tables' size/quality curve
                crops.append(c.reshape(-1)); qs.append(int(rng.integers(20, 91)))
                bpp.append(curve[:, 0]); dist.append(curve[:, 1])
        inputs = [torch.from_numpy(np.stack(crops)), torch.tensor(qs, dtype=torch.int32),
                  torch.from_numpy(np.stack(bpp)), torch.from_numpy(np.stack(dist))]
        print(f"{len(crops)} training crops from {len(train)} images; tuning YQT and UVQT (128 numbers)", flush=True)

        start = header
        if a.start == "mozjpeg":                                          # tune on top of mozjpeg's default tables
            d = os.path.join(tmp, "start"); os.makedirs(d); start = os.path.join(d, "stb_image_write.h")
            apply(header, MOZJPEG["mozjpeg default tables"], start, DEFINES)
        t0 = time.time()
        r = learnc.tune(os.path.join(HERE, "score.c"), inputs, entry="crop_cost", constants={start: ["YQT", "UVQT"]},
                        flags=["-I", a.stb, *DEFINES], steps=a.steps, batch=48, pairs=a.pairs, restarts=1, workers=4, verbose=True)
        print(f"tuned in {time.time() - t0:.0f}s\n{r.report()}", flush=True)
        r.write(a.out)

        versions = {"stb (JPEG Annex K tables)": a.stb, "tuned by learnc": a.out}
        for name, tables in MOZJPEG.items():                              # the same encoder with mozjpeg's tables
            d = os.path.join(tmp, name.replace(" ", "_")); os.makedirs(d)
            apply(header, tables, os.path.join(d, "stb_image_write.h"), DEFINES)
            versions[name] = d
        results = {name: curves(Encoder(d, a.stb, tmp), test) for name, d in versions.items()}
        base = results["stb (JPEG Annex K tables)"]
        print(f"\nheld-out Kodak images ({len(test)}), BD-rate against stb's stock tables (negative = smaller files):")
        for name, res in results.items():
            ssim = np.mean([bd_rate(b[0], b[1], r_[0], r_[1]) for b, r_ in zip(base, res)])
            psnr = np.mean([bd_rate(b[0], b[2], r_[0], r_[2]) for b, r_ in zip(base, res)])
            print(f"  {name:28s} at equal SSIM {ssim:+6.2f}%   at equal PSNR {psnr:+6.2f}%", flush=True)


if __name__ == "__main__":
    main()
