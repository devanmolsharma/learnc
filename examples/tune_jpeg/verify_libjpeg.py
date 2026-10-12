"""Independent check of the tuned tables: a different encoder (libjpeg, through Pillow) and standard metrics
(scikit-image SSIM with the usual Gaussian window, and PSNR over all three RGB channels) on the held-out images.

Usage:
    python verify_libjpeg.py STB_DIR KODAK_DIR [TUNED_DIR]
"""
import glob, io, json, os, re, sys
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tune import HERE, MOZJPEG, QUALITIES, bd_rate


def tables(header):
    s = open(header).read()
    return {n: [int(v) for v in re.search(n + r"\[\] = \{([^}]*)\}", s).group(1).split(",")] for n in ("YQT", "UVQT")}


def scaled(table, quality):
    """libjpeg / stb quality scaling, clamped to baseline JPEG's 1..255."""
    q = 5000 // quality if quality < 50 else 200 - 2 * quality
    return [min(max((v * q + 50) // 100, 1), 255) for v in table]


def curve(img, t):
    out = []
    for q in QUALITIES:
        b = io.BytesIO()
        Image.fromarray(img).save(b, "JPEG", qtables=[scaled(t["YQT"], q), scaled(t["UVQT"], q)], subsampling=2)
        dec = np.array(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
        ssim = structural_similarity(img, dec, channel_axis=2, gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False, data_range=255)
        out.append((8 * b.tell() / (img.shape[0] * img.shape[1]), -10 * np.log10(1 - ssim),
                    peak_signal_noise_ratio(img, dec, data_range=255)))
    return np.array(out)


def main():
    stb, kodak = sys.argv[1], sys.argv[2]
    tuned = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "tuned")
    test = [np.array(Image.open(p).convert("RGB")) for p in sorted(glob.glob(os.path.join(kodak, "kodim*.png")))[0::2]]
    sets = {"JPEG Annex K (stock)": tables(os.path.join(stb, "stb_image_write.h")), **MOZJPEG,
            "tuned by learnc": tables(os.path.join(tuned, "stb_image_write.h"))}
    res = {name: [curve(img, t) for img in test] for name, t in sets.items()}
    base = res["JPEG Annex K (stock)"]
    print(f"libjpeg, {len(test)} held-out Kodak images, BD-rate against Annex K tables (negative = smaller files):")
    for name, r in res.items():
        s = np.mean([bd_rate(b[:, 0], b[:, 1], x[:, 0], x[:, 1]) for b, x in zip(base, r)])
        p = np.mean([bd_rate(b[:, 0], b[:, 2], x[:, 0], x[:, 2]) for b, x in zip(base, r)])
        print(f"  {name:28s} at equal SSIM {s:+6.2f}%   at equal RGB PSNR {p:+6.2f}%", flush=True)


if __name__ == "__main__":
    main()
