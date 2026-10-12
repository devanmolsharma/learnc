# Smaller JPEGs from better quantization tables

`learnc tune` tunes the two quantization tables (`YQT`, `UVQT`: 128 numbers) inside the unmodified
[stb_image_write.h](https://github.com/nothings/stb) JPEG encoder. The tables are stored in every JPEG file, so the
results decode in any JPEG decoder (browsers, phones, image libraries) without changes.

## Results

Tuned on 256x256 crops of 12 Kodak images (the even-numbered ones), tested on the other 12. Each number is the
BD-rate against the standard JPEG Annex K tables: the average change in file size at equal quality, over JPEG
quality 20 to 90. Negative means smaller files.

Checked independently with a different encoder (libjpeg, through Pillow) and standard metrics (scikit-image SSIM,
RGB PSNR), using `verify_libjpeg.py`:

| tables | at equal SSIM | at equal RGB PSNR |
|---|---|---|
| JPEG Annex K (stb, libjpeg default) | 0 | 0 |
| mozjpeg default (Robidoux) | -2.7% | -6.4% |
| mozjpeg MS-SSIM (tuned on Kodak) | -2.2% | -7.9% |
| **tuned by learnc** | **-9.1%** | **-12.2%** |

By bitrate, at equal SSIM:

| stock bitrate | tuned by learnc | mozjpeg default |
|---|---|---|
| 0.3 to 1.0 bits per pixel | +0.3% | -1.8% |
| 1.0 to 2.0 bits per pixel | -9.5% | -3.3% |
| 2.0 to 4.0 bits per pixel | -12.8% | -2.2% |

The gain is at medium and high quality, where most web JPEGs are. At very low bitrates there is none, because
training only covered quality settings 20 to 90.

The tuned tables are finer than the stock ones, so a given quality number gives a larger, better image than before.
Pick the quality by the size or quality you need, not by the old number.

## Using the tables

`tuned_tables.json` holds both tables in natural (row-major) order. With Pillow:

```python
import json
t = json.load(open("tuned_tables.json"))
scale = lambda table, q: [min(max((v * (5000 // q if q < 50 else 200 - 2 * q) + 50) // 100, 1), 255) for v in table]
image.save("out.jpg", qtables=[scale(t["YQT"], 85), scale(t["UVQT"], 85)], subsampling=2)
```

With stb_image_write.h, `tune.py` writes the tuned header to `tuned/` (only the two tables differ).

## How it works

`score.c` encodes a crop in memory, decodes it with stb_image.h, and measures SSIM on Y, Cb and Cr (weighted 6:1:1).
Its cost is the log of the file size divided by the size the stock tables need for the same SSIM on that crop (read
off the stock tables' curve). That is the BD-rate of one point, so minimising it means smaller files at equal
quality, whatever each quality setting maps to.

## Run it

```bash
git clone https://github.com/nothings/stb
mkdir kodak && for i in $(seq -w 1 24); do curl -sLo kodak/kodim$i.png https://r0k.us/graphics/kodak/kodak/kodim$i.png; done
pip install pillow scipy scikit-image
python tune.py stb kodak                       # about 20 minutes on a laptop CPU, under 2 GB of memory
python verify_libjpeg.py stb kodak             # the independent check
```

`mozjpeg_tables.json` holds two of mozjpeg's tables (from its `jcparam.c`) for the comparison. `--start mozjpeg`
tunes from them instead.
