/* Scoring for tuning the JPEG quantization tables of stb_image_write.h (unmodified) with learnc.tune.
   Each image is encoded in memory, decoded with stb_image.h, and compared with the original by SSIM on Y, Cb and
   Cr (weighted 6:1:1, as in video codec testing). Any JPEG decoder reads the result: the tables travel in the file. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "stb_image_write.h"
#include "stb_image.h"

#define CROP 256                                   /* training examples are 256x256 crops */

typedef struct { unsigned char *data; int size, cap; } buffer;

static void append(void *context, void *data, int size) {
    buffer *b = context;
    if (b->size + size > b->cap) { b->cap = 2 * (b->size + size); b->data = realloc(b->data, b->cap); }
    memcpy(b->data + b->size, data, size); b->size += size;
}

static void to_ycbcr(const unsigned char *rgb, int n, float *y, float *cb, float *cr) {
    for (int i = 0; i < n; i++) {
        float r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
        y[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        cb[i] = 128 - 0.168736f * r - 0.331264f * g + 0.5f * b;
        cr[i] = 128 + 0.5f * r - 0.418688f * g - 0.081312f * b;
    }
}

/* mean SSIM over 8x8 windows with stride 4 */
static double ssim(const float *a, const float *b, int w, int h) {
    const double c1 = 6.5025, c2 = 58.5225;       /* (0.01 * 255)^2, (0.03 * 255)^2 */
    double total = 0; int count = 0;
    for (int y = 0; y + 8 <= h; y += 4)
        for (int x = 0; x + 8 <= w; x += 4) {
            double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
            for (int j = 0; j < 8; j++)
                for (int i = 0; i < 8; i++) {
                    double p = a[(y + j) * w + x + i], q = b[(y + j) * w + x + i];
                    sa += p; sb += q; saa += p * p; sbb += q * q; sab += p * q;
                }
            double ma = sa / 64, mb = sb / 64;
            double va = saa / 64 - ma * ma, vb = sbb / 64 - mb * mb, cov = sab / 64 - ma * mb;
            total += (2 * ma * mb + c1) * (2 * cov + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            count++;
        }
    return total / count;
}

/* encode at `quality`; out[0] = bits per pixel, out[1] = distortion (1 - weighted SSIM), out[2] = PSNR of Y in dB.
   Returns 0 on success. */
int measure(const unsigned char *rgb, int w, int h, int quality, double out[3]) {
    buffer b = {0};
    if (!stbi_write_jpg_to_func(append, &b, w, h, 3, rgb, quality)) return 1;
    int dw, dh, dc;
    unsigned char *dec = stbi_load_from_memory(b.data, b.size, &dw, &dh, &dc, 3);
    if (!dec) { free(b.data); return 1; }
    int n = w * h;
    float *p = malloc(sizeof(float) * n * 6);
    to_ycbcr(rgb, n, p, p + n, p + 2 * n);
    to_ycbcr(dec, n, p + 3 * n, p + 4 * n, p + 5 * n);
    double s = (6 * ssim(p, p + 3 * n, w, h) + ssim(p + n, p + 4 * n, w, h) + ssim(p + 2 * n, p + 5 * n, w, h)) / 8;
    double mse = 0;
    for (int i = 0; i < n; i++) mse += (p[i] - p[3 * n + i]) * (p[i] - p[3 * n + i]);
    out[0] = 8.0 * b.size / n;
    out[1] = 1 - s;
    out[2] = 10 * log10(255.0 * 255.0 / (mse / n + 1e-10));
    free(p); free(dec); free(b.data);
    return 0;
}

#define CURVE 8                                    /* stock tables measured at quality 20, 30, ..., 90 */

/* tuning entry: log of (file size) / (size the stock tables need for the same SSIM), read off the stock tables'
   size/quality curve for this crop (linear in log-log, extended at the ends). This is the BD-rate of one point:
   minimising it means smaller files at equal quality, whatever the quality setting does. */
double crop_cost(const unsigned char rgb[CROP * CROP * 3], int quality, const double stock_bpp[CURVE],
                 const double stock_distortion[CURVE]) {
    double m[3];
    if (measure(rgb, CROP, CROP, quality, m)) return 10;
    double d = log(m[1]);
    int k = 0;                                     /* stock distortion falls as quality rises: find the segment */
    while (k < CURVE - 2 && log(stock_distortion[k + 1]) > d) k++;
    double d0 = log(stock_distortion[k]), d1 = log(stock_distortion[k + 1]);
    double t = (d - d0) / (d1 - d0);
    double stock_log_bpp = log(stock_bpp[k]) + t * (log(stock_bpp[k + 1]) - log(stock_bpp[k]));
    return log(m[0]) - stock_log_bpp;
}
