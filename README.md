# learnc: auto-tune the magic numbers in your C code

[![PyPI](https://img.shields.io/pypi/v/learnc)](https://pypi.org/project/learnc/)
[![Python](https://img.shields.io/pypi/pyversions/learnc)](https://pypi.org/project/learnc/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/devanmolsharma/learnc/blob/main/LICENSE)

**Every C codebase is full of hand-picked constants: thresholds, tables, weights, cutoffs. learnc tunes them from
data and gives you back the same C file with better numbers.** No rewrite, no runtime library, no neural network in
production. Just a smaller diff.

```bash
learnc tune score.c --data examples.pt --tune encoder.h:YQT,UVQT --out tuned/
```

### Results on real, unmodified C code

| Code | What learnc tuned | Result on data it never saw |
|---|---|---|
| **JPEG encoder** ([stb_image_write.h](https://github.com/nothings/stb)) | 128 numbers in its quantization tables | **9.1% smaller files at equal quality** (SSIM), vs 2.7% for mozjpeg's tables. Standard JPEG: every decoder reads it |
| **Google's WebRTC voice activity detector** | every constant in `vad_core.c` | **F1 0.810 to 0.859**, accuracy 0.816 to 0.858 on LibriParty |
| **A 2x2 matrix product** | 84 coefficients | rediscovers a 7-multiplication **Strassen-type algorithm** from examples alone |
| **Sorting networks** | which elements each step compares | finds **provably optimal** networks for 4 and 5 inputs |

Each result runs on a laptop CPU in minutes and is reproducible from [`examples/`](https://github.com/devanmolsharma/learnc/tree/main/examples).

---

## Why learnc

- **Works on code you already have.** Point it at constants by name: file-level, `static` inside a function, or in
  a header. Your files are never touched; tuned copies differ only in the numbers that changed.
- **Any C that compiles.** Pointers, structs, `malloc`, files, `qsort` callbacks, `sscanf`, libraries without
  source. If clang and gcc can build it, learnc can tune it.
- **Optimise what you actually care about.** Your score function returns a number: an error rate, a file size, a
  latency, a match against expected outputs. learnc minimises it.
- **Safe for production.** Held-out validation with early stopping, a pull toward the original values, and a final
  pass that reverts every change that doesn't help. The diff holds only what matters.
- **Zero runtime cost.** The output is plain C with constants. Ship it with any compiler, on any platform.

## Use cases

- **Compression:** JPEG/WebP quantization tables, zlib match parameters, codec rate-control constants
- **Audio and signal processing:** voice activity detection thresholds, noise suppression, filter coefficients
- **Heuristics:** cache sizes, cutoffs in sort and search, scoring weights, retry and backoff constants
- **Control:** PID gains, physics and simulation parameters
- **Embedded and fixed-point code:** Q-format constants tuned as whole numbers, kept in their original types
- **Small models written as C:** rules, decision structures and tiny networks that compile to dependency-free C

## Install

```bash
pip install learnc
```

learnc also needs **clang**, **llvm-link** and **gcc** on your system:

```bash
sudo apt install clang llvm gcc        # Debian / Ubuntu
brew install llvm gcc                  # macOS (add $(brew --prefix llvm)/bin to PATH)
```

Python 3.10+. PyTorch and NumPy are installed with it. To work on learnc itself:

```bash
git clone https://github.com/devanmolsharma/learnc && cd learnc
pip install -e ".[dev]" && pytest
```

## Quick start: tune existing code

1. **See what can be tuned:**

   ```bash
   learnc constants encoder.c --cflags "-I include"
   ```

2. **Write a score function** that takes ONE example and returns a number to minimise (or a result to match):

   ```c
   #include "encoder.h"
   double score(const unsigned char image[256 * 256 * 3], int quality) {
       return compressed_size(image, quality) / reference_size(image, quality);
   }
   ```

3. **Tune and review the diff:**

   ```bash
   learnc tune score.c --data train.pt --tune encoder.c:kTable,kThreshold --source encoder.c --out tuned/
   diff encoder.c tuned/encoder.c
   ```

From Python:

```python
import learnc
r = learnc.tune("score.c", [images, qualities], sources=["encoder.c"],
                constants={"encoder.c": ["kTable", "kThreshold"]})     # or "all"
print(r.report())               # held-out loss before and after, every changed constant
r.write("tuned/")               # your files with the new numbers, plus TUNING_REPORT.txt
```

**Read the diff before you ship it.** learnc optimises exactly what you score. In the WebRTC example it set
`kMaxSpeechFrames` to -2, which makes the detector always use its long hangover. That is valid C and it helped on
held-out data, but it is a change in behaviour.

## Quick start: write programs with learned values

Mark the values you can't work out by hand as `learned`, and learnc fits them to input/output examples:

```c
#include <learnc.h>

learned float w1[32][784];                   // numbers to learn
learned float b1[32];
learned float w2[10][32];
learned float b2[10];

int classify(const float px[784]) {          // the structure is just C
    float h[32];
    for (int j = 0; j < 32; j++) {
        float s = b1[j];
        for (int i = 0; i < 784; i++) s += w1[j][i] * px[i];
        h[j] = s > 0 ? s : 0;
    }
    int best = 0; float best_score = -1e30f;
    for (int d = 0; d < 10; d++) {
        float s = b2[d];
        for (int j = 0; j < 32; j++) s += w2[d][j] * h[j];
        if (s > best_score) { best_score = s; best = d; }
    }
    return best;
}
```

```bash
learnc fit mlp.c --data train.pt --val test.pt     # writes mlp.trained.c
gcc -O2 -c mlp.trained.c                           # learnc is no longer needed
```

96.4% on MNIST after 2.4 minutes on a CPU, and the gcc build of `mlp.trained.c` gives the same answer as the
trained model on all 10,000 test images.

| Declaration | Learns |
|---|---|
| `learned float w[...]` | real numbers: weights, thresholds, coefficients |
| `learned_num int k = 8;` | a whole number tuned smoothly and rounded: sizes, counts, fixed-point constants |
| `learned_in(lo, hi) int k[...]` | a choice among lo..hi: an index, a count, a `switch` case. This learns **which** operation, input or rule to use |
| `learned_of(1, 2, 4, 8) int k;` | a choice from a list of numbers |
| `learned_op(ADD, SUB, MUL) op[3];` | operators (ADD, SUB, MUL, DIV, MAX, MIN, LEFT, RIGHT), applied with `learnc_apply(op[k], a, b)` |

## Examples

| Example | What is learned | Result |
|---|---|---|
| [`tune_jpeg/`](https://github.com/devanmolsharma/learnc/tree/main/examples/tune_jpeg) | JPEG quantization tables of the unmodified stb encoder | 9.1% smaller at equal SSIM (libjpeg, held-out Kodak); 12.2% at equal PSNR |
| [`tune_webrtc_vad/`](https://github.com/devanmolsharma/learnc/tree/main/examples/tune_webrtc_vad) | constants of Google's unmodified WebRTC VAD | F1 0.810 to 0.859 on LibriParty eval |
| [`mnist/`](https://github.com/devanmolsharma/learnc/tree/main/examples/mnist) | 7,850 or 25,450 weights | 92.3% (linear) and 96.4% (MLP) on MNIST |
| [`count/`](https://github.com/devanmolsharma/learnc/tree/main/examples/count) | a counting rule, trained on lengths 1 to 10 | 97.1% exact on lengths 40 to 64; a neural network on the same data gets 0% |
| [`rules/`](https://github.com/devanmolsharma/learnc/tree/main/examples/rules) | which features a 2-of-3 rule reads, directions, thresholds | recovers the hidden rule |
| [`formula/`](https://github.com/devanmolsharma/learnc/tree/main/examples/formula) | inputs and operators of a formula | recovers `(x3 + x0) * (x1 - x2)` exactly in 11 s |
| [`discover/`](https://github.com/devanmolsharma/learnc/tree/main/examples/discover) | Strassen-type matrix product, sorting networks | finds exact and optimal algorithms |

More detail on JPEG, including the results by bitrate: [examples/tune_jpeg](https://github.com/devanmolsharma/learnc/tree/main/examples/tune_jpeg). The gain is at
medium and high quality (1 to 4 bits per pixel); at very low bitrates there is none.

## How it works

1. **Compile to LLVM IR.** clang compiles your program and any sources it uses.
2. **Pick an engine.** `Program` chooses automatically, and `p.native_reason` tells you why:
   - **Differentiable runner:** the IR executes in PyTorch with one example per lane, so gradients are exact.
     Branches, loops and recursion run per example; fixed loop nests run vectorised.
   - **Flat memory runner:** adds byte-addressed memory per example for pointers in data, linked structures,
     `malloc`, structs and function pointers. Gradients are still exact.
   - **Native engine:** for anything else that compiles. Your program is built and run as is; gradients are
     estimated from native runs (central differences, antithetic random perturbations, sampling for choices) in
     forked worker processes, so a crash in C becomes an error message.
3. **Fit.** Adam or L-BFGS for real values. Comparisons that depend on learned values become soft while training,
   and an `if` that depends on them runs both sides and blends them. Whole-number choices are relaxed or sampled,
   then polished exactly.
4. **Write C.** The values are written back into a copy of your file.

The runners are checked against the natively compiled program example by example, including floating-point sums.

## Command line

```bash
learnc constants lib.c                   # constants of existing code that can be tuned
learnc tune entry.c --data d.pt --tune lib.c:NAME,NAME --source lib.c [--minimize] [--out tuned/]
learnc show prog.c                       # entry function and learned variables
learnc fit prog.c --data train.pt [--val test.pt] [--steps N --lr X --batch B]
learnc run prog.c --data test.pt
```

Data is a `.pt` file (`torch.save`) holding a dict with one tensor per parameter of the entry function, examples
along the first dimension, plus `output` with the expected results when there is something to match.

## Limits

- **The native engine suits tens to a few hundred constants**, not neural-network-sized weight sets.
- **Coupled constants can stall.** When two constants compensate for each other, the tuner can settle where
  neither alone improves. More `restarts` help.
- **`#define` constants can't be tuned yet,** because they are not variables. Constants declared as variables
  (`static const int k = ...`) can.
- **In the differentiable runners,** gradients don't pass through float-to-int conversions or branches that never
  rejoin, and loops with data-dependent branches run step by step, which is slower.
- **Discovery needs several starts** and works for small cases.

## FAQ

**Does learnc change my source files?** No. It writes tuned copies to the folder you choose.

**Is there a runtime dependency?** No. The output is plain C with different constants.

**Is this a neural network?** No. Your program keeps its structure; only the numbers you name change. You can
also write small models in C with `learned` variables when you want one.

**How is overfitting handled?** 20% of your examples are held out by default, the best values on them are kept,
and any change that doesn't help on them is reverted.

## License

[MIT](https://github.com/devanmolsharma/learnc/blob/main/LICENSE)
