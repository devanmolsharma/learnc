# learnc

Write an ordinary C program and mark the values you can't work out by hand as `learned`. learnc fits them to
input/output examples and writes your program back out with the values filled in: plain C that any compiler builds.

It also tunes the constants of C code you already have, without editing it.

## Install

```bash
pip install -e .            # needs clang and gcc on PATH
pip install -e ".[dev]"     # plus pytest
```

Requires Python 3.10+, PyTorch 2.2+ and NumPy.

## Quick start: learn values in a new program

```c
#include <learnc.h>

learned float w1[32][784];   // numbers to learn
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
learnc fit mlp.c --data train.pt --val test.pt     # writes mlp.trained.c and mlp.values.pt
gcc -O2 -c mlp.trained.c                           # learnc is no longer needed
```

On MNIST this program reaches 96.4% test accuracy after 2.4 minutes of training on a CPU. The gcc build of
`mlp.trained.c` gives the same prediction as the trained model on all 10,000 test images.

From Python:

```python
p = learnc.Program("mlp.c")
p.fit(px, target=digits)
p(px)                      # run with the learned values
```

## Tune existing C code

Point `learnc tune` at the constants of code you already have. Your files are never modified: tuned copies are
written to `--out`, identical to your files except for the numbers that changed.

```bash
learnc constants vad_core.c --cflags "-I include"           # list what can be tuned
learnc tune score.c --data dev.pt --tune vad_core.c:kLocalThresholdQ,kSpectrumWeight \
       --source vad_core.c --source vad_gmm.c --cflags "-I include" --out tuned/
```

```python
r = learnc.tune("score.c", [audio, labels], entry="chunk_error", sources=[...],
                constants={"vad_core.c": ["kLocalThresholdQ", "kSpectrumWeight"]})   # or "all"
print(r.report())
r.write("tuned/")
```

- **Entry function.** `score.c` holds a function that takes ONE example and returns a number. That number is either
  a result to match (`output` in the data) or a cost to minimise (an error rate, a size, a time).
- **Any C that compiles works:** pointers, structs, malloc, files, `qsort` with callbacks, `sscanf`, libraries
  without source.
- **Built for real data:**
  - 20% of the examples are held out, and the values that score best on them are kept (early stopping).
  - `keep_close` pulls values toward the originals.
  - A final pass reverts every change that doesn't help on the held-out set, so the diff holds only what matters.
- **Report.** It shows the held-out loss before and after, and every changed constant. It is also saved as
  `TUNING_REPORT.txt`.

### Example: Google's WebRTC voice activity detector

`examples/tune_webrtc_vad/` tunes the unmodified WebRTC VAD on LibriParty dev (2.5 h) and tests it on LibriParty
eval, which it never sees:

| | eval F1 | eval accuracy |
|---|---|---|
| Google's constants | 0.810 | 0.816 |
| 8 named constants tuned (7 min, laptop CPU) | 0.815 | 0.826 |
| every constant of `vad_core.c` tuned (`--all`, 14 min) | **0.859** | **0.858** |

With `--all`, the held-out dev error drops from 22.4% to 12.2%. The constants of the three modes that weren't used
stay unchanged. The output is Google's `vad_core.c` with only the tuned numbers replaced.

**Read the diff before you ship it.** The tuner optimises exactly what you score. Here it set `kMaxSpeechFrames`
to -2, which makes the VAD always use its long hangover. That is valid C and it helped on held-out data, but it is
a change in behaviour.

## What can be learned

| Declaration | Meaning |
|---|---|
| `learned float w[...]` | real numbers: weights, thresholds, coefficients |
| `learned_num int k = 8;` | a whole number tuned like a real one and rounded: sizes, counts, fixed-point constants |
| `learned_in(lo, hi) int k[...]` | a choice among the whole numbers lo..hi: use it in a `switch`, as an index, as a count. This is how you learn **which** operation, input or rule to use |
| `learned_of(1, 2, 4, 8) int k;` | a choice from a list of whole numbers |
| `learned_op(ADD, SUB, MUL) op[3];` | operators from a list (ADD, SUB, MUL, DIV, MAX, MIN, LEFT, RIGHT), applied with `learnc_apply(op[k], a, b)`. A short list means a smaller search |

Everything else is your program, unchanged. Any model is just C over learned variables: rules with learned
thresholds, choices of operations, decision structures, neural networks, and programs whose path depends on the
input. Plain logic runs exactly.

## Examples

Run each script from its own folder.

| Example | What is learned | Result |
|---|---|---|
| `mnist/linear.c` | 7,850 weights | 92.3% on MNIST test |
| `mnist/mlp.c` (784, 32, 10) | 25,450 weights | 96.4% on MNIST test; the gcc build of the trained file matches on 10,000/10,000 |
| `count/count.c`: count the elements that pass a rule, trained on lengths 1 to 10 | rule weights and threshold | 97.1% exact on lengths 40 to 64 (a neural network on the same data: 0%) |
| `rules/approve.c`: 2 of 3 tests must pass | which feature each test reads, its direction, its threshold | recovers the hidden rule (f[2] > 0.606, f[7] > 0.450, f[5] < 0.297; true 0.6, 0.45, 0.3) |
| `formula/formula.c`: `(x[i] op x[j]) op (x[k] op x[l])` | which inputs, and an operator at each node | recovers `(x3 + x0) * (x1 - x2)` exactly from 2,000 examples (11 s) |
| `discover/matmul7.c`: 2x2 matrix product with 7 multiplications | 84 coefficients in -1..1 | finds an exact Strassen-type algorithm (1 in 6 starts, about 30 s each) |
| `discover/sort4.c`, `sort5.c`: sorting networks | which two positions each compare-exchange step uses | finds optimal networks: 5 steps for 4 inputs (13 s), 9 steps for 5 inputs (42 s), proven by the 0-1 principle |
| `tune_webrtc_vad/` | constants of Google's VAD, source unmodified | see above |

`mnist/make_data.py` downloads MNIST (needs torchvision). The WebRTC example needs the C sources from
[py-webrtcvad](https://github.com/wiseman/py-webrtcvad), the LibriParty dataset and `soundfile`.

## How it works

1. **Compile to IR.** clang compiles the program, plus the sources of any libraries it uses (`--source lib.c`), to
   LLVM IR.
2. **Run.** `Program` picks one of three ways to run it, and `p.native_reason` says why:
   - **Fast runner**, for arrays, loops and math. The IR runs in PyTorch with one example per lane. Each lane
     follows its own path through branches, loops and recursion. Loop nests with fixed trip counts run all their
     iterations at once. Gradients are exact.
   - **Flat memory runner**, for pointers stored in data, linked structures, malloc/free, structs, function
     pointers, memcpy and strlen. Memory is byte-addressed per example; gradients are still exact.
   - **Native engine**, for anything else that compiles (stdio, files, sscanf, qsort, libraries without source).
     The program is built with the compiler and run as is. Gradients are estimated from native runs:
     - central differences, for a few constants with smooth outputs;
     - antithetic random perturbations, for outputs that jump;
     - sampling, for whole-number choices.

     Runs go to forked worker processes, so a crash in the C code becomes an error message.
3. **Fit.**
   - **Real-valued learned variables** are fitted by gradient descent (Adam by default, `optimizer="lbfgs"` for
     smooth fits that must become exact). A comparison that depends on them becomes soft while training
     (`sigmoid(difference / temperature)`): a value it chooses becomes a blend, and an `if` that depends on them runs
     both sides and blends them where the paths meet again.
   - **Whole-number choices** are fitted in one of these ways:
     - `choices="auto"` (default): restarts alternate between "relax" and "sample", and the best result is kept.
     - `choices="sample"`: several complete settings are tried per step, and the better ones become more likely.
       This always works, because it only needs exact runs.
     - `choices="relax"`: each choice is a weighted mix of its values, so a learned index blends what it may pick
       and a `switch` runs every case and blends them.
     - `choices="continuous"`: each choice is a real number that is pulled to whole numbers late in training.
       Suits choices used as numbers, such as algorithm coefficients.
     - `sparsity=`: a cost per non-zero choice.

     An exact polish follows: each choice tries all its values in turn, with the others fixed.
4. **Bake.** The learned values are written into a copy of your file as initial values.

The runners are checked against the natively compiled program example by example (`tests/test_exact.py`). The
results are identical, including floating-point sums.

## Command line

Data is a `.pt` file (`torch.save`) holding a dict with one tensor per parameter of the entry function (examples
along the first dimension) and `output`, the expected results. The entry function is the last function in the
file, unless `--entry` names another.

```bash
learnc show prog.c                       # entry function and learned variables
learnc fit prog.c --data train.pt [--val test.pt] [--steps N --lr X --batch B --temperature T]
learnc run prog.c --data test.pt         # with prog.values.pt
learnc constants lib.c                   # constants of existing code that can be tuned
learnc tune entry.c --data d.pt --tune lib.c:NAME,NAME --source lib.c [--minimize] [--out tuned/]
```

## Limits

- **The native engine is for tens to a few hundred constants**, not neural-network-sized weight sets. Programs
  that call library functions without source always use it.
- **Coupled constants can stall.** When two constants compensate for each other (a gain and a window length), the
  tuner can settle where neither alone can improve. More `restarts` help.
- **Reading memory as a different type** (a float's bytes as an int) is not supported by the runners; such
  programs use the native engine.
- **Gradients don't pass through some steps** in the runners:
  - converting a learned float to an int;
  - integers that are stored to memory and loaded back while relaxed;
  - branches that never rejoin (one side returns early) are taken the hard way.
- **Globals are reset for each example in training.** A natively built program keeps any changes to globals
  between calls.
- **Some loops run step by step,** which is slow to train: loops with data-dependent branches, or loops that carry
  a value other than a running total (such as the running maximum in an argmax loop).
- **Discovery needs several starts** and works for small cases (2x2 matrix products, sorting networks of 4 and 5
  inputs).

## Tests

```bash
pytest
```
