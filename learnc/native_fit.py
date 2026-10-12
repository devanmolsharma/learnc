"""Fitting learned variables by running the program natively: works for any C program that compiles (pointers,
malloc, files, library calls, function pointers...), because nothing is emulated.

Real-valued learned variables (and `learned_num` whole numbers, tuned smoothly and rounded): Adam on a gradient
ESTIMATED from native runs:
  - gradient "fd": central differences, one nudge per value (smooth outputs, few values);
  - gradient "es": antithetic random perturbations (outputs that jump: decisions, error counts, sizes).
Whole-number choices (learned_in / learned_of / learned_op): sampled settings, better ones made more likely, then an
exact search over every value of each element.
Guarding against overfitting: when validation data is given, it is scored regularly and the values that did best on
it are returned (early stopping); `keep_close` pulls values toward where they started (relative to their size);
finally, changes that do not help on validation are undone (`minimal`), so the diff holds only what matters.
Speed and safety: with workers > 1, evaluations run in forked worker processes, so a crash in the program becomes
an error message instead of ending the Python process.
"""
import multiprocessing as mp
import os
import numpy as np
import torch

_CTX = {}                                     # shared with forked workers: program, data, value layout


def _loss(out, y):
    """Per example: squared error for real targets, 0/1 mismatch for whole-number targets."""
    out = out.double().numpy(); y = y.numpy()
    if np.issubdtype(y.dtype, np.floating): return (out - y) ** 2
    return (np.round(out).astype(np.int64) != y.astype(np.int64)).astype(np.float64)


def _pick(inputs, idx):
    return [[x[i] for i in idx] if isinstance(x, (list, tuple)) else torch.as_tensor(x)[idx] for x in inputs]


def _values_of(theta, choice):
    floats, sizes, ints = _CTX["floats"], _CTX["sizes"], _CTX["ints"]
    v, o = {}, 0
    for k, s in zip(floats, sizes):
        x = theta[o:o + s]
        if getattr(floats[k], "numeric", False): x = np.round(x)
        v[k] = x.reshape(floats[k].shape) if floats[k].shape else x[0]; o += s
    for k in ints:
        c = np.array(ints[k].choices)[choice[k]]
        v[k] = c.reshape(ints[k].shape) if ints[k].shape else c[0]
    return v


def _eval(job):
    """One loss: (theta, choice, which data set, example indices) -> mean loss."""
    theta, choice, split, idx = job
    inputs, target = _CTX[split]
    out = _CTX["native"].run(*_pick(inputs, idx), values=_values_of(theta, choice))
    l = out.double().numpy() if _CTX["objective"] == "minimize" else _loss(out, torch.as_tensor(target)[idx])
    return float(np.mean(l))


def _init_worker():
    torch.set_num_threads(1)                  # forked after torch started its thread pool: stay single-threaded


class _Evaluator:
    def __init__(self, workers):
        self.pool = mp.get_context("fork").Pool(workers, initializer=_init_worker) if workers > 1 else None

    def __call__(self, jobs):
        if self.pool is None: return [_eval(j) for j in jobs]
        try: return self.pool.map(_eval, jobs)
        except Exception as e:
            raise RuntimeError(f"the program failed while being evaluated (crash or error in the C code): {e}") from None

    def close(self):
        if self.pool is not None: self.pool.terminate()


class _Adam:
    """Adam that returns a direction; the caller scales the step to each value's own size."""
    def __init__(self, n): self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0
    def direction(self, g):
        self.t += 1; self.m = 0.9 * self.m + 0.1 * g; self.v = 0.999 * self.v + 0.001 * g * g
        return (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-12)


def fit_native(prog, inputs, target, steps=300, batch=256, lr=0.02, sigma=0.05, pairs=16, fd_limit=32, samples=16,
               polish=2, restarts=1, seed=0, gradient="auto", objective="match", validation=None, keep_close=0.0,
               eval_every=None, workers=None, minimal=True, verbose=False):
    """validation: (inputs, target) to select the final values on (None: the training examples).
    minimal: at the end, put back every original value whose change does not lower the validation loss.
    Returns prog with prog.values set; prog.fit_report holds start / final validation losses."""
    if target is None: target = torch.zeros(len(inputs[0]))
    y = torch.as_tensor(target); n = len(y)
    if validation is None: vin, vy = inputs, y
    else:
        vin = validation[0]
        vy = torch.as_tensor(validation[1]) if validation[1] is not None else torch.zeros(len(vin[0]))
    floats = {k: L for k, L in prog.learned.items() if L.etype[0] == "fp" or getattr(L, "numeric", False)}
    ints = {k: L for k, L in prog.learned.items() if L.etype[0] == "int" and not getattr(L, "numeric", False)}
    sizes = [floats[k].count for k in floats]; P = sum(sizes)
    _CTX.update(native=prog.native(), floats=floats, sizes=sizes, ints=ints, objective=objective,
                train=(inputs, y), val=(vin, vy))
    real_target = y.is_floating_point()
    use_fd = gradient == "fd" or (gradient == "auto" and objective == "match" and real_target and 0 < P <= fd_limit)
    if workers is None: workers = 1 if use_fd else min(os.cpu_count() or 1, 8)
    vidx = np.arange(min(len(vy), 4096))                               # validation examples used for selection
    eval_every = eval_every or max(1, steps // 20)

    start = prog.values or {}
    theta0 = []
    for k in floats:
        L = floats[k]
        theta0.append(np.asarray(torch.as_tensor(start[k]).reshape(-1), np.float64) if k in start
                      else np.asarray(L.init, np.float64).reshape(-1))
    theta0 = np.concatenate(theta0) if theta0 else np.zeros(0)
    random_start = bool(P) and not theta0.any()
    if random_start:                                                   # all-zero start (e.g. weights): break symmetry
        theta0 = np.random.default_rng(seed).standard_normal(P) * 0.1
    whole = np.array([getattr(floats[k], "numeric", False) for k, s in zip(floats, sizes) for _ in range(s)], bool)
    floor = np.where(whole, 1.0, 0.1)
    scale0 = np.maximum(np.abs(theta0), floor) if P else np.zeros(0)

    def penalty(theta): return keep_close * float(np.mean(((theta - theta0) / scale0) ** 2)) if P else 0.0
    def penalty_grad(theta): return keep_close * 2 * (theta - theta0) / scale0 ** 2 / max(P, 1) if P else 0.0

    start_choice = {}
    for k in ints:                                                     # choices start at the program's own values
        ch = list(ints[k].choices)
        start_choice[k] = np.array([ch.index(int(v)) if int(v) in ch else 0 for v in np.asarray(ints[k].init).reshape(-1)])
    ev = _Evaluator(workers)
    try:
        start_val = ev([(theta0, start_choice, "val", vidx)])[0]
        best = (start_val, theta0.copy(), {k: v.copy() for k, v in start_choice.items()})
        history = []
        for r in range(restarts):
            rng = np.random.default_rng(seed + r)
            theta = theta0.copy() if r == 0 else theta0 + rng.standard_normal(P) * 0.2 * scale0
            logits = {k: np.eye(len(ints[k].choices))[start_choice[k]] * 2.0 for k in ints}
            choice = {k: v.copy() for k, v in start_choice.items()}
            opt = _Adam(P)
            for step in range(steps):
                idx = rng.choice(n, min(batch, n), replace=False)
                if P:                                                   # ---- real values (and learned_num)
                    scale = np.maximum(np.abs(theta), scale0)
                    if use_fd:
                        h = np.where(whole, 1.0, 1e-3 * scale); jobs = []          # whole numbers: step to the next one
                        for i in range(P):
                            e = np.zeros(P); e[i] = h[i]
                            jobs += [(theta + e, choice, "train", idx), (theta - e, choice, "train", idx)]
                        L = ev(jobs); g = np.array([(L[2 * i] - L[2 * i + 1]) / (2 * h[i]) for i in range(P)])
                    else:
                        s = np.where(whole, np.maximum(sigma * scale, 1.0), sigma * scale)    # whole numbers: cross to others
                        eps = rng.standard_normal((pairs, P)); jobs = []
                        for e in eps: jobs += [(theta + s * e, choice, "train", idx), (theta - s * e, choice, "train", idx)]
                        L = np.array(ev(jobs)).reshape(pairs, 2)
                        g = ((L[:, 0] - L[:, 1])[:, None] * eps).mean(0) / (2 * s)
                    g = g + penalty_grad(theta)
                    decay = 0.5 * (1 + np.cos(np.pi * step / steps))
                    theta = theta - lr * decay * scale * opt.direction(g)
                if ints:                                                # ---- whole-number choices
                    drawn = []
                    for _ in range(samples):
                        c = {}
                        for k in ints:
                            p = np.exp(logits[k] - logits[k].max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                            c[k] = np.array([rng.choice(len(row), p=row) for row in p])
                        drawn.append(c)
                    rewards = -np.array(ev([(theta, c, "train", idx) for c in drawn]))
                    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
                    for k in ints:
                        p = np.exp(logits[k] - logits[k].max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                        grad = np.zeros_like(logits[k])
                        for c, a in zip(drawn, adv):
                            oh = np.zeros_like(p); oh[np.arange(len(c[k])), c[k]] = 1; grad += a * (oh - p)
                        logits[k] += 0.5 * grad / samples; choice[k] = logits[k].argmax(1)
                if (step + 1) % eval_every == 0 or step == steps - 1:   # ---- early stopping on validation
                    v = ev([(theta, choice, "val", vidx)])[0] + penalty(theta)
                    history.append(v)
                    if v < best[0]: best = (v, theta.copy(), {k: c.copy() for k, c in choice.items()})
                    if verbose: print(f"  step {step + 1}/{steps}  validation loss {v:.5f}  (best {best[0]:.5f}, start {start_val:.5f})", flush=True)
        _, theta, choice = best
        for _ in range(polish if ints else 0):                          # exact search over each choice element
            changed = False; cur = ev([(theta, choice, "val", vidx)])[0]
            for k in ints:
                for e in range(ints[k].count):
                    trials = []
                    for v in range(len(ints[k].choices)):
                        if v == choice[k][e]: continue
                        t = {kk: vv.copy() for kk, vv in choice.items()}; t[k][e] = v; trials.append(t)
                    losses = ev([(theta, t, "val", vidx) for t in trials])
                    if losses:
                        i = int(np.argmin(losses))
                        if losses[i] < cur - 1e-12: choice, cur, changed = trials[i], losses[i], True
            if not changed: break
        if minimal and P and not random_start:                                             # minimal diff: undo changes that do not help
            cur = ev([(theta, choice, "val", vidx)])[0]
            for _ in range(2):
                changed = [i for i in range(P) if not np.isclose(theta[i], theta0[i])]
                trials = []
                for i in changed: t = theta.copy(); t[i] = theta0[i]; trials.append(t)
                losses = ev([(t, choice, "val", vidx) for t in trials])
                undone, base = False, cur
                for i in np.argsort(losses):                            # most harmless first, re-checked together
                    if losses[i] > base + 1e-12: break                  # reverting it alone already hurt
                    t = theta.copy(); t[changed[i]] = theta0[changed[i]]
                    l = ev([(t, choice, "val", vidx)])[0]
                    if l <= cur + 1e-12: theta, cur, undone = t, l, True
                if not undone: break
        final_val = ev([(theta, choice, "val", vidx)])[0]
    finally:
        ev.close()
    prog.values = {k: torch.as_tensor(np.asarray(v)) for k, v in _values_of(theta, choice).items()}
    prog.fit_report = {"validation_loss_start": start_val, "validation_loss_final": final_val, "history": history}
    return prog
