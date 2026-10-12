"""Native build of a program: your C file compiled as is, plus a generated driver that #includes it (so even static
functions and variables are reachable), sets the learned variables and calls the entry function once per example."""
import ctypes, hashlib, os
import numpy as np
import torch
from .build import cache_dir, compile_shared, learned_declarations

C_NAME = {("fp", 32): "float", ("fp", 64): "double", ("int", 8, True): "signed char", ("int", 8, False): "unsigned char",
          ("int", 16, True): "short", ("int", 16, False): "unsigned short", ("int", 32, True): "int",
          ("int", 32, False): "unsigned int", ("int", 64, True): "long long", ("int", 64, False): "unsigned long long",
          ("int", 1, False): "_Bool"}
NP = {"float": np.float32, "double": np.float64, "signed char": np.int8, "unsigned char": np.uint8, "short": np.int16,
      "unsigned short": np.uint16, "int": np.int32, "unsigned int": np.uint32, "long long": np.int64,
      "unsigned long long": np.uint64, "_Bool": np.bool_}


def _cname(t): return C_NAME[t[:2]] if t[0] == "fp" else C_NAME[t]


class Native:
    def __init__(self, prog):
        self.prog = prog
        d = cache_dir(prog.paths)
        lines = [f'#include "{prog.path}"', "#include <string.h>"]
        local = set(learned_declarations(prog.path, getattr(prog, "flags", ())))    # learned variables of the entry file
        for n, L in prog.learned.items():
            if n in local: lines.append(f"void learnc_set_{n}(const void *v) {{ memcpy((void *)&{n}, v, sizeof({n})); }}")
            else: lines.append(f"void learnc_set_{n}(const void *v);   /* defined in a lifted source file */")
        params = []
        call = []
        for k, (name, ct) in enumerate(prog.params):
            if ct[0] == "ptr":
                params.append(f"const {_cname(ct[1])} *p{k}"); call.append(f"({_cname(ct[1])} *)(p{k} + e * strides[{k}])")
            else:
                params.append(f"const {_cname(ct)} *p{k}"); call.append(f"p{k}[e]")
        rt = prog.ret
        self.out_type = "double" if rt[0] == "fp" else "long long"
        lines.append(f"void learnc_run(long long n, const long long *strides{''.join(', ' + p for p in params)}, {self.out_type} *out) {{")
        lines.append(f"  for (long long e = 0; e < n; e++) out[e] = ({self.out_type}) {prog.entry}({', '.join(call)});")
        lines.append("}")
        text = "\n".join(lines) + "\n"
        tag = f"{prog.entry}_{hashlib.sha1(text.encode()).hexdigest()[:10]}"      # the OS reuses a loaded library per path
        drv = os.path.join(d, f"driver_{tag}.c"); open(drv, "w").write(text)
        lib = os.path.join(d, f"native_{tag}.so")
        if not os.path.exists(lib): compile_shared([drv] + prog.paths[1:], lib, getattr(prog, "flags", ()))
        self.lib = ctypes.CDLL(lib)

    def set_values(self, values):
        for n, L in self.prog.learned.items():
            if n not in values: continue
            t = NP[_cname(L.etype if L.etype[0] == "fp" else (L.etype[0], L.etype[1], True))]
            raw = torch.as_tensor(values[n]).detach().cpu().numpy().reshape(-1)
            if np.issubdtype(t, np.integer):                              # whole numbers: round and keep in range
                info = np.iinfo(t); raw = np.clip(np.round(raw), info.min, info.max)
            arr = np.ascontiguousarray(raw.astype(t))
            assert arr.size == L.count, f"{n}: expected {L.count} values, got {arr.size}"
            getattr(self.lib, f"learnc_set_{n}")(arr.ctypes.data_as(ctypes.c_void_p))

    def run(self, *inputs, values=None):
        if values: self.set_values(values)
        n = len(inputs[0]); keep = []; ptrs = []; strides = []
        for (name, ct), x in zip(self.prog.params, inputs):
            t = NP[_cname(ct[1] if ct[0] == "ptr" else ct)]
            if isinstance(x, (list, tuple)) and x and isinstance(x[0], (str, bytes)):     # strings: zero-terminated rows
                raw = [s.encode() if isinstance(s, str) else s for s in x]
                a = np.zeros((n, max(len(r) for r in raw) + 1), np.uint8)
                for i, r in enumerate(raw): a[i, :len(r)] = np.frombuffer(r, np.uint8)
            else:
                a = np.ascontiguousarray(torch.as_tensor(x).detach().cpu().numpy().reshape(n, -1).astype(t))
            keep.append(a); ptrs.append(a.ctypes.data_as(ctypes.c_void_p)); strides.append(a.shape[1])
        st = np.array(strides, dtype=np.int64)
        out = np.zeros(n, dtype=np.float64 if self.out_type == "double" else np.int64)
        self.lib.learnc_run(ctypes.c_longlong(n), st.ctypes.data_as(ctypes.c_void_p), *ptrs, out.ctypes.data_as(ctypes.c_void_p))
        if self.prog.ret == ("fp", 32): out = out.astype(np.float32)
        return torch.from_numpy(out)
