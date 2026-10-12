"""Full C programs the differentiable runner cannot run (malloc, linked lists, text parsing, qsort with a function
pointer, files): the native engine fits their learned constants anyway."""
import textwrap
import numpy as np
import pytest
import torch
from learnc import Program


def write(tmp_path, name, src):
    p = tmp_path / name; p.write_text(textwrap.dedent(src)); return str(p)


def test_linked_list_with_malloc(tmp_path):
    p = Program(write(tmp_path, "list.c", """
        #include <stdlib.h>
        #include <learnc.h>
        struct node { float v; struct node *next; };
        learned float decay = 0.5f;
        learned float gain = 1.0f;
        float smooth(const float x[16]) {
            struct node *head = NULL;                        /* build a linked list, newest first */
            for (int i = 0; i < 16; i++) { struct node *n = malloc(sizeof *n); n->v = x[i]; n->next = head; head = n; }
            float acc = 0, w = 1;
            for (struct node *n = head; n; n = n->next) { acc += w * n->v; w *= decay; }
            while (head) { struct node *n = head->next; free(head); head = n; }
            return gain * acc;
        }
    """), device="cpu")
    assert p.flat and not p.native_only                           # runs in the differentiable flat memory
    g = torch.Generator().manual_seed(0); x = torch.rand(2000, 16, generator=g)
    vals = {"decay": torch.tensor(0.7), "gain": torch.tensor(1.3)}
    assert torch.allclose(p(x[:50], values=vals).double(), p.native().run(x[:50], values=vals).double(), atol=1e-5)
    w = 0.8 ** torch.arange(16).flip(0).float()                   # true decay 0.8, gain 0.3
    y = 0.3 * (x * w).sum(1)
    for engine in ("runner", "native"):                           # exact gradients through the linked list, and estimated
        p.values = None
        p.fit(x, target=y, steps=300, batch=128, lr=0.05, engine=engine)
        assert abs(p.values["decay"].item() - 0.8) < 0.01 and abs(p.values["gain"].item() - 0.3) < 0.01, (engine, p.values)


def test_text_parsing_rule(tmp_path):
    p = Program(write(tmp_path, "parse.c", """
        #include <stdio.h>
        #include <learnc.h>
        learned float max_temp = 20.0f;
        learned float max_hum = 50.0f;
        int alarm(const char *line) {                         /* "temp=31.5 hum=62" */
            float t = 0, h = 0;
            if (sscanf(line, "temp=%f hum=%f", &t, &h) != 2) return 0;
            return t > max_temp && h > max_hum;
        }
    """), device="cpu")
    assert p.native_only and "sscanf" in p.native_reason
    rng = np.random.default_rng(1); t, h = rng.uniform(10, 45, 3000), rng.uniform(20, 95, 3000)
    lines = [f"temp={a:.2f} hum={b:.1f}" for a, b in zip(t, h)]
    y = torch.tensor(((t > 33.0) & (h > 70.0)).astype(np.int64))
    p.fit(lines, target=y, steps=400, batch=512, lr=0.03, restarts=3)
    assert p.accuracy(lines, target=y) > 0.98, p.values


def test_qsort_with_function_pointer_and_choice(tmp_path):
    p = Program(write(tmp_path, "rank.c", """
        #include <stdlib.h>
        #include <learnc.h>
        learned float w[3];
        learned_in(0, 1) int descending;
        static float key(const float *item) { return w[0] * item[0] + w[1] * item[1] + w[2] * item[2]; }
        static int cmp(const void *a, const void *b) {
            float ka = key(a), kb = key(b);
            int r = (ka > kb) - (ka < kb);
            return descending ? -r : r;
        }
        int top_item(const float items[8][3]) {               /* which item comes first after sorting */
            float copy[8][4];
            for (int i = 0; i < 8; i++) { for (int j = 0; j < 3; j++) copy[i][j] = items[i][j]; copy[i][3] = i; }
            qsort(copy, 8, sizeof copy[0], cmp);
            return (int)copy[0][3];
        }
    """), device="cpu")
    assert p.native_only
    g = torch.Generator().manual_seed(2); items = torch.rand(3000, 8, 3, generator=g)
    y = (items @ torch.tensor([2.0, -1.0, 0.5])).argmax(1)           # best item by a hidden scoring rule
    p.fit(items, target=y, steps=300, batch=256, lr=0.05, restarts=3)
    assert p.accuracy(items, target=y) > 0.95, p.values


def test_reads_a_file(tmp_path):
    table = tmp_path / "calibration.txt"
    table.write_text("\n".join(f"{v:.4f}" for v in np.linspace(0, 1, 11) ** 2))       # a lookup table on disk
    p = Program(write(tmp_path, "calib.c", f"""
        #include <stdio.h>
        #include <learnc.h>
        learned float offset = 0.0f;
        learned float scale = 1.0f;
        float corrected(float raw) {{
            static float table[11]; static int loaded = 0;
            if (!loaded) {{ FILE *f = fopen("{table}", "r"); for (int i = 0; i < 11; i++) fscanf(f, "%f", &table[i]); fclose(f); loaded = 1; }}
            float x = raw * 10; int i = (int)x; if (i < 0) i = 0; if (i > 9) i = 9;
            float t = x - i, v = table[i] * (1 - t) + table[i + 1] * t;    /* interpolate the file's table */
            return scale * v + offset;
        }}
    """), device="cpu")
    assert p.native_only
    raw = torch.rand(2000, generator=torch.Generator().manual_seed(3))
    y = p(raw, values={"offset": 0.25, "scale": 2.0})                # targets from the program with the true constants
    p.fit(raw, target=y, steps=300, batch=256, lr=0.05)
    assert abs(p.values["offset"].item() - 0.25) < 0.01 and abs(p.values["scale"].item() - 2.0) < 0.02, p.values
