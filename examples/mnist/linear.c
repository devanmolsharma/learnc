// A linear digit classifier: one score per digit, pick the best.
#include <learnc.h>

learned float w[10][784];
learned float bias[10];

int classify(const float px[784]) {
    int best = 0;
    float best_score = -1e30f;
    for (int d = 0; d < 10; d++) {
        float s = bias[d];
        for (int i = 0; i < 784; i++) s += w[d][i] * px[i];
        if (s > best_score) { best_score = s; best = d; }
    }
    return best;
}
