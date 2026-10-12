// A small neural network, written as ordinary C: 784 pixels -> 32 hidden units (ReLU) -> 10 digit scores.
#include <learnc.h>

learned float w1[32][784];
learned float b1[32];
learned float w2[10][32];
learned float b2[10];

int classify(const float px[784]) {
    float h[32];
    for (int j = 0; j < 32; j++) {
        float s = b1[j];
        for (int i = 0; i < 784; i++) s += w1[j][i] * px[i];
        h[j] = s > 0 ? s : 0;
    }
    int best = 0;
    float best_score = -1e30f;
    for (int d = 0; d < 10; d++) {
        float s = b2[d];
        for (int j = 0; j < 32; j++) s += w2[d][j] * h[j];
        if (s > best_score) { best_score = s; best = d; }
    }
    return best;
}
