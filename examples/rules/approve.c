// Approve when at least 2 of 3 tests pass. Learned: which feature each test reads, its direction and its threshold.
#include <learnc.h>

learned_in(0, 7) int feature[3];
learned_in(0, 1) int above[3];      // 1: pass when the value is above the threshold, 0: below
learned float threshold[3];

int approve(const float f[8]) {
    int passed = 0;
    for (int k = 0; k < 3; k++) {
        float v = f[feature[k]];
        int ok = above[k] ? v > threshold[k] : v < threshold[k];
        passed += ok;
    }
    return passed >= 2;
}
