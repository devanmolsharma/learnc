// Count the elements of a sequence that pass a learned rule. The loop is written by hand; only the rule is learned.
#include <learnc.h>

learned float w[3];
learned float t;

int count(const float x[64][3], int n) {
    int c = 0;
    for (int i = 0; i < n; i++) {
        float s = w[0] * x[i][0] + w[1] * x[i][1] + w[2] * x[i][2];
        if (s > t) c++;
    }
    return c;
}
