// A sorting network for 4 numbers with 5 compare-exchange steps. Learned: which two positions each step compares.
#include <learnc.h>

learned_in(0, 3) int first[5];    // step c puts the smaller of v[first[c]], v[second[c]] at first[c]
learned_in(0, 3) int second[5];

int sort4(const int x[4], int which) {
    int v[4];
    for (int k = 0; k < 4; k++) v[k] = x[k];
    for (int c = 0; c < 5; c++) {
        int p = first[c], q = second[c];
        int lo = v[p] < v[q] ? v[p] : v[q];
        int hi = v[p] < v[q] ? v[q] : v[p];
        v[p] = lo;
        v[q] = hi;
    }
    return v[which];
}
