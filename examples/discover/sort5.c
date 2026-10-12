// A sorting network for 5 numbers with 9 compare-exchange steps. Learned: which two positions each step compares.
#include <learnc.h>

learned_in(0, 4) int first[9];    // step c puts the smaller of v[first[c]], v[second[c]] at first[c]
learned_in(0, 4) int second[9];

int sort5(const int x[5], int which) {
    int v[5];
    for (int k = 0; k < 5; k++) v[k] = x[k];
    for (int c = 0; c < 9; c++) {
        int p = first[c], q = second[c];
        int lo = v[p] < v[q] ? v[p] : v[q];
        int hi = v[p] < v[q] ? v[q] : v[p];
        v[p] = lo;
        v[q] = hi;
    }
    return v[which];
}
