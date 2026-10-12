// 2x2 matrix product with only 7 multiplications. Learned: the -1/0/+1 coefficients (Strassen-style).
#include <learnc.h>

learned_in(-1, 1) int U[7][4];   // product k multiplies (sum of U[k][i] * a[i]) ...
learned_in(-1, 1) int V[7][4];   // ... by (sum of V[k][i] * b[i])
learned_in(-1, 1) int W[4][7];   // c[j] = sum of W[j][k] * product k

float entry(const float a[4], const float b[4], int j) {
    float m[7];
    for (int k = 0; k < 7; k++) {
        float x = 0, y = 0;
        for (int i = 0; i < 4; i++) { x += U[k][i] * a[i]; y += V[k][i] * b[i]; }
        m[k] = x * y;
    }
    float c = 0;
    for (int k = 0; k < 7; k++) c += W[j][k] * m[k];
    return c;
}
