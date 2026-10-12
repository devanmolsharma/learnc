// Find a formula: (x[i] op0 x[j]) op2 (x[k] op1 x[l]). Learned: which inputs, and which operator at each node.
#include <learnc.h>

learned_op(ADD, SUB, MUL, MAX) op[3];
learned_in(0, 4) int leaf[4];

float formula(const float x[5]) {
    float left = learnc_apply(op[0], x[leaf[0]], x[leaf[1]]);
    float right = learnc_apply(op[1], x[leaf[2]], x[leaf[3]]);
    return learnc_apply(op[2], left, right);
}
