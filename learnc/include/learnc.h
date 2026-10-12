/* learnc: mark the values a C program should learn from examples. Everything else is ordinary C.

   learned float w[10][784];            numbers to learn (weights, thresholds, ...)
   learned_in(0, 3) int k[16];           whole numbers to learn, each one of 0..3 (in a switch, as an index, ...)
   learned_of(1, 2, 4, 8) int s;         whole numbers to learn, each one of the listed values
   learned_op(ADD, SUB, MUL) op[3];      operators to learn, each one of the listed ones; use learnc_apply(op, a, b)
   learned_num short gain[6];            whole numbers tuned smoothly and rounded (any integer constant: thresholds,
                                         fixed-point coefficients, table entries); native engine

   Without learnc (e.g. any normal compiler) the program still builds: learned variables are plain globals. */
#ifndef LEARNC_H
#define LEARNC_H
#define learned __attribute__((used, annotate("learnc")))
#define learned_in(lo, hi) __attribute__((used, annotate("learnc:" #lo ":" #hi)))
#define learned_of(...) __attribute__((used, annotate("learnc:of:" #__VA_ARGS__)))
#define learned_op(...) __attribute__((used, annotate("learnc:op:" #__VA_ARGS__))) int
#define learned_num __attribute__((used, annotate("learnc:num")))

/* operators for learned_op: a OP b */
enum { ADD, SUB, MUL, DIV, MAX, MIN, LEFT, RIGHT };

static inline float learnc_apply(int op, float a, float b) {
    switch (op) {
        case ADD: return a + b;
        case SUB: return a - b;
        case MUL: return a * b;
        case DIV: return b != 0 ? a / b : 0;
        case MAX: return a > b ? a : b;
        case MIN: return a < b ? a : b;
        case LEFT: return a;
        default: return b;          /* RIGHT */
    }
}
#endif
