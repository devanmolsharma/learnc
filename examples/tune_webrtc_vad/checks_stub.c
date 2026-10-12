/* WebRTC's error reporter (C++ in the original tree). The VAD only calls it on programming errors. */
#include <stdio.h>
#include <stdlib.h>
void rtc_FatalMessage(const char *file, int line, const char *msg) { fprintf(stderr, "%s:%d: %s\n", file, line, msg); abort(); }
