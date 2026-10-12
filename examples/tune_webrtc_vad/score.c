/* Entry points for tuning WebRTC VAD with learnc.tune(): the VAD itself is Google's unmodified source. */
#include "webrtc/common_audio/vad/vad_core.h"

#define FRAME 480                                   /* 30 ms at 16 kHz */
#define FRAMES 1000                                 /* one 30 s chunk */

/* VAD decisions (mode 0) for every 30 ms frame of a recording */
void decide_stream(const short *audio, int frames, signed char *out) {
    VadInstT vad;
    WebRtcVad_InitCore(&vad); WebRtcVad_set_mode_core(&vad, 0);
    for (int f = 0; f < frames; f++) out[f] = WebRtcVad_CalcVad16khz(&vad, audio + f * FRAME, FRAME) > 0;
}

/* tuning entry: share of the frames of one 30 s chunk that the VAD gets wrong */
double chunk_error(const short audio[FRAMES * FRAME], const signed char labels[FRAMES]) {
    signed char d[FRAMES];
    decide_stream(audio, FRAMES, d);
    int wrong = 0;
    for (int f = 0; f < FRAMES; f++) wrong += d[f] != labels[f];
    return (double)wrong / FRAMES;
}
