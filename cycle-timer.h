/*
 * cycle-timer.h
 *
 * A timer that returns results in CPU cycles in addition to nanoseconds.
 * It measures cycles indirectly by measuring the wall-time, and then converting
 * that to a cycle count based on a calibration loop performed once at startup.
 */

#ifndef CYCLE_TIMER_H_
#define CYCLE_TIMER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <inttypes.h>
#include <stdbool.h>

/**
 * A point in time, or an interval when subtracted. You should probably
 * treat this as an opaque struct, in case I change the implementation
 * someday.
 */
struct cl_timepoint_ {
    int64_t nanos;
};
typedef struct cl_timepoint_ cl_timepoint;

/**
 * An interval created by subtracting two points in time, measured
 * in nanoseconds.
 */
struct cl_interval_ {
    int64_t nanos;
};
typedef struct cl_interval_ cl_interval;

/* return the current moment in time as a cycletimer_result */
cl_timepoint cl_now();

/*
 * Return the interval between timepoints first and second.
 * This value is positive iff second occus after first.
 */
static inline cl_interval cl_delta(cl_timepoint first, cl_timepoint second) {
    return (cl_interval){second.nanos - first.nanos};
}

/*
 * Take an interval value and convert it to cycles based on the
 * detected frequency of this host.
 */
double cl_to_cycles(cl_interval interval);

double cl_to_nanos(cl_interval interval);

/*
 * Initialize the cycletimer infrastructure. Mostly this just means calculating
 * the cycle to nanoseconds value (i.e., the CPU frequency). You never *need* to
 * use this function, if you haven't call it, it will happens automatically when
 * init is necessary (usually lazily - when accessing the cl_to_cycles),
 * but may be lengthy, so this method is offfered so that the user can trigger
 * it at a time of their choosing (and allowing the user to elect whether to
 * print out diagnostic information about the calibration).
 *
 * If you pass true for print, dignostic information like the detected CPU
 * frequency is printed to stderr.
 */
void cl_init(bool print);

#ifdef __cplusplus
}
#endif

#endif /* CYCLE_TIMER_HPP_ */
