/*
 * cycle-timer.c
 *
 * Implementation for cycle-timer.h
 */

#include "cycle-timer.h"
#include "hedley.h"

#include <stdlib.h>
#include <stdio.h>

#include <time.h>


const size_t ITERS = 10000;
const size_t TRIES = 11;
const size_t WARMUP = 1000;

volatile size_t sink;

/**
 * Calibration loop that relies on store throughput being exactly 1 per cycle
 * on all modern x86 chips, and the loop overhead running totally in parallel.
 */
HEDLEY_NEVER_INLINE
__attribute__((aligned(32)))
void store_calibration(size_t iters) {
    do {
        sink = iters;
    } while (--iters > 0);
}

int intcompare(const void *l_, const void *r_) {
    int64_t l = *(const uint64_t *)l_;
    int64_t r = *(const uint64_t *)r_;
    return (l > r) - (l < r);
}

/*
 * Calculate the frequency of the CPU based on timing a tight loop that we expect to
 * take one iteration per cycle.
 *
 * ITERS is the base number of iterations to use: the calibration routine is actually
 * run twice, once with ITERS iterations and once with 2*ITERS, and a delta is used to
 * remove measurement overhead.
 */
HEDLEY_NEVER_INLINE
static double get_ghz(bool print) {

    const char *force = getenv("CYCLE_TIMER_FORCE_MHZ");
    if (force) {
        int mhz = atoi(force);
        if (mhz) {
            double ghz = mhz / 1000.;
            if (print) fprintf(stderr, "Forced CPU speed (CYCLE_TIMER_FORCE_MHZ): %5.2f GHz\n", ghz);
            return ghz;
        } else {
            if (print) fprintf(stderr, "Bad value for CYCLE_TIMER_FORCE_MHZ: '%s' (falling back to cal loop)\n", force);
        }
    }

    int64_t results[TRIES];

    for (size_t w = 0; w < WARMUP + 1; w++) {
        for (size_t r = 0; r < TRIES; r++) {
            cl_timepoint t0 = cl_now();
            store_calibration(ITERS);
            cl_timepoint t1 = cl_now();
            store_calibration(ITERS * 2);
            cl_timepoint t2 = cl_now();
            results[r] = cl_delta(t1, t2).nanos - cl_delta(t0, t1).nanos;
        }
    }

    // return the median value
    qsort(results, TRIES, sizeof(results[0]), intcompare);
    double ghz = ((double)ITERS / results[TRIES/2]);
    if (print) fprintf(stderr, "Estimated CPU speed: %5.2f GHz\n", ghz);
    return ghz;
}

static bool is_init = false;
double ghz;

void cl_init(bool print) {
    if (HEDLEY_UNLIKELY(!is_init)) {
        ghz = get_ghz(print);
        is_init = true;
    }
};

cl_timepoint cl_now() {
    struct timespec spec;
    if (clock_gettime(CLOCK_MONOTONIC, &spec)) {
        return (cl_timepoint){0};
    } else {
        return (cl_timepoint){spec.tv_sec * 1000000000ll + spec.tv_nsec};
    }
}

/*
 * Take an interval value and convert it to cycles based on the
 * detected frequency of this host.
 */
double cl_to_cycles(cl_interval interval) {
    cl_init(false);
    return interval.nanos * ghz;
}

/*
 * Take an interval value and "convert" it to nanos.
 */
double cl_to_nanos(cl_interval interval) {
    return interval.nanos;
}
