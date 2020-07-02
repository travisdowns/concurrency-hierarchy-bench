#ifndef CYCLIC_BARRIER_H_
#define CYCLIC_BARRIER_H_

#include <atomic>
#include <cinttypes>
#include <functional>

/**
 * Auto-reset spin-based barrier.
 *
 * Waits for N threads to arrive, then releases those N threads and resets itself, again
 * waiting for N threads to arrive.
 */
template <typename T>
struct cyclic_barrier_template {
    using int32_t = std::int32_t;
    int32_t break_count;
    std::atomic<int32_t> current;

    std::function<T()> breaker_function;
    T return_value;
    std::mutex lock;

    template <typename F>
    cyclic_barrier_template(uint32_t count, F f) : break_count(count), current{0}, breaker_function{f} {}

    /* increment and hot spin on the waiter count until it hits the break point, returns the spin count in case you care */
    T wait() {
        for (size_t count = 0; ; count++) {

            auto waiters = current.load();

            if (waiters < 0) {
                // while waiters < 0, there are draining earlier waiters, so we wait
                // for them to leave
                continue;
            }
            assert(waiters < break_count);

            // two remaining cases: we are not the breaking waiter, in which case we increment and wait ...
            if (waiters < break_count - 1) {
                if (current.compare_exchange_strong(waiters, waiters + 1)) {
//                    printf("> tid %zu is waiting  (w: %u)\n", (size_t)gettid(), waiters);
                    // we successfully started our wait
                    auto original = waiters;
                    while ((waiters = current.load()) >= 0) {
                        count++;
                        assert(waiters >= original); // waiters can only go up, until it goes negative
                    }
                    auto ret = return_value;
                    current++;
                    return ret;
                }
            } else {
                // ... or else we are (potentially) the breaking waiter, in which case we flip the sign of waiters
                // which unblocks the other waiters
//                std::lock_guard<std::mutex> guard(lock);
                waiters = current.load();
                if (waiters == break_count - 1) {
                    auto ret = return_value = breaker_function();
                    current.store(-waiters);
                    // printf("> tid %zu is breaking (w: %d)\n", (size_t)gettid(), waiters);
                    return ret;
                }

            }
        }
    }
};

struct cyclic_barrier : public cyclic_barrier_template<size_t> {
    cyclic_barrier(uint32_t count) : cyclic_barrier_template<size_t>{count, [](){ return 0; }} {}
};

#endif // guard