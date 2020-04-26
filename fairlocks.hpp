#ifndef FAIRLOCKS_HPP_
#define FAIRLOCKS_HPP_

#include <assert.h>
#include <sched.h>
#include <stddef.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>

namespace locks {

class spin_base {
protected:
    std::atomic<bool> islocked{};
public:
    void unlock();
};

struct spinlock_hot : spin_base {
    void lock();
};

struct spinlock_pause : spin_base {
    void lock();
};

struct spinlock_yield : spin_base {
    void lock();
};

using spin_f = int();

template <spin_f SPINF>
class ticket_template {
    std::atomic<size_t> dispenser{}, serving{};

public:
    void lock() {
        auto ticket = dispenser++;

        while (ticket != serving.load(std::memory_order_acquire))
            SPINF();
    }

    void unlock() {
        serving.store(serving.load() + 1, std::memory_order_release);
    }
};

static int nop() {
    return 0;
}

using ticket_spin   = ticket_template<nop>;
using yielding_spin = ticket_template<sched_yield>;

class blocking_ticket {
    std::atomic<size_t> dispenser{}, serving{};
    std::mutex mutex;
    std::condition_variable cvar;

public:
    void lock();

    void unlock();
};

class fifo_queued {
    struct queue_elem;

    std::mutex mutex;
    std::deque<queue_elem*> cvar_queue;
    bool locked = false;

public:
    void lock();

    void unlock();
};

/**
 * mutex3 from "Futexes Are Tricky"
 * https://akkadia.org/drepper/futex.pdf
 */
class mutex3 {
public:
    mutex3() : val(0) {}

    void lock();

    void unlock();
private:
    int val;
};

}  // namespace locks

#endif