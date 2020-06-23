#include "fairlocks.hpp"

#include <linux/futex.h>
#include <unistd.h>
#include <sys/syscall.h>

#ifdef __x86_64__
#include <immintrin.h>
inline void cpu_relax() { _mm_pause(); }
#elif defined(__aarch64__)
inline void cpu_relax() { asm volatile ("yield"); }
#else
inline void cpu_relax() {}
#endif

using namespace locks;

void spin_base::unlock() {
    islocked.store(false, std::memory_order_release);
}

template <typename S>
static inline void spin_impl(std::atomic<bool>& is_locked, S spinner) {
    do {
        while (is_locked.load(std::memory_order_relaxed)) {
            spinner();
        }
    } while (is_locked.exchange(true, std::memory_order_acquire));
}

void spinlock_hot::lock() { spin_impl(islocked, []{}); }

void spinlock_pause::lock() { spin_impl(islocked, cpu_relax); }

void spinlock_yield::lock() { spin_impl(islocked, []{ sched_yield(); }); }

void blocking_ticket::lock() {
    auto ticket = dispenser.fetch_add(1, std::memory_order_relaxed);

    if (ticket == serving.load(std::memory_order_acquire))
        return;

    std::unique_lock<std::mutex> lock(mutex);
    while (ticket != serving.load(std::memory_order_acquire)) {
        cvar.wait(lock);
    }
}

void blocking_ticket::unlock() {
    std::unique_lock<std::mutex> lock(mutex);
    auto s = serving.load(std::memory_order_relaxed) + 1;
    serving.store(s, std::memory_order_release);
    auto d = dispenser.load(std::memory_order_relaxed);
    assert(s <= d);
    if (s < d) {
        // wake waiters
        cvar.notify_all();
    }
}

struct fifo_queued::queue_elem {
    std::condition_variable cvar;
    bool owner = false;
};  

void fifo_queued::lock() {
    std::unique_lock<std::mutex> guard(mutex);
    if (!locked) {
        locked = true;
        return;
    }

    queue_elem node;
    cvar_queue.push_back(&node);

    do {
        node.cvar.wait(guard);
    } while (!node.owner);

    assert(locked && cvar_queue.front() == &node);
    cvar_queue.pop_front();
}

void fifo_queued::unlock() {
    std::unique_lock<std::mutex> guard(mutex);
    if (cvar_queue.empty()) {
        locked = false;
    } else {
        auto& next = cvar_queue.front();
        next->owner = true;
        next->cvar.notify_one();
    }
}

int cmpxchg(int& var, int old, int desired) {
    return __sync_val_compare_and_swap(&var, old, desired);
}

int xchg(int& var, int val) {
    return __atomic_exchange_n(&var, val, __ATOMIC_ACQUIRE);
}

int atomic_dec(int& var) {
    return __sync_fetch_and_sub(&var, 1);
}

/**
 * The futex related calls are cribbed from:
 * // https://github.com/eliben/code-for-blog/blob/master/2018/futex-basics/futex-basic-process.c
 */
int futex(int* uaddr, int futex_op, int val, const struct timespec* timeout,
          int* uaddr2, int val3) {
  return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
}


void futex_wait(int* futex_addr, int val) {
    int ret = futex(futex_addr, FUTEX_WAIT, val, NULL, NULL, 0);
    (void)ret;
    assert(ret == 0 || (ret == -1 && errno == EAGAIN));
}

void futex_wake(int* futex_addr, int nwait) {
    int futex_rc = futex(futex_addr, FUTEX_WAKE, nwait, NULL, NULL, 0);
    if (futex_rc == -1) {
      perror("futex wake");
      exit(1);
    }
}

void mutex3::lock() {
    int c;
    if ((c = cmpxchg(val, 0, 1)) != 0) {
        if (c != 2) {
            c = xchg(val, 2);
        }
        while (c != 0) {
            futex_wait(&val, 2);
            c = xchg(val, 2);
        }
    }
}
 
void mutex3::unlock() {
    if (atomic_dec(val) != 1) {
        // printf("%d unlock wake\n", tid);
        val = 0;
        futex_wake(&val, 1);
    } else {
        // printf("%d unlock fast\n", tid);
    }
}