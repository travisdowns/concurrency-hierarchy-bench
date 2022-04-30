/*
 * bench.cpp
 */

#include <err.h>
#include <error.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

#include "args-wrap.hpp"
#include "cyclic-barrier.hpp"
#include "fairlocks.hpp"
#include "fmt/format.h"
#include "hedley.h"
#include "stats.hpp"
#include "table.hpp"
#include "util.hpp"

#if USE_RDTSC
#include "tsc-support.hpp"
#define DefaultClock RdtscClock
#else
#define DefaultClock StdClock<std::chrono::high_resolution_clock>
#endif

using std::uint64_t;
using namespace std::chrono;

using namespace Stats;

using cal_f = uint64_t(size_t iters, size_t id);
using check_f = uint64_t();

#ifndef CHECK_RAC
#define CHECK_RAC 0
#endif

#if CHECK_RAC
#define IF_RAC(...) __VA_ARGS__
#else
#define IF_RAC(...)
#endif

struct test_func {
    // function pointer to the test function
    cal_f* func;
    const char* id;
    const char* description;
    check_f* check_func;
};

std::string to_string(const test_func& f) {
    return f.id;
}

/**
 * Keeps a counter per thread, readers need to sum
 * the counters from all active threads and add the
 * accumulated value from dead threads.
 */
class tls_counter {
    std::atomic<uint64_t> counter{0};

    /* protects all_counters and accumulator */
    static std::mutex lock;
    /* list of all active counters */
    static std::vector<tls_counter *> all_counters;
    /* accumulated value of counters from dead threads */
    static uint64_t accumulator;
    /* per-thread tls_counter object */
    static thread_local tls_counter tls;

    /** add ourself to the counter list */
    tls_counter() {
        std::lock_guard<std::mutex> g(lock);
        all_counters.push_back(this);
    }

    /**
     * destruction means the thread is going away, so
     * we stash the current value in the accumulator and
     * remove ourselves from the array
     */
    ~tls_counter() {
        std::lock_guard<std::mutex> g(lock);
        accumulator += counter.load(std::memory_order_relaxed);
        all_counters.erase(std::remove(all_counters.begin(), all_counters.end(), this), all_counters.end());
    }

    void incr() {
        auto cur = counter.load(std::memory_order_relaxed);
        counter.store(cur + 1, std::memory_order_relaxed);
    }

public:

    static uint64_t read() {
        std::lock_guard<std::mutex> g(lock);
        uint64_t sum = 0, count = 0;
        for (auto h : all_counters) {
            sum += h->counter.load(std::memory_order_relaxed);
            count++;
        }
        return sum + accumulator;
    }

    HEDLEY_NEVER_INLINE
    static void increment() {
        tls.incr();
    }
};

std::mutex tls_counter::lock;
std::vector<tls_counter *> tls_counter::all_counters;
uint64_t tls_counter::accumulator;
thread_local tls_counter tls_counter::tls;

HEDLEY_NEVER_INLINE
uint64_t tls_add(size_t iters, size_t id) {
    while (iters--) {
        tls_counter::increment();
    }
    return 0;
}

static volatile uint64_t plain_counter;
// racy
uint64_t plain_add(size_t iters, size_t id) {
    uint64_t rac_count = 0;
    IF_RAC(uint64_t last = plain_counter - 1;)
    while (iters--) {
        auto cur = plain_counter++;
        IF_RAC(rac_count += (cur == last + 1); last = cur;)
    }
    return rac_count;
}

static constexpr size_t NUM_FS_COUNTERS = 64;
// Deterministic packed false-sharing.
alignas(64) static volatile uint8_t fs_counters[NUM_FS_COUNTERS];
uint64_t fs_add(size_t iters, size_t id) {
    if (id >= NUM_FS_COUNTERS) {
        error(EXIT_FAILURE, errno, "thread count exceeds structure %ld>=%ld",
              id, NUM_FS_COUNTERS);
    }
    fs_counters[id] = 0;
    while (iters--) {
        fs_counters[id]++;
    }
    return fs_counters[id] != (iters & 0xff);
}

/**
 * Simple counter which just uses a relaxed std::atomic increment.
 */
class atomic_add_counter {
    std::atomic<uint64_t> counter{};
public:
    uint64_t operator++(int) {
        return counter.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t read() const {
        return counter.load(std::memory_order_relaxed);
    }
};

template <typename std::memory_order O>
class add_template {
    std::atomic<uint64_t> counter{};
public:
    uint64_t operator++(int) {
        return counter.fetch_add(1, O);
    }

    uint64_t read() const {
        return counter.load(std::memory_order_relaxed);
    }
};

atomic_add_counter atomic_counter;

class atomic_cas_counter {
    std::atomic<uint64_t> counter{0};
public:
    uint64_t operator++(int) {
        auto cur = counter.load();
        while (!counter.compare_exchange_weak(cur, cur + 1))
            ;
        return cur;
    }

    uint64_t read() const {
        return counter.load(std::memory_order_relaxed);
    }
};

atomic_cas_counter cas_counter;

struct multi_holder {
    alignas(64) std::atomic<uint64_t> counter;
};

/**
 * Uses a total of NUM_COUNTERS to represent the count, allowing
 * the different CPUs to use different. CAS failure is used as the
 * hint that two CPUs are concurrently sharing a counter slot and
 * so we should adjust the index.
 */
class cas_multi_counter {
    static constexpr size_t NUM_COUNTERS = 64;
    static thread_local size_t idx;

    multi_holder array[NUM_COUNTERS];

public:

    /** increment the logical counter value */
    uint64_t operator++(int) {
        while (true) {
            auto& counter = array[idx].counter;

            auto cur = counter.load();
            if (counter.compare_exchange_strong(cur, cur + 1)) {
                return cur;
            }

            // CAS failure indicates contention,
            // so try again at a different index
            idx = (idx + 1) % NUM_COUNTERS;
        }
    }

    /**
     * Read the current value of the counter by summing all
     * physical counters.
     */
    uint64_t read() {
        uint64_t sum = 0;
        for (auto& h : array) {
            sum += h.counter.load();
        }
        return sum;
    }
};

thread_local size_t cas_multi_counter::idx;

cas_multi_counter cas_mc;

/**
 * Generic benchmark method for objects that offer ++ and read() methods.
 * @param iters number of iterations to run the benchmark
 * @return the rac count: the number of times increments were consecutive
 */
template <typename T>
HEDLEY_NEVER_INLINE
uint64_t bench_template(T& counter, size_t iters) {
    uint64_t rac_count = 0;
    IF_RAC(uint64_t last = counter.read() - 1;)
    while (iters--) {
        auto cur = counter++;
        IF_RAC(rac_count += (cur == last + 1); last = cur;)
    }
    return rac_count;
}

// template for any bench which just uses a lock around
// the addition, you provide the lock type T
template <typename T>
struct adaptor {

    static T lock;
    static uint64_t counter;

    static uint64_t bench(size_t iters, size_t id) {
        uint64_t rac_count = 0;
        IF_RAC(uint64_t last = counter - 1;)
        while (iters--) {
            std::lock_guard<T> holder(lock);
            auto cur = counter++;
            IF_RAC(rac_count += (cur == last + 1); last = cur;)
        }
        return rac_count;
    }

    static uint64_t get_counter() {
        return counter;
    }
};

template <typename T>
alignas(64) T adaptor<T>::lock;

template <typename T>
uint64_t adaptor<T>::counter;

/** make a test_func object from adaptor<T> using the given lock type T */
template <typename T>
test_func make_from_lock(const char *name, const char* desc = "desc") {
    return { adaptor<T>::bench, name, desc, adaptor<T>::get_counter };
}

template <typename T, T* O>
test_func make_from_type(const char *name) {
    return { [](size_t i, size_t id) { return bench_template(*O, i); } , name , "desc" , []{ return O->read(); } };
}

template <typename T>
test_func make_unchecked(const char *name) {
    static T counter;
    return { [](size_t i, size_t id) {
        return bench_template(counter, i);
    }
    , name , "desc" , nullptr };
}

std::vector<test_func> ALL_FUNCS = {
        {plain_add                                      , "plain add"  , "desc" , nullptr }                             ,
        {fs_add                                         , "fs add"  , "desc" , nullptr }                             ,
        {tls_add                                        , "tls add"    , "desc" , tls_counter::read }                   ,
        make_from_type<atomic_add_counter, &atomic_counter>("atomic add"),
        make_from_type<atomic_cas_counter, &cas_counter>("cas add"),
        make_from_type<cas_multi_counter, &cas_mc>("cas multi"),
        make_from_lock<std::mutex>("mutex add")                   ,
        make_from_lock<locks::spinlock_hot>("pure spin")          ,
        make_from_lock<locks::spinlock_pause>("pause spin")       ,
        make_from_lock<locks::spinlock_yield>("yield spin")       ,
        make_from_lock<locks::ticket_spin>("ticket spin")         ,
        make_from_lock<locks::ticket_yield>("ticket yield")       ,
        make_from_lock<locks::blocking_ticket>("ticket blocking") ,
        make_from_lock<locks::fifo_queued>("queued fifo")         ,
        make_from_lock<locks::mutex3>("mutex3")                   ,
        make_unchecked<atomic_add_counter>("aadd-1"),
        make_unchecked<add_template<std::memory_order_relaxed>>("aadd-relaxed"),
        make_unchecked<add_template<std::memory_order_acquire>>("aadd-acquire"),
        make_unchecked<add_template<std::memory_order_release>>("aadd-release"),
        make_unchecked<add_template<std::memory_order_acq_rel>>("aadd-acq_rel"),
        make_unchecked<add_template<std::memory_order_seq_cst>>("aadd-seq_cst"),
};

static void pin_to_cpu(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1) {
        error(EXIT_FAILURE, errno, "could not pin to CPU %d", cpu);
    }
}

/** args */
static argsw::ArgumentParser parser{"conc-bench: Demonstrate concurrency perforamnce levels"};
static argsw::HelpFlag help{parser, "help", "Display this help menu", {"help"}};
static argsw::Flag arg_force_tsc_cal{parser, "force-tsc-calibrate",
    "Force manual TSC calibration loop, even if cpuid TSC Hz is available", {"force-tsc-calibrate"}};
static argsw::Flag arg_no_pin{parser, "no-pin",
    "Don't try to pin threads to CPU - gives worse results but works around affinity issues on TravisCI", {"no-pin"}};
static argsw::Flag arg_verbose{parser, "verbose", "Output more info", {"verbose"}};
static argsw::Flag arg_list{parser, "list", "List the available tests and their descriptions", {"list"}};
static argsw::Flag arg_csv{parser, "", "Output a csv table instead of the default", {"csv"}};
static argsw::Flag arg_progress{parser, "", "Display progress to stdout", {"progress"}};

static argsw::Flag arg_hyperthreads{parser, "allow-hyperthreads", "By default we try to filter down the available cpus to include only physical cores, but "
    "with this option we'll use all logical cores meaning you'll run two tests on cores with hyperthreading", {"allow-hyperthreads"}};
static argsw::ValueFlag<std::string> arg_algos{parser, "TEST-ID", "Run only the algorithms in the comma separated list", {"algos"}};
static argsw::ValueFlag<size_t> arg_batch{parser, "BATCH-SIZE", "Make BATCH-SIZE calls to the function under test in between checks for test termination", {"batch"}, 1000};
static argsw::ValueFlag<uint64_t> arg_trial_time{parser, "TIME-MS", "The time for each trial in ms", {"trial-time"}, 10};
static argsw::ValueFlag<uint32_t> arg_min_threads{parser, "MIN", "The minimum number of threads to use", {"min-threads"}, 1};
static argsw::ValueFlag<uint32_t> arg_max_threads{parser, "MAX", "The maximum number of threads to use", {"max-threads"}};
static argsw::ValueFlag<uint64_t> arg_warm_ms{parser, "MILLISECONDS", "Warmup milliseconds for each thread after pinning (default 100)", {"warmup-ms"}, 10};

bool verbose;

/**
 * Clock that uses std::chrono clocks.
 */
template <typename CHRONO_CLOCK>
struct StdClock {
    using now_t   = decltype(CHRONO_CLOCK::now());
    using delta_t = typename CHRONO_CLOCK::duration;

    static now_t now() {
        return CHRONO_CLOCK::now();
    }

    /* accept the result of subtraction of durations and convert to nanos */
    static uint64_t to_nanos(delta_t d) {
        return duration_cast<std::chrono::nanoseconds>(d).count();
    }

    static uint64_t now_to_nanos(now_t tp) {
        return to_nanos(tp.time_since_epoch());
    }
};

#if USE_RDTSC
/**
 * Clock that uses rdtsc if available. Generally
 * someone faster than the std::chrono clocks and
 * plenty accurate on any modern x86 machine.
 */
struct RdtscClock {
    using now_t   = uint64_t;
    using delta_t = uint64_t;

    static now_t now() {
        _mm_lfence();
        now_t ret = rdtsc();
        _mm_lfence();
        return ret;
    }

    /* accept the result of subtraction of durations and convert to nanos */
    static uint64_t now_to_nanos(now_t diff) {
        static double tsc_to_nanos = 1000000000.0 / tsc_freq();
        return diff * tsc_to_nanos;
    }

    static uint64_t to_nanos(delta_t diff) {
        return now_to_nanos(diff);
    }

    static uint64_t tsc_freq() {
        static uint64_t freq = get_tsc_freq(arg_force_tsc_cal);
        return freq;
    }

};

#endif

template <typename CLOCK = DefaultClock>
static uint64_t now_nanos() {
    return CLOCK::now_to_nanos(CLOCK::now());
}

struct timer_barrier : public cyclic_barrier_template<uint64_t> {
    timer_barrier(uint32_t break_count, uint64_t offset_nanos) :
          cyclic_barrier_template<uint64_t>{break_count, [=](){ return now_nanos() + offset_nanos; }} {}
};

struct usage_error : public std::runtime_error {
    using runtime_error::runtime_error;
};

/* find the test that exactly matches the given ID or return nullptr if not found */
test_func find_one_test(const std::string& id) {
    for (const auto& t : ALL_FUNCS) {
        if (id == t.id) {
            return t;
        }
    }
    throw usage_error("benchmark " + id + " not found");
}


struct result {
    uint64_t    start_ts;   // start timestamp
    uint64_t      end_ts;   // end   timestamp
    uint64_t delta_nanos; // end - start
    uint64_t total_iters = 0;
    uint64_t reacquires  = 0; // number of times the counter was incremement consecutively by this thread
};

struct result_holder {
    test_func spec;
    size_t iters;
    size_t trial;
    uint64_t nanos; // total runtime for all trials
    std::vector<result> results; // will have spec.count() elements

    result_holder(test_func spec, size_t iters, size_t trial, uint64_t nanos)
        : spec{std::move(spec)}, iters{iters}, trial{trial}, nanos{nanos} {}

    template <typename InitT, typename BinOp>
    double inner_accum(InitT init, BinOp op) const {
        InitT sum = init;
        for (const auto& result : results) {
            sum = op(sum, result);
        }
        return static_cast<double>(sum);
    }

    template <typename E>
    double inner_sum(E e) const {
        double a = 0;
        for (const auto& result : results) {
            a += e(result);
        }
        return a;
    }

    typedef uint64_t (result::*ir_u64);

    double inner_sum(result_holder::ir_u64 pmem) const {
        return inner_sum(std::mem_fn(pmem));
    }

    /** calculate the overlap ratio based on the start/end timestamps */
    double get_overlap1() const {
        std::vector<std::pair<uint64_t, uint64_t>> ranges = transformv(results, [](const result& r){ return std::make_pair(r.start_ts, r.end_ts);} );
        return conc_ratio(ranges.begin(), ranges.end());
    }
};

struct warmup {
    uint64_t millis;
    explicit warmup(uint64_t millis) : millis{millis} {}

    long warm() {
        int64_t start = (int64_t)now_nanos();
        long iters = 0;
        while ((now_nanos() - start) < 1000000u * millis) {
            iters++;
        }
        return iters;
    }
};

constexpr int WARMUP = 3;
constexpr int TRIALS = 15;

struct test_thread {
    size_t id;
    int cpu; // pin to this cpu id
    timer_barrier* start_barrier;

    // results, one per trial
    std::vector<result> results;
    size_t total_iters = 0; // total iterations across all trials and warmups

    /* input */
    test_func func;
    size_t iters;
    uint64_t trial_nanos;

    std::thread thread;

    test_thread(size_t id, int cpu, timer_barrier& start_barrier, const test_func& func, size_t iters,
                uint64_t trial_nanos) :
        id{id}, cpu{cpu}, start_barrier{&start_barrier}, func{func},
        iters{iters}, trial_nanos{trial_nanos}, thread{std::ref(*this)}
    {
        if (verbose) fmt::print("Constructed test in thread {}, this = {}\n", id, (void *)this);
    }

    test_thread(const test_thread&) = delete;
    test_thread(test_thread&&) = delete;
    void operator=(const test_thread&) = delete;

    void operator()() {
        using CLOCK = DefaultClock;
        if (cpu >= 0) {
            pin_to_cpu(cpu);
        }
        warmup w{arg_warm_ms.Get()};
        long warms = w.warm();
        if (verbose) fmt::print("{:2} Warmup iters {}\n", id, warms);

        results.resize(TRIALS);

        for (int trial = -WARMUP; trial < TRIALS; trial++) {
            auto deadline = start_barrier->wait();
            if (verbose) fmt::print("{:2} Thread deadline: {}\n", id, deadline);

            result result{};
            result.start_ts = now_nanos();

            auto f = func.func;
            CLOCK::delta_t delta;

            uint64_t reacquires = 0;
            auto t0             = CLOCK::now();
            do {
                reacquires += f(iters, id);
                result.total_iters += iters;
            } while (now_nanos() < deadline);
            auto t1             = CLOCK::now();
            result.start_ts = CLOCK::now_to_nanos(t0);
            result.end_ts = CLOCK::now_to_nanos(t1);
            result.delta_nanos  = CLOCK::to_nanos(t1 - t0);
            result.reacquires  = reacquires;

            // this indexing eliminates the warmup runs from the results
            results.at(trial < 0 ? 0 : trial) = result;
            total_iters += result.total_iters;
        }
    }
};

template <typename E>
double aggregate_results(const std::vector<result>& results, E e) {
    double a = 0;
    for (const auto& result : results) {
        a += e(result);
    }
    return a;
}

/**
 * Aggregate normalized results sum(N(ri)) / sum(M(ri)).
 */
template <typename N, typename M>
double aggregate_results(const std::vector<result>& results, N top, M bottom) {
    return aggregate_results(results, top) / aggregate_results(results, bottom);
}

using Row = table::Row;
auto LEFT = table::ColInfo::LEFT;
auto RIGHT = table::ColInfo::RIGHT;
using extractor = std::function<void(Row& row, const result_holder& holder)>;

struct column {
    const char* heading;
    table::ColInfo::Justify j;
    extractor e;
};

column make_inner(const char* name, result_holder::ir_u64 pmem, const char* format = "%.1f", double mult = 1.0) {
    return column{name, RIGHT,
            [=](Row& r, const result_holder& h) {
                r.addf(format, h.inner_sum(pmem) * mult);
            }
    };
}

template <typename F>
column make_inner2(const char* name, uint64_t init, F f, const char* format = "%.1f") {
    return column{name, RIGHT, [=](Row& r, const result_holder& h) {
                      auto res = h.inner_accum<uint64_t>(init, f);
                      r.addf(format, res);
                  }};
}

static column col_core{"Cores", RIGHT, [](Row& r, const result_holder& h) { r.add(h.results.size()); }};
static column col_trial{"Trial", RIGHT, [](Row& r, const result_holder& h) { r.add(h.trial); }};
static column col_desc{"Description", LEFT, [](Row& r, const result_holder& h) { r.add(h.spec.description); }};
static column col_id{"Implementation", LEFT, [](Row& r, const result_holder& h) { r.add(h.spec.id); }};
static column col_olap{"Overlap", RIGHT, [](Row& r, const result_holder& h) { r.addf("%.3f", h.get_overlap1()); }};
static column col_ns{"Nanos/Op", RIGHT, [](Row& r, const result_holder& h) {
                         r.addf("%.1f", aggregate_results(
                                                h.results, [](const result& r) { return r.delta_nanos; },
                                                [](const result& r) { return r.total_iters; }));
                     }};
static column col_cs{"Clock sum ms", RIGHT, [](Row& r, const result_holder& h) {
                         r.addf("%.1f", aggregate_results(h.results, [](const result& r) { return r.delta_nanos / 1000000.; }));
                     }};

static column col_rt{"Runtime ms", RIGHT, [](Row& r, const result_holder& h) { r.addf("%.0f", h.nanos / 1000000.); }};
static column col_iter = make_inner("Total I", &result::total_iters, "%.0f");
static column col_reac = make_inner("Reac" , &result::reacquires);
static column col_mini = make_inner2("Min I", std::numeric_limits<uint64_t>::max(),
        [](uint64_t min, const result& r) { return std::min(min, r.total_iters); });
static column col_maxi = make_inner2("Max I", std::numeric_limits<uint64_t>::min(),
        [](uint64_t max, const result& r) { return std::max(max, r.total_iters); });
static column col_runs{"Rlen", RIGHT, [](Row& r, const result_holder& h) {
    auto reac  = h.inner_sum(&result::reacquires);
    auto total = h.inner_sum(&result::total_iters);
    r.addf("%.1f", (double)total / (1 + total - reac));
}};

void report_results(const std::vector<result_holder>& results_list) {

    auto cols = arg_csv ?
        std::vector<column>{col_trial, col_core, col_id, col_ns, col_iter, col_runs} :
        std::vector<column>{col_trial, col_core, col_id, col_olap, col_ns, col_cs, col_rt, col_iter,
        col_mini, col_maxi
        IF_RAC(, col_reac, col_runs)
        };

    // report
    table::Table table;
    table.setColColumnSeparator(" | ");
    auto &header = table.newRow();

    for (size_t c = 0; c < cols.size(); c++) {
        auto& col = cols[c];
        header.add(col.heading);
        table.colInfo(c).justify = col.j;
    }

    for (const result_holder& holder : results_list) {
        auto& row = table.newRow();
        for (auto& c : cols) {
            c.e(row, holder);
        }
    }

    fmt::print("{}", arg_csv ? table.csv_str() : table.str());
}

void list_tests() {
    table::Table table;
    table.newRow().add("ID").add("Description");
    for (auto& t : ALL_FUNCS) {
        table.newRow().add(t.id).add(t.description);
    }
    fmt::print("Available tests:\n\n{}\n", table.str().c_str());
}

std::vector<int> get_cpus() {
    cpu_set_t cpu_set;
    if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set)) {
        err(EXIT_FAILURE, "failed while getting cpu affinity");
    }
    std::vector<int> ret;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &cpu_set)) {
            ret.push_back(cpu);
        }
    }
    return ret;
}


int main(int argc, char** argv) {

    parser.ParseCLI(argc, argv,
            [](const std::string& help) {
                fmt::print("{}\n", help);
                exit(EXIT_SUCCESS);
            }, [](const std::string& parse_error) {
                fmt::print(stderr, "ERROR while parsing arguments: {}\n", parse_error);
                fmt::print(stderr, "\nUsage:\n{}\n", parser.Help());
                exit(EXIT_FAILURE);
            });

    // if csv mode is on, only the table should go to stdout
    // the rest goes to stderr
    FILE* out = arg_csv ? stderr : stdout;
#if USE_RDTSC
    set_logging_file(out);
#endif

    if (arg_list) {
        list_tests();
        exit(EXIT_SUCCESS);
    }

    verbose = arg_verbose;
    bool is_root = (geteuid() == 0);
    auto batch_size = arg_batch.Get();
    auto trial_time = arg_trial_time.Get();
    std::vector<int> cpus = get_cpus();

#if USE_RDTSC
    fmt::print(out, "tsc_freq = {:.1f} MHz ({})\n", RdtscClock::tsc_freq() / 1000000.0, get_tsc_cal_info(arg_force_tsc_cal));
#endif
    fmt::print(out, "Running as root      : [{}]\n", is_root     ? "YES" : "NO ");
    fmt::print(out, "CPU pinning enabled  : [{}]\n", !arg_no_pin ? "YES" : "NO ");
    fmt::print(out, "available CPUs ({:4}): [{}]\n", cpus.size(), join(cpus, ", ").c_str());
    fmt::print(out, "get_nprocs_conf()    : [{}]\n", get_nprocs_conf());
    fmt::print(out, "get_nprocs()         : [{}]\n", get_nprocs());
    fmt::print(out, "batch count          : [{}]\n", batch_size);
    fmt::print(out, "trial time           : [{} ms]\n", trial_time);

    auto min_threads = arg_min_threads.Get();
    auto max_threads = arg_max_threads ? arg_max_threads.Get() : (uint32_t)cpus.size();

    std::vector<test_func> specs;
    if (arg_algos) {
        auto arglist = split(arg_algos.Get(), ",");
        for (auto& s : ALL_FUNCS) {
            if (std::find(arglist.begin(), arglist.end(), s.id) != arglist.end()) {
                specs.push_back(s);
            }
        }
    } else {
        specs.insert(specs.begin(), std::begin(ALL_FUNCS), std::end(ALL_FUNCS));
    }

    auto total_benches = specs.size() * (max_threads - min_threads + 1);
    std::vector<result_holder> results_list;
    for (auto func : specs) {
        for (auto count = min_threads; count <= max_threads; count++) {

            auto nanos_before = now_nanos();

            const bool has_check = func.check_func;
            uint64_t counter_before = has_check ? func.check_func() : 0;

            // run
            timer_barrier start{count, trial_time * 1000000};
            std::deque<test_thread> threads;
            for (size_t tid = 0; tid < count; tid++) {
                int cpu = arg_no_pin ? -1 : cpus.at(tid % cpus.size());
                if (verbose) fmt::print("thread {} pinned to {}\n", tid, cpu);
                threads.emplace_back(tid, cpu, start, func, batch_size, trial_time * 1000000);
            }

            uint64_t total_iters = 0;
            for (auto& t : threads) {
                t.thread.join();
                total_iters += t.total_iters;
            }

            for (size_t trial = 0; trial < TRIALS; trial++) {
                results_list.emplace_back(func, batch_size, trial, now_nanos() - nanos_before);
                for (auto& t : threads) {
                    auto& r = t.results.at(trial);
                    results_list.back().results.push_back(r);
                }
            }

            auto counter_after = has_check ? func.check_func() : 0;
            auto total_counter_delta =  counter_after - counter_before;
            if (has_check && total_counter_delta != total_iters) {
                throw std::runtime_error(fmt::format("threads {}, algo {} failed check: {} actual vs {} expected",
                        count, func.id, total_counter_delta, total_iters));
            }

            if (arg_progress) {
                fmt::print(stderr, "{}/{}: finished {} with {} threads in {} ms\n",  results_list.size() / TRIALS,
                           total_benches, to_string(func), count, (now_nanos() - nanos_before) / 1000000);
            }
        }
    }

    report_results(results_list);

    return EXIT_SUCCESS;
}




