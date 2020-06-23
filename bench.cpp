/*
 * bench.cpp
 */

#include <err.h>
#include <error.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
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
#include <limits>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "args-wrap.hpp"
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

using cal_f = uint64_t(uint64_t iters);
using check_f = uint64_t();

#define CHECK_RAC 0

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
    check_f* rac_func;
};

/**
 * Keeps a counter per thread, readers need to sum
 * the counters from all active threads and add the
 * accumulated value from dead threads.
 */
class tls_counter {
    std::atomic<uint64_t> counter{0};

    /* protects all_counters */
    static std::mutex lock;
    /* list of all active counters */
    static std::vector<tls_counter *> all_counters;
    /* accumulated value of counters from dead threads */
    static std::atomic<uint64_t> accumulator;
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
std::atomic<uint64_t> tls_counter::accumulator;
thread_local tls_counter tls_counter::tls;

HEDLEY_NEVER_INLINE
uint64_t tls_add(size_t iters) {
    while (iters--) {
        tls_counter::increment();
    }
    return 0;
}

static volatile uint64_t plain_counter;
// racy
uint64_t plain_add(size_t iters) {
    uint64_t rac_count = 0;
    IF_RAC(uint64_t last = plain_counter - 1;)
    while (iters--) {
        auto cur = plain_counter++;
        IF_RAC(rac_count += (cur == last + 1); last = cur;)
    }
    return rac_count;
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

    static uint64_t bench(size_t iters) {
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
    return { [](size_t i) { return bench_template(*O, i); } , name , "desc" , []{ return O->read(); } };
}

std::vector<test_func> ALL_FUNCS = {
        {plain_add                                      , "plain add"  , "desc" , nullptr }                             ,
        {tls_add                                        , "tls add"    , "desc" , tls_counter::read }                   ,
        make_from_type<atomic_add_counter, &atomic_counter>("atomic add"),
        make_from_type<atomic_cas_counter, &cas_counter>("cas add"),
        make_from_type<cas_multi_counter, &cas_mc>("cas multi"),
        make_from_lock<std::mutex>("mutex add")                   ,
        make_from_lock<locks::spinlock_hot>("pure spin")          ,
        make_from_lock<locks::spinlock_pause>("pause spin")       ,
        make_from_lock<locks::spinlock_yield>("yield spin")       ,
        make_from_lock<locks::ticket_spin>("ticket spin")         ,
        make_from_lock<locks::ticket_yield>("ticket yield")      ,
        make_from_lock<locks::blocking_ticket>("ticket blocking") ,
        make_from_lock<locks::fifo_queued>("queued fifo")         ,
        make_from_lock<locks::mutex3>("mutex3")                   ,
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
static argsw::Flag arg_nobarrier{parser, "no-barrier", "Don't sync up threads before each test (debugging only)", {"no-barrier"}};
static argsw::Flag arg_list{parser, "list", "List the available tests and their descriptions", {"list"}};
static argsw::Flag arg_csv{parser, "", "Output a csv table instead of the default", {"csv"}};

static argsw::Flag arg_hyperthreads{parser, "allow-hyperthreads", "By default we try to filter down the available cpus to include only physical cores, but "
    "with this option we'll use all logical cores meaning you'll run two tests on cores with hyperthreading", {"allow-hyperthreads"}};
static argsw::ValueFlag<std::string> arg_algos{parser, "TEST-ID", "Run only the algorithms in the comma separated list", {"algos"}};
static argsw::ValueFlag<size_t> arg_iters{parser, "ITERS", "Run the test loop ITERS times (default 100000)", {"iters"}, 10};
static argsw::ValueFlag<int> arg_min_threads{parser, "MIN", "The minimum number of threads to use", {"min-threads"}, 1};
static argsw::ValueFlag<int> arg_max_threads{parser, "MAX", "The maximum number of threads to use", {"max-threads"}};
static argsw::ValueFlag<uint64_t> arg_warm_ms{parser, "MILLISECONDS", "Warmup milliseconds for each thread after pinning (default 100)", {"warmup-ms"}, 100};


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

/*
 * The result of the run_test method, with only the stuff
 * that can be calculated from within that method.
 */
struct inner_result {
    DescriptiveStats elapsedns_stats;
    uint64_t ostart_ts; // start of the benchmark
    uint64_t oend_ts1; // end of the timed portion of the benchmark (not including spinning iters
    uint64_t oend_ts2; // includes spinning iters waiting for other tests to finish
    uint64_t istart_ts, iend_ts; // start and end timestamps for the "critical" benchmark portion
    uint64_t timed_iters = 0; // the number of iterationreac the ctimed part of the test
    uint64_t total_iters = 0;
    uint64_t reacquires  = 0; // number of times the counter was incremement consecutively by this thread
};

/*
 * Calculate the frequency of the CPU based on timing a tight loop that we expect to
 * take one iteration per cycle.
 *
 * ITERS is the base number of iterations to use: the calibration routine is actually
 * run twice, once with ITERS iterations and once with 2*ITERS, and a delta is used to
 * remove measurement overhead.
 */
struct hot_barrier {
    size_t break_count;
    std::atomic<size_t> current;
    hot_barrier(size_t count) : break_count(count), current{0} {}

    /* increment the arrived count of the barrier (do this once per thread generally) */
    void increment() {
        current++;
    }

    /* return true if all the threads have arrived, never blocks */
    bool is_broken() {
        return current.load() == break_count;
    }

    /* increment and hot spin on the waiter count until it hits the break point, returns the spin count in case you care */
    long wait() {
        increment();
        long count = 0;
        while (!is_broken()) {
            count++;
        }
        return count;
    }
};


template <typename CLOCK, size_t TRIES = 17, size_t WARMUP = 2>
inner_result run_test(cal_f* func, size_t iters, hot_barrier *barrier) {

    std::array<typename CLOCK::delta_t, TRIES> results;

    inner_result result;

    result.ostart_ts = now_nanos();
    for (size_t w = 0; w < WARMUP + 1; w++) {
        uint64_t reacquires = 0;
        result.istart_ts = now_nanos();
        for (size_t r = 0; r < TRIES; r++) {
            auto t0 = CLOCK::now();
            reacquires += func(iters);
            auto t1 = CLOCK::now();
            results[r] = t1 - t0;
        }
        result.iend_ts = now_nanos();
        result.reacquires = reacquires;
    }

    result.oend_ts1 = now_nanos();

    // this loop keeps running the test function until all other
    // threads are done too, to keep the environment consistent
    for (barrier->increment(); !barrier->is_broken();) {
        func(iters);
        result.total_iters += iters;
    }
    result.oend_ts2 = now_nanos();

    result.timed_iters = TRIES * iters;
    result.total_iters += (WARMUP + 1) * TRIES * iters;

    std::array<uint64_t, TRIES> nanos = {};
    std::transform(results.begin(), results.end(), nanos.begin(), CLOCK::to_nanos);
    result.elapsedns_stats = get_stats(nanos.begin(), nanos.end());

    return result;
}

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
    static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    inner_result    inner;

    uint64_t  start_ts;  // start timestamp
    uint64_t    end_ts;  // end   timestamp
};

struct result_holder {
    test_func spec;
    size_t iters;
    std::vector<result> results; // will have spec.count() elements

    result_holder(test_func spec, size_t iters) : spec{std::move(spec)}, iters{iters} {}

    template <typename E>
    double inner_sum(E e) const {
        double a = 0;
        for (const auto& result : results) {
            a += e(result.inner);
        }
        return a;
    }

    typedef uint64_t (inner_result::*ir_u64);

    double inner_sum(result_holder::ir_u64 pmem) const {
        return inner_sum(std::mem_fn(pmem));
    }

    /** calculate the overlap ratio based on the start/end timestamps */
    double get_overlap1() const {
        std::vector<std::pair<uint64_t, uint64_t>> ranges = transformv(results, [](const result& r){ return std::make_pair(r.start_ts, r.end_ts);} );
        return conc_ratio(ranges.begin(), ranges.end());
    }

    /** calculate the overlap ratio based on the start/end timestamps */
    double get_overlap2() const {
        std::vector<std::pair<uint64_t, uint64_t>> ranges = transformv(results, [](const result& r){ return std::make_pair(r.inner.istart_ts, r.inner.iend_ts);} );
        return conc_ratio(ranges.begin(), ranges.end());
    }

    /** calculate the inner overlap ratio based on the start/end timestamps */
    double get_overlap3() const {
        auto orange = transformv(results, [](const result& r){ return std::make_pair(r.inner.ostart_ts, r.inner.oend_ts2);} );
        auto irange = transformv(results, [](const result& r){ return std::make_pair(r.inner.istart_ts, r.inner.iend_ts);} );
        return nconc_ratio(orange.begin(), orange.end(), irange.begin(), irange.end());
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

struct test_thread {
    size_t id;
    int cpu; // pin to this cpu id
    hot_barrier* start_barrier;
    hot_barrier* stop_barrier;

    /* output */
    result res;

    /* input */
    test_func test;
    size_t iters;

    std::thread thread;

    test_thread(size_t id, int cpu, hot_barrier& start_barrier, hot_barrier& stop_barrier, const test_func& test, size_t iters) :
        id{id}, cpu{cpu}, start_barrier{&start_barrier}, stop_barrier{&stop_barrier}, test{test},
        iters{iters}, thread{std::ref(*this)}
    {
        if (verbose) fmt::print("Constructed test in thread {}, this = {}\n", id, (void *)this);
    }

    test_thread(const test_thread&) = delete;
    test_thread(test_thread&&) = delete;
    void operator=(const test_thread&) = delete;

    void operator()() {
        if (cpu >= 0) {
            pin_to_cpu(cpu);
        }
        warmup w{arg_warm_ms.Get()};
        long warms = w.warm();
        if (verbose) fmt::print("{:2} Warmup iters {}\n", id, warms);
        if (!arg_nobarrier) {
            long count = start_barrier->wait();
            if (verbose) fmt::print("{:2} Thread loop count: {}\n", id, count);
        }
        res.start_ts = now_nanos();
        res.inner = run_test<DefaultClock>(test.func, iters, stop_barrier);
        res.end_ts = now_nanos();
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

using Row = table::Row;
auto LEFT = table::ColInfo::LEFT;
auto RIGHT = table::ColInfo::RIGHT;
using extractor = std::function<void(Row& row, const result_holder& holder)>;

struct column {
    const char* heading;
    table::ColInfo::Justify j;
    extractor e;
};

column make_inner(const char* name, result_holder::ir_u64 pmem, const char* format = "%.1f") {
    return column{name, RIGHT,
            [=](Row& r, const result_holder& h) {
                r.addf(format, h.inner_sum(pmem));
            }
    };
}



static column col_core{"Cores", RIGHT, [](Row& r, const result_holder& h){ r.add(h.results.size()); }};
static column col_desc{"Description", LEFT, [](Row& r, const result_holder& h){ r.add(h.spec.description); }};
static column col_id  {"Implementation", LEFT, [](Row& r, const result_holder& h){ r.add(h.spec.id); }};
static column col_olap{"Overlap", RIGHT, [](Row& r, const result_holder& h){ r.addf("%.3f", h.get_overlap3()); }};
static column col_ns  {"Nanos", RIGHT, [](Row& r, const result_holder& h) {
    r.addf("%.1f", aggregate_results(h.results, [](const result& r){ return r.inner.elapsedns_stats.getMedian(); })
                / (h.iters * h.results.size())); }};

static column col_iter  = make_inner("Iters", &inner_result::timed_iters, "%.0f");
static column col_titer = make_inner("Total", &inner_result::total_iters, "%.0f");
static column col_reac  = make_inner("Reac" , &inner_result::reacquires);
static column col_runs{"Rlen", RIGHT, [](Row& r, const result_holder& h) {
    auto reac  = h.inner_sum(&inner_result::reacquires);
    auto total = h.inner_sum(&inner_result::timed_iters);
    r.addf("%.1f", (double)total / (1 + total - reac));
}};

void report_results(const std::vector<result_holder>& results_list) {

    auto cols = arg_csv ?
        std::vector<column>{col_core, col_id, col_ns, col_titer, col_runs} :
        std::vector<column>{col_core, col_id, col_olap, col_ns, col_iter, col_titer, col_reac, col_runs};

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
    auto iters = arg_iters.Get();
    std::vector<int> cpus = get_cpus();

#if USE_RDTSC
    fmt::print(out, "tsc_freq = {:.1f} MHz ({})\n", RdtscClock::tsc_freq() / 1000000.0, get_tsc_cal_info(arg_force_tsc_cal));
#endif
    fmt::print(out, "Running as root      : [{}]\n", is_root     ? "YES" : "NO ");
    fmt::print(out, "CPU pinning enabled  : [{}]\n", !arg_no_pin ? "YES" : "NO ");
    fmt::print(out, "available CPUs ({:4}): [{}]\n", cpus.size(), join(cpus, ", ").c_str());
    fmt::print(out, "get_nprocs_conf()    : [{}]\n", get_nprocs_conf());
    fmt::print(out, "get_nprocs()         : [{}]\n", get_nprocs());
    fmt::print(out, "iterations           : [{}]\n", iters);

    auto max_threads = arg_max_threads ? arg_max_threads.Get() : cpus.size();

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

    std::vector<result_holder> results_list;
    for (auto func : specs) {
        for (size_t count = arg_min_threads.Get(); count <= max_threads; count++) {

            const bool has_check = func.check_func;
            uint64_t counter_before = has_check ? func.check_func() : 0;

            // run
            hot_barrier start{count}, stop{count};
            std::deque<test_thread> threads;
            for (size_t tid = 0; tid < count; tid++) {
                int cpu = arg_no_pin ? -1 : cpus.at(tid % cpus.size());
                if (verbose) fmt::print("thread {} pinned to {}\n", tid, cpu);
                threads.emplace_back(tid, cpu, start, stop, func, iters);
            }

            for (auto& t : threads) {
                t.thread.join();
            }

            results_list.emplace_back(func, iters);
            uint64_t total_iters = 0;
            for (auto& t : threads) {
                results_list.back().results.push_back(t.res);
                total_iters += t.res.inner.total_iters;
            }

            auto counter_after = has_check ? func.check_func() : 0;
            auto total_counter_delta =  counter_after - counter_before;
            if (has_check && total_counter_delta != total_iters) {
                throw std::runtime_error(fmt::format("threads {}, algo {} failed check: {} actual vs {} expected",
                        count, func.id, total_counter_delta, total_iters));
            }
        }
    }

    report_results(results_list);

    return EXIT_SUCCESS;
}




