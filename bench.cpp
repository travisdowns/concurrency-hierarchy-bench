/*
 * bench.cpp
 */

#include "args.hxx"
#include "fairlocks.hpp"
#include "stats.hpp"
#include "table.hpp"
#include "tsc-support.hpp"
#include "util.hpp"
#include "hedley.h"
#include "fmt/format.h"

#include <array>
#include <atomic>
#include <deque>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <cinttypes>
#include <exception>
#include <limits>
#include <mutex>
#include <set>
#include <functional>
#include <thread>
#include <vector>

#include <error.h>
#include <err.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>


#define MSR_IA32_MPERF 0x000000e7
#define MSR_IA32_APERF 0x000000e8

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

static volatile thread_local uint64_t tls_counter;
static std::atomic<uint64_t> tls_accumulator;

uint64_t tls_add(size_t iters) {
    uint64_t rac_count = 0;
    IF_RAC(uint64_t last = tls_counter - 1;)
    while (iters--) {
        auto cur = tls_counter++;
        IF_RAC(rac_count += (cur == last + 1); last = cur;)
    }
    return rac_count;
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


std::atomic<uint64_t> atomic_counter{};

uint64_t atomic_add(size_t iters) {
    uint64_t rac_count = 0;
    IF_RAC(uint64_t last = atomic_counter.load() - 1;)
    while (iters--) {
        auto cur = atomic_counter++;
        IF_RAC(rac_count += (cur == last + 1); last = cur;)
    }
    return rac_count;
}

std::atomic<uint64_t> cas_counter;

uint64_t cas_add(size_t iters) {
    uint64_t rac_count = 0;
    IF_RAC(uint64_t last = cas_counter.load() - 1;)
    while (iters--) {
        auto cur = cas_counter.load();
        while (!cas_counter.compare_exchange_weak(cur, cur + 1))
            ;
        IF_RAC(rac_count += (cur == last + 1); last = cur;)
    }
    return rac_count;
}

struct multi_holder {
    alignas(64) std::atomic<uint64_t> counter;
};

constexpr size_t MULTI_MAX  = 64;
constexpr size_t MULTI_MASK = MULTI_MAX - 1;
static_assert((MULTI_MAX & MULTI_MASK) == 0, "MULTI_MAX must be a power of 2");

static thread_local size_t multi_idx;
static multi_holder multi_array[MULTI_MAX];

uint64_t cas_multi(size_t iters) {
    uint64_t rac_count = 0;
    // size_t contention = 0;
    auto& counter = multi_array[multi_idx].counter;
    IF_RAC(uint64_t last = counter.load() - 1;)
    while (iters--) {
        auto cur = counter.load();

        if (counter.compare_exchange_weak(cur, cur + 1)) {
            continue;
        }

        multi_idx = (multi_idx + 1) & MULTI_MASK;

        while (!counter.compare_exchange_weak(cur, cur + 1))
            ;
        IF_RAC(rac_count += (cur == last + 1); last = cur;)
    }

    return rac_count;
}

uint64_t cas_multi_read() {
    uint64_t sum = 0;
    for (auto& h : multi_array) {
        sum += h.counter.load(std::memory_order_relaxed);
    }
    return sum;
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


void per_thread_init() {
    
}

void per_thread_deinit() {
    tls_accumulator += tls_counter;
}


uint64_t read_tls_counters() {
    return tls_accumulator.load();
}

/** make a test_func object from adaptor<T> using the given lock type T */
template <typename T>
test_func make(const char *name, const char* desc = "desc") {
    return { adaptor<T>::bench, name, desc, adaptor<T>::get_counter };
}

std::vector<test_func> ALL_FUNCS = {
    {plain_add                                      , "plain add"  , "desc" , nullptr }                             ,
    {tls_add                                        , "tls add"    , "desc" , read_tls_counters }                   ,
    {atomic_add                                     , "atomic add" , "desc" , []{ return atomic_counter.load(); } } ,
    {cas_add                                        , "cas add"    , "desc" , []{ return cas_counter.load(); } }    ,
    {cas_multi                                      , "cas multi"  , "desc" , cas_multi_read }    ,
    make<std::mutex>("mutex add")                   ,
    make<locks::spinlock_hot>("pure spin")          ,
    make<locks::spinlock_pause>("pause spin")       ,
    make<locks::spinlock_yield>("yield spin")       ,
    make<locks::ticket_spin>("ticket spin")         ,
    make<locks::yielding_spin>("ticket yield")      ,
    make<locks::blocking_ticket>("ticket blocking") ,
    make<locks::fifo_queued>("queued fifo")         ,
    make<locks::mutex3>("mutex3")                   ,
};

void pin_to_cpu(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1) {
        error(EXIT_FAILURE, errno, "could not pin to CPU %d", cpu);
    }
}

/** args */
static args::ArgumentParser parser{"conc-bench: Demonstrate concurrency perforamnce levels"};
static args::HelpFlag help{parser, "help", "Display this help menu", {'h', "help"}};
static args::Flag arg_force_tsc_cal{parser, "force-tsc-calibrate",
    "Force manual TSC calibration loop, even if cpuid TSC Hz is available", {"force-tsc-calibrate"}};
static args::Flag arg_no_pin{parser, "no-pin",
    "Don't try to pin threads to CPU - gives worse results but works around affinity issues on TravisCI", {"no-pin"}};
static args::Flag arg_verbose{parser, "verbose", "Output more info", {"verbose"}};
static args::Flag arg_nobarrier{parser, "no-barrier", "Don't sync up threads before each test (debugging only)", {"no-barrier"}};
static args::Flag arg_list{parser, "list", "List the available tests and their descriptions", {"list"}};
static args::Flag arg_csv{parser, "", "Output a csv table instead of the default", {"csv"}};

static args::Flag arg_hyperthreads{parser, "allow-hyperthreads", "By default we try to filter down the available cpus to include only physical cores, but "
    "with this option we'll use all logical cores meaning you'll run two tests on cores with hyperthreading", {"allow-hyperthreads"}};
static args::ValueFlag<std::string> arg_algos{parser, "TEST-ID", "Run only the algorithms in the comma separated list", {"algos"}};
static args::ValueFlag<size_t> arg_iters{parser, "ITERS", "Run the test loop ITERS times (default 100000)", {"iters"}, 100000};
static args::ValueFlag<int> arg_min_threads{parser, "MIN", "The minimum number of threads to use", {"min-threads"}, 1};
static args::ValueFlag<int> arg_max_threads{parser, "MAX", "The maximum number of threads to use", {"max-threads"}};
static args::ValueFlag<uint64_t> arg_warm_ms{parser, "MILLISECONDS", "Warmup milliseconds for each thread after pinning (default 100)", {"warmup-ms"}, 100};


bool verbose;

template <typename CHRONO_CLOCK>
struct StdClock {
    using now_t   = decltype(CHRONO_CLOCK::now());
    using delta_t = typename CHRONO_CLOCK::duration;

    static now_t now() {
        return CHRONO_CLOCK::now();
    }

    /* accept the result of subtraction of durations and convert to nanos */
    static uint64_t to_nanos(typename CHRONO_CLOCK::duration d) {
        return duration_cast<std::chrono::nanoseconds>(d).count();
    }
};

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
    static uint64_t to_nanos(now_t diff) {
        static double tsc_to_nanos = 1000000000.0 / tsc_freq();
        return diff * tsc_to_nanos;
    }

    static uint64_t tsc_freq() {
        static uint64_t freq = get_tsc_freq(arg_force_tsc_cal);
        return freq;
    }

};

/**
 * We pass an outer_clock to run_test which times outside the iteration of the innermost loop (i.e.,
 * it times around the loop that runs TRIES times), start should reset the state unless you want to
 * time warmup iterations.
 */
struct outer_timer {
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual ~outer_timer() {}
};

struct dummy_outer : outer_timer {
    static dummy_outer dummy;
    virtual void start() override {};
    virtual void stop() override {};
};
dummy_outer dummy_outer::dummy{};

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
inner_result run_test(cal_f* func, size_t iters, outer_timer& outer, hot_barrier *barrier) {

    std::array<typename CLOCK::delta_t, TRIES> results;

    inner_result result;

    result.ostart_ts = RdtscClock::now();
    for (size_t w = 0; w < WARMUP + 1; w++) {
        uint64_t reacquires = 0;
        result.istart_ts = RdtscClock::now();
        outer.start();
        for (size_t r = 0; r < TRIES; r++) {
            auto t0 = CLOCK::now();
            reacquires += func(iters);
            auto t1 = CLOCK::now();
            results[r] = t1 - t0;
        }
        outer.stop();
        result.iend_ts = RdtscClock::now();
        result.reacquires = reacquires;
    }

    result.oend_ts1 = RdtscClock::now();

    // this loop keeps running the test function until all other
    // threads are done too, to keep the environment consistent
    // printf("%d at barrier\n", gettid());
    for (barrier->increment(); !barrier->is_broken();) {
        func(iters);
        result.total_iters += iters;
    }
    result.oend_ts2 = RdtscClock::now();

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
    warmup(uint64_t millis) : millis{millis} {}

    long warm() {
        int64_t start = (int64_t)RdtscClock::now();
        long iters = 0;
        while (RdtscClock::to_nanos(RdtscClock::now() - start) < 1000000u * millis) {
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
        // if (verbose) printf("Constructed test in thread %lu, this = %p\n", id, this);
    }

    test_thread(const test_thread&) = delete;
    test_thread(test_thread&&) = delete;
    void operator=(const test_thread&) = delete;

    void operator()() {
        per_thread_init();

        // if (verbose) printf("Running test in thread %lu, this = %p\n", id, this);
        if (cpu >= 0) {
            pin_to_cpu(cpu);
        }
        outer_timer& outer = dummy_outer::dummy;
        warmup w{arg_warm_ms.Get()};
        long warms = w.warm();
        if (verbose) printf("[%2lu] Warmup iters %lu\n", id, warms);
        if (!arg_nobarrier) {
            long count = start_barrier->wait();
            if (verbose) printf("[%2lu] Thread loop count: %ld\n", id, count);
        }
        res.start_ts = RdtscClock::now();
        res.inner = run_test<RdtscClock>(test.func, iters, outer, stop_barrier);
        res.end_ts = RdtscClock::now();

        per_thread_deinit();
    }
};


// TODO remove
void phelper(const char *msg, uint64_t start, uint64_t end) {
    printf("%s: %zu\n", msg, RdtscClock::to_nanos(end - start));
}

template <typename E>
double aggregate_results(const std::vector<result>& results, E e) {
    double a = 0;
    for (const auto& result : results) {
        // printf("it tim: %zu\n", result.inner.timed_iters);
        // printf("it tot: %zu\n", result.inner.total_iters);
        // phelper("t1", result.inner.ostart_ts, result.inner.oend_ts1);
        // phelper("t2", result.inner.ostart_ts, result.inner.oend_ts2);
        // printf("\n");
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
static column col_id  {"ID", LEFT, [](Row& r, const result_holder& h){ r.add(h.spec.id); }};
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

    printf("%s", (arg_csv ? table.csv_str() : table.str()).c_str());
}

void list_tests() {
    table::Table table;
    table.newRow().add("ID").add("Description");
    for (auto& t : ALL_FUNCS) {
        table.newRow().add(t.id).add(t.description);
    }
    printf("Available tests:\n\n%s\n", table.str().c_str());
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

    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help& help) {
        printf("%s\n", parser.Help().c_str());
        exit(EXIT_SUCCESS);
    } catch (const args::ParseError& e) {
        printf("ERROR while parsing arguments: %s\n", e.what());
        printf("\nUsage:\n%s\n", parser.Help().c_str());
        exit(EXIT_FAILURE);
    }

    // if csv mode is on, only the table should go to stdout
    // the rest goes to stderr
    FILE* out = arg_csv ? stderr : stdout;
    set_logging_file(out);

    if (arg_list) {
        list_tests();
        exit(EXIT_SUCCESS);
    }

    verbose = arg_verbose;
    bool is_root = (geteuid() == 0);
    auto iters = arg_iters.Get();
    std::vector<int> cpus = get_cpus();
    
    fprintf(out, "tsc_freq = %.1f MHz (%s)\n", RdtscClock::tsc_freq() / 1000000.0, get_tsc_cal_info(arg_force_tsc_cal));
    fprintf(out, "Running as root      : [%s]\n", is_root     ? "YES" : "NO ");
    fprintf(out, "CPU pinning enabled  : [%s]\n", !arg_no_pin ? "YES" : "NO ");
    fprintf(out, "available CPUs (%4lu): [%s]\n", cpus.size(), join(cpus, ", ").c_str());
    fprintf(out, "get_nprocs_conf()    : [%d]\n", get_nprocs_conf());
    fprintf(out, "get_nprocs()         : [%d]\n", get_nprocs());
    fprintf(out, "iterations           : [%zu]\n", iters);

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




