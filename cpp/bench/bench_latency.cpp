// Tick-to-trade latency harness.
//
//   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
//   ./build/bench_latency [iterations] [--quick] [--csv]
//
// Two things this harness tries to be honest about:
//
//   1. Clock resolution. Per-operation timing is only meaningful if the clock
//      can resolve a single operation. On x86 rdtsc ticks at the TSC frequency
//      and can; on Apple silicon cntvct_el0 runs at 24 MHz, so one tick is
//      ~41.7 ns -- far coarser than the operation being measured. The harness
//      measures its own resolution and says which mode it used. In block mode
//      the percentiles describe the per-operation cost *within* a block, which
//      understates the tail, because one slow operation is averaged over the
//      whole block.
//
//   2. A baseline. "4 ns" means little alone, so every operation is also run
//      against a naive std::map-of-deques book, the obvious implementation.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "lob/book.hpp"
#include "reference_book.hpp"

#if defined(__x86_64__)
#include <x86intrin.h>
static inline std::uint64_t cycles() { return __rdtsc(); }
static constexpr const char* kCounter = "rdtsc";
#elif defined(__aarch64__)
static inline std::uint64_t cycles() {
    std::uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static constexpr const char* kCounter = "cntvct_el0";
#else
static inline std::uint64_t cycles() { return 0; }
static constexpr const char* kCounter = "unavailable";
#endif

using clk = std::chrono::steady_clock;

namespace {

struct Stats {
    double p50, p99, p999, mean, min, max;
};

Stats percentiles(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    auto at = [&](double q) {
        return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))];
    };
    double sum = 0;
    for (double x : v) sum += x;
    return {at(0.50), at(0.99), at(0.999), sum / v.size(), v.front(), v.back()};
}

bool g_csv = false;

void report(const char* name, const char* impl, std::vector<double> ns) {
    Stats s = percentiles(ns);
    if (g_csv) {
        std::printf("%s,%s,%zu,%.3f,%.3f,%.3f,%.3f\n", name, impl, ns.size(),
                    s.p50, s.p99, s.p999, s.mean);
    } else {
        std::printf("  %-26s %-10s n=%-8zu p50=%8.2f p99=%9.2f p99.9=%10.2f mean=%9.2f\n",
                    name, impl, ns.size(), s.p50, s.p99, s.p999, s.mean);
    }
}

// Nanoseconds per counter tick, measured rather than assumed. Note the
// NOMINAL frequency is not what matters: on Apple silicon cntfrq_el0 reports
// 1 GHz while the counter physically advances in much larger steps, so the
// figure that decides the timing mode is the measured resolution below.
double tick_ns() {
    auto c0 = cycles();
    auto t0 = clk::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count() < 50) {
    }
    auto c1 = cycles();
    auto t1 = clk::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::uint64_t ticks = c1 - c0;
    return ticks ? ns / static_cast<double>(ticks) : 0.0;
}

// The smallest nonzero gap between two consecutive reads: what the timer can
// actually distinguish, as opposed to what its frequency claims.
double timer_resolution_ns(double ns_per_tick) {
    std::uint64_t best = UINT64_MAX;
    for (int i = 0; i < 200000; ++i) {
        std::uint64_t a = cycles();
        std::uint64_t b = cycles();
        if (b > a) best = std::min(best, b - a);
    }
    return best == UINT64_MAX ? 1e9 : static_cast<double>(best) * ns_per_tick;
}

// Times a block of operations with the cycle counter rather than steady_clock,
// which is both cheaper and finer on the platforms where per-op timing works.
struct Timer {
    double ns_per_tick;
    std::uint64_t t0 = 0;
    void start() { t0 = cycles(); }
    double per_op(std::size_t n) {
        return static_cast<double>(cycles() - t0) * ns_per_tick / static_cast<double>(n);
    }
};

std::string cpu_name() {
#if defined(__APPLE__)
    FILE* f = popen("sysctl -n machdep.cpu.brand_string 2>/dev/null", "r");
#else
    FILE* f = popen("grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2-", "r");
#endif
    if (!f) return "unknown";
    char buf[256] = {0};
    if (!std::fgets(buf, sizeof(buf), f)) buf[0] = '\0';
    pclose(f);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
    return s.empty() ? "unknown" : s;
}

std::string compiler_name() {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    return "unknown";
#endif
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t N = 200000;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        else if (std::strcmp(argv[i], "--csv") == 0) g_csv = true;
        else N = std::strtoul(argv[i], nullptr, 10);
    }
    if (quick) N = 20000;

    const double ns_per_tick = tick_ns();
    const double resolution = timer_resolution_ns(ns_per_tick);
    // Per-operation timing needs a clock that can resolve one operation, which
    // takes single-digit nanoseconds here. Anything coarser is timed in blocks.
    const bool per_op = resolution > 0.0 && resolution <= 2.0;
    constexpr std::size_t kBlock = 64;
    const std::size_t block = per_op ? 1 : kBlock;
    Timer timer{ns_per_tick};

    if (!g_csv) {
        std::printf("machine    : %s (%s)\n", cpu_name().c_str(),
#if defined(__aarch64__)
                    "arm64"
#elif defined(__x86_64__)
                    "x86_64"
#else
                    "unknown arch"
#endif
        );
        std::printf("compiler   : %s\n", compiler_name().c_str());
        std::printf("build flags: %s\n", LOBRL_BUILD_FLAGS);
        std::printf("counter    : %s, %.3f ns/tick nominal, %.1f ns measured resolution\n",
                    kCounter, ns_per_tick, resolution);
        std::printf("timing     : %s\n",
                    per_op ? "per operation (clock resolves a single operation)"
                           : "blocks of 64 operations -- the clock is coarser than one\n"
                             "             operation, so these percentiles describe the\n"
                             "             per-operation cost WITHIN a block and understate\n"
                             "             the true tail");
        std::printf("iterations : %zu\n\n", N);
        std::printf("nanoseconds per operation\n");
    } else {
        std::printf("operation,impl,samples,p50,p99,p999,mean\n");
    }

    auto fast = std::make_unique<lob::OrderBook>();
    ref::ReferenceBook slow;
    std::mt19937_64 rng(42);
    const lob::Tick mid = 10000;

    // Warm both books with the same ladder so neither is measured on an empty
    // book, and both do the same amount of real work.
    for (int d = 1; d <= 64; ++d) {
        for (int i = 0; i < 8; ++i) {
            fast->limit(lob::kBid, mid - d, 50, lob::kNoise);
            fast->limit(lob::kAsk, mid + d, 50, lob::kNoise);
            slow.limit(lob::kBid, mid - d, 50, lob::kNoise);
            slow.limit(lob::kAsk, mid + d, 50, lob::kNoise);
        }
    }

    const std::size_t blocks = std::max<std::size_t>(N / block, 1);
    std::uniform_int_distribution<int> depth(1, 32);

    // ---- limit add ----------------------------------------------------
    std::vector<double> add_fast, add_slow;
    std::vector<lob::OrderId> live_fast;
    std::vector<lob::OrderId> live_slow;
    add_fast.reserve(blocks);
    add_slow.reserve(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        std::vector<lob::Tick> px(block);
        std::vector<lob::Side> sd(block);
        for (std::size_t i = 0; i < block; ++i) {
            sd[i] = (i & 1) ? lob::kBid : lob::kAsk;
            px[i] = sd[i] == lob::kBid ? mid - depth(rng) : mid + depth(rng);
        }
        timer.start();
        for (std::size_t i = 0; i < block; ++i) live_fast.push_back(fast->limit(sd[i], px[i], 20, lob::kNoise));
        add_fast.push_back(timer.per_op(block));

        timer.start();
        for (std::size_t i = 0; i < block; ++i) live_slow.push_back(slow.limit(sd[i], px[i], 20, lob::kNoise));
        add_slow.push_back(timer.per_op(block));
    }
    report("limit add (rest)", "engine", add_fast);
    report("limit add (rest)", "std::map", add_slow);

    // ---- cancel -------------------------------------------------------
    std::shuffle(live_fast.begin(), live_fast.end(), std::mt19937_64(7));
    std::shuffle(live_slow.begin(), live_slow.end(), std::mt19937_64(7));
    std::vector<double> cancel_fast, cancel_slow;
    for (std::size_t b = 0; b + block <= live_fast.size(); b += block) {
        timer.start();
        for (std::size_t i = 0; i < block; ++i) fast->cancel(live_fast[b + i]);
        cancel_fast.push_back(timer.per_op(block));
    }
    for (std::size_t b = 0; b + block <= live_slow.size(); b += block) {
        timer.start();
        for (std::size_t i = 0; i < block; ++i) slow.cancel(live_slow[b + i]);
        cancel_slow.push_back(timer.per_op(block));
    }
    report("cancel", "engine", cancel_fast);
    report("cancel", "std::map", cancel_slow);

    // ---- aggressive cross (match + pop the level) ----------------------
    std::vector<double> cross_fast, cross_slow;
    for (std::size_t b = 0; b < blocks; ++b) {
        for (std::size_t i = 0; i < block; ++i) {
            lob::Side s = (i & 1) ? lob::kBid : lob::kAsk;
            fast->limit(s, s == lob::kBid ? mid - 1 : mid + 1, 30, lob::kNoise);
        }
        timer.start();
        for (std::size_t i = 0; i < block; ++i) {
            fast->market((i & 1) ? lob::kAsk : lob::kBid, 30, lob::kNoise);
        }
        cross_fast.push_back(timer.per_op(block));

        for (std::size_t i = 0; i < block; ++i) {
            lob::Side s = (i & 1) ? lob::kBid : lob::kAsk;
            slow.limit(s, s == lob::kBid ? mid - 1 : mid + 1, 30, lob::kNoise);
        }
        timer.start();
        for (std::size_t i = 0; i < block; ++i) {
            slow.market((i & 1) ? lob::kAsk : lob::kBid, 30, lob::kNoise);
        }
        cross_slow.push_back(timer.per_op(block));
    }
    report("aggressive cross (1 lvl)", "engine", cross_fast);
    report("aggressive cross (1 lvl)", "std::map", cross_slow);

    // ---- level-2 snapshot ---------------------------------------------
    std::vector<lob::Tick> bp(10), ap(10);
    std::vector<std::int64_t> bv(10), av(10);
    std::int64_t sink = 0;          // consumed below so this cannot be elided
    std::vector<double> snap_fast, snap_slow;
    for (std::size_t b = 0; b < blocks; ++b) {
        timer.start();
        for (std::size_t i = 0; i < block; ++i) {
            fast->snapshot(10, bp.data(), bv.data(), ap.data(), av.data());
            sink += bv[0] + av[0] + bp[0];
        }
        snap_fast.push_back(timer.per_op(block));

        timer.start();
        for (std::size_t i = 0; i < block; ++i) {
            lob::Tick p = slow.best_bid();
            for (int k = 0; k < 10 && p >= 0; ++k, --p) sink += slow.volume_at(lob::kBid, p);
            p = slow.best_ask();
            for (int k = 0; k < 10 && p >= 0; ++k, ++p) sink += slow.volume_at(lob::kAsk, p);
        }
        snap_slow.push_back(timer.per_op(block));
    }
    report("L2 snapshot (10 lvls)", "engine", snap_fast);
    report("L2 snapshot (10 lvls)", "std::map", snap_slow);

    if (!g_csv) {
        auto ratio = [](std::vector<double> slow_v, std::vector<double> fast_v) {
            double a = percentiles(slow_v).p50, b = percentiles(fast_v).p50;
            return b > 0.0 ? a / b : 0.0;   // 0 means the timer could not resolve it
        };
        std::printf("\nspeedup over std::map (median): add %.1fx  cancel %.1fx  cross %.1fx  snapshot %.1fx\n",
                    ratio(add_slow, add_fast), ratio(cancel_slow, cancel_fast),
                    ratio(cross_slow, cross_fast), ratio(snap_slow, snap_fast));
        std::printf("checksum=%lld  live=%d  traded=%lld  rejects=%lld\n",
                    static_cast<long long>(sink), fast->live_orders(),
                    static_cast<long long>(fast->cum_volume()),
                    static_cast<long long>(fast->rejects()));
    }
    return 0;
}
