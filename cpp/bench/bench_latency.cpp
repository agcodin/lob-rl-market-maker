// Tick-to-trade latency harness: measures single-operation cost on the hot path
// and reports p50/p99/p99.9. No Google Benchmark dependency so it builds with a
// single clang++ invocation.
//
//   c++ -std=c++20 -O3 -DNDEBUG -Icpp/include cpp/bench/bench_latency.cpp -o bench
//   ./bench [iterations]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "lob/book.hpp"

#if defined(__x86_64__)
#include <x86intrin.h>
static inline std::uint64_t cycles() { return __rdtsc(); }
static constexpr const char* kCounter = "rdtsc cycles";
#elif defined(__aarch64__)
static inline std::uint64_t cycles() {
    std::uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static constexpr const char* kCounter = "cntvct_el0 ticks";
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
    auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))]; };
    double sum = 0;
    for (double x : v) sum += x;
    return {at(0.50), at(0.99), at(0.999), sum / v.size(), v.front(), v.back()};
}

void report(const char* name, std::vector<double> ns) {
    Stats s = percentiles(ns);
    std::printf("%-26s n=%-9zu p50=%7.1f  p99=%8.1f  p99.9=%9.1f  mean=%8.1f  max=%9.1f\n",
                name, ns.size(), s.p50, s.p99, s.p999, s.mean, s.max);
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t N = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 200000;
    auto book = std::make_unique<lob::OrderBook>();
    std::mt19937_64 rng(42);

    const lob::Tick mid = 10000;
    // Warm ladder so every measured operation touches a populated book.
    for (int d = 1; d <= 64; ++d) {
        for (int i = 0; i < 8; ++i) {
            book->limit(lob::kBid, mid - d, 50, lob::kNoise);
            book->limit(lob::kAsk, mid + d, 50, lob::kNoise);
        }
    }

    // Cycle-counter calibration against the steady clock.
    {
        auto c0 = cycles();
        auto t0 = clk::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count() < 50) {
        }
        auto c1 = cycles();
        auto t1 = clk::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        std::printf("counter: %s, %.3f ticks/ns\n\n", kCounter, (c1 - c0) / ns);
    }

    // The steady clock granularity is ~41.7 ns on Apple silicon, which is coarser
    // than a single book operation. Operations are therefore timed in blocks of
    // kBlock and the reported percentiles are per-operation costs within a block.
    constexpr std::size_t kBlock = 64;
    const std::size_t blocks = std::max<std::size_t>(N / kBlock, 1);

    std::vector<double> add_ns, cancel_ns, cross_ns, snap_ns;
    add_ns.reserve(blocks); cancel_ns.reserve(blocks); cross_ns.reserve(blocks);
    snap_ns.reserve(blocks);
    std::vector<lob::OrderId> live;
    live.reserve(blocks * kBlock);

    std::uniform_int_distribution<int> depth(1, 32);

    for (std::size_t b = 0; b < blocks; ++b) {
        lob::OrderId ids[kBlock];
        auto t0 = clk::now();
        for (std::size_t i = 0; i < kBlock; ++i) {
            lob::Side s = (i & 1) ? lob::kBid : lob::kAsk;
            lob::Tick px = s == lob::kBid ? mid - depth(rng) : mid + depth(rng);
            ids[i] = book->limit(s, px, 20, lob::kNoise);
        }
        auto t1 = clk::now();
        add_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / kBlock);
        for (auto id : ids) if (id > 0) live.push_back(id);
    }

    std::shuffle(live.begin(), live.end(), rng);
    for (std::size_t b = 0; b + kBlock <= live.size(); b += kBlock) {
        auto t0 = clk::now();
        for (std::size_t i = 0; i < kBlock; ++i) book->cancel(live[b + i]);
        auto t1 = clk::now();
        cancel_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / kBlock);
    }

    // Aggressive orders: refill a level, then cross it (measures match + pop).
    for (std::size_t b = 0; b < blocks; ++b) {
        for (std::size_t i = 0; i < kBlock; ++i) {
            lob::Side s = (i & 1) ? lob::kBid : lob::kAsk;
            book->limit(s, s == lob::kBid ? mid - 1 : mid + 1, 30, lob::kNoise);
        }
        auto t0 = clk::now();
        for (std::size_t i = 0; i < kBlock; ++i) {
            lob::Side taker = (i & 1) ? lob::kAsk : lob::kBid;
            book->market(taker, 30, lob::kNoise);
        }
        auto t1 = clk::now();
        cross_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / kBlock);
    }

    std::vector<lob::Tick> bp(10), ap(10);
    std::vector<std::int64_t> bv(10), av(10);
    std::int64_t sink = 0;  // consumed below so the snapshot cannot be optimized out
    for (std::size_t b = 0; b < blocks; ++b) {
        auto t0 = clk::now();
        for (std::size_t i = 0; i < kBlock; ++i) {
            book->snapshot(10, bp.data(), bv.data(), ap.data(), av.data());
            sink += bv[0] + av[0] + bp[0];
        }
        auto t1 = clk::now();
        snap_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / kBlock);
    }

    std::printf("latency, nanoseconds per operation (blocks of %zu)\n", kBlock);
    report("limit add (rest)", add_ns);
    report("cancel", cancel_ns);
    report("aggressive cross (1 lvl)", cross_ns);
    report("L2 snapshot (10 lvls)", snap_ns);
    std::printf("\nsnapshot checksum=%lld\n", static_cast<long long>(sink));
    std::printf("live orders=%d  cum volume=%lld  rejects=%lld\n",
                book->live_orders(), static_cast<long long>(book->cum_volume()),
                static_cast<long long>(book->rejects()));
    return 0;
}
