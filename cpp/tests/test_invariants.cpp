// Property and differential tests: replay random order streams and check, after
// every single operation, that the engine still satisfies its invariants and
// still agrees with a naive reference implementation.
//
// Run under sanitizers with:  cmake -B build -DLOBRL_SANITIZE=ON && ctest
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <memory>
#include <random>
#include <vector>

#include "lob/book.hpp"
#include "reference_book.hpp"

using namespace lob;

namespace {

// Random orders are confined to a narrow band so the price grid can be scanned
// exhaustively after every operation, and so the stream actually produces
// crossings instead of two sides that never meet.
constexpr Tick kLow = 9950;
constexpr Tick kHigh = 10050;

std::int64_t resting_volume_in_band(const OrderBook& b) {
    std::int64_t total = 0;
    for (Tick p = kLow; p <= kHigh; ++p) {
        total += b.volume_at(kBid, p);
        total += b.volume_at(kAsk, p);
    }
    return total;
}

// Every structural property the book claims, checked against itself.
void check_self_consistency(const OrderBook& b) {
    for (Tick p = kLow; p <= kHigh; ++p) {
        for (Side s : {kBid, kAsk}) {
            std::int64_t vol = b.volume_at(s, p);
            std::int32_t count = b.count_at(s, p);
            REQUIRE(vol >= 0);
            REQUIRE(count >= 0);
            // A level holds orders exactly when it holds volume; every resting
            // order has strictly positive size, so neither can be zero alone.
            REQUIRE((vol == 0) == (count == 0));
        }
    }

    const Tick bb = b.best_bid();
    const Tick ba = b.best_ask();
    if (bb >= 0 && ba >= 0) REQUIRE(bb < ba);            // never crossed

    // The touch must be the extreme populated level, with nothing beyond it.
    if (bb >= 0) {
        REQUIRE(b.count_at(kBid, bb) > 0);
        for (Tick p = bb + 1; p <= kHigh; ++p) REQUIRE(b.count_at(kBid, p) == 0);
    } else {
        for (Tick p = kLow; p <= kHigh; ++p) REQUIRE(b.count_at(kBid, p) == 0);
    }
    if (ba >= 0) {
        REQUIRE(b.count_at(kAsk, ba) > 0);
        for (Tick p = kLow; p < ba; ++p) REQUIRE(b.count_at(kAsk, p) == 0);
    } else {
        for (Tick p = kLow; p <= kHigh; ++p) REQUIRE(b.count_at(kAsk, p) == 0);
    }
}

}  // namespace

TEST_CASE("volume is conserved across every operation", "[fuzz][invariant]") {
    auto book = std::make_unique<OrderBook>();
    std::mt19937_64 rng(20240101);
    std::uniform_int_distribution<int> price(kLow, kHigh);
    std::uniform_int_distribution<int> qty(1, 40);
    std::uniform_int_distribution<int> pick(0, 99);

    std::vector<OrderId> live;

    for (int step = 0; step < 40000; ++step) {
        const std::int64_t rest_before = resting_volume_in_band(*book);
        const std::int64_t cum_before = book->cum_volume();
        const int roll = pick(rng);
        Side side = (pick(rng) < 50) ? kBid : kAsk;

        if (roll < 65) {
            // A limit order either rests its whole size or trades some of it
            // away first, so resting volume moves by qty less twice whatever
            // traded: once for the liquidity it removed, once for what it
            // therefore did not rest.
            const Qty q = qty(rng);
            OrderId id = book->limit(side, price(rng), q, kNoise);
            const std::int64_t traded = book->cum_volume() - cum_before;
            REQUIRE(resting_volume_in_band(*book) - rest_before == q - 2 * traded);
            if (id > 0) live.push_back(id);
        } else if (roll < 85) {
            const Qty filled = book->market(side, qty(rng), kNoise);
            const std::int64_t traded = book->cum_volume() - cum_before;
            REQUIRE(filled == traded);
            REQUIRE(resting_volume_in_band(*book) - rest_before == -traded);
        } else if (!live.empty()) {
            std::uniform_int_distribution<std::size_t> idx(0, live.size() - 1);
            const std::size_t k = idx(rng);
            const OrderId id = live[k];
            const Qty before = book->qty_of(id);        // 0 if already gone
            const bool cancelled = book->cancel(id);
            REQUIRE(cancelled == (before > 0));
            REQUIRE(resting_volume_in_band(*book) - rest_before == -before);
            REQUIRE(book->cum_volume() == cum_before);
            live[k] = live.back();
            live.pop_back();
        }

        if (step % 200 == 0) check_self_consistency(*book);
    }
    check_self_consistency(*book);
}

TEST_CASE("the engine agrees with a naive reference book on every step",
          "[fuzz][differential]") {
    auto fast = std::make_unique<OrderBook>();
    ref::ReferenceBook slow;

    std::mt19937_64 rng(777);
    std::uniform_int_distribution<int> price(kLow, kHigh);
    std::uniform_int_distribution<int> qty(1, 30);
    std::uniform_int_distribution<int> pick(0, 99);
    std::uniform_int_distribution<int> owner(0, 3);

    // The two books assign ids independently, so the test keeps them paired.
    std::vector<std::pair<OrderId, OrderId>> live;   // (fast id, reference id)

    for (int step = 0; step < 30000; ++step) {
        const int roll = pick(rng);
        const Side side = (pick(rng) < 50) ? kBid : kAsk;
        const OwnerId who = static_cast<OwnerId>(owner(rng));

        if (roll < 60) {
            const Tick p = price(rng);
            const Qty q = qty(rng);
            OrderId a = fast->limit(side, p, q, who);
            OrderId b = slow.limit(side, p, q, who);
            // Both must agree on whether it rested, filled out, or was rejected.
            REQUIRE((a > 0) == (b > 0));
            REQUIRE((a == 0) == (b == 0));
            REQUIRE((a < 0) == (b < 0));
            if (a > 0) live.emplace_back(a, b);
        } else if (roll < 82) {
            const Qty q = qty(rng);
            REQUIRE(fast->market(side, q, who) == slow.market(side, q, who));
        } else if (!live.empty()) {
            std::uniform_int_distribution<std::size_t> idx(0, live.size() - 1);
            const std::size_t k = idx(rng);
            const auto [a, b] = live[k];
            REQUIRE(fast->cancel(a) == slow.cancel(b));
            live[k] = live.back();
            live.pop_back();
        }

        REQUIRE(fast->best_bid() == slow.best_bid());
        REQUIRE(fast->best_ask() == slow.best_ask());
        REQUIRE(fast->cum_volume() == slow.cum_volume());
        REQUIRE(static_cast<std::size_t>(fast->live_orders()) == slow.live_orders());

        if (step % 50 == 0) {
            for (Tick p = kLow; p <= kHigh; ++p) {
                for (Side s : {kBid, kAsk}) {
                    REQUIRE(fast->volume_at(s, p) == slow.volume_at(s, p));
                    REQUIRE(fast->count_at(s, p) == slow.count_at(s, p));
                }
            }
        }
    }
}

TEST_CASE("time priority within a level survives a random stream",
          "[fuzz][priority]") {
    auto fast = std::make_unique<OrderBook>();
    ref::ReferenceBook slow;
    std::mt19937_64 rng(31337);
    std::uniform_int_distribution<int> price(kLow, kHigh);
    std::uniform_int_distribution<int> qty(1, 25);
    std::uniform_int_distribution<int> pick(0, 99);

    std::vector<std::pair<OrderId, OrderId>> live;

    for (int step = 0; step < 15000; ++step) {
        const int roll = pick(rng);
        const Side side = (pick(rng) < 50) ? kBid : kAsk;
        if (roll < 70) {
            const Tick p = price(rng);
            const Qty q = qty(rng);
            OrderId a = fast->limit(side, p, q, kNoise);
            OrderId b = slow.limit(side, p, q, kNoise);
            if (a > 0 && b > 0) live.emplace_back(a, b);
        } else if (roll < 88) {
            // One draw, fed to both books: drawing twice sends them different
            // order sizes and the comparison becomes meaningless.
            const Qty q = qty(rng);
            fast->market(side, q, kNoise);
            slow.market(side, q, kNoise);
        } else if (!live.empty()) {
            std::uniform_int_distribution<std::size_t> idx(0, live.size() - 1);
            const std::size_t k = idx(rng);
            fast->cancel(live[k].first);
            slow.cancel(live[k].second);
            live[k] = live.back();
            live.pop_back();
        }

        // Queue position is the observable form of time priority: the shares
        // resting ahead of an order must match the naive FIFO deque exactly.
        if (step % 25 == 0) {
            for (const auto& [a, b] : live) {
                if (!fast->is_live(a)) continue;
                REQUIRE(slow.is_live(b));
                REQUIRE(fast->queue_ahead(a) == slow.queue_ahead(b));
            }
        }
    }
}

TEST_CASE("a stream of crossing orders never leaves the book crossed",
          "[fuzz][cross]") {
    auto book = std::make_unique<OrderBook>();
    std::mt19937_64 rng(2468);
    // A deliberately narrow band so nearly every order is marketable.
    std::uniform_int_distribution<int> price(9999, 10001);
    std::uniform_int_distribution<int> qty(1, 50);
    std::uniform_int_distribution<int> pick(0, 1);

    for (int step = 0; step < 20000; ++step) {
        book->limit(pick(rng) ? kBid : kAsk, price(rng), qty(rng), kNoise);
        const Tick bb = book->best_bid();
        const Tick ba = book->best_ask();
        if (bb >= 0 && ba >= 0) REQUIRE(bb < ba);
    }
    check_self_consistency(*book);
}

TEST_CASE("reset leaves no state behind for the next run", "[fuzz][reset]") {
    auto book = std::make_unique<OrderBook>();
    std::mt19937_64 rng(13);
    std::uniform_int_distribution<int> price(kLow, kHigh);
    std::uniform_int_distribution<int> qty(1, 20);

    for (int round = 0; round < 5; ++round) {
        for (int i = 0; i < 2000; ++i) {
            book->limit(i % 2 ? kBid : kAsk, price(rng), qty(rng), kNoise);
            if (i % 7 == 0) book->market(i % 2 ? kAsk : kBid, qty(rng), kNoise);
        }
        book->reset();
        REQUIRE(book->best_bid() == -1);
        REQUIRE(book->best_ask() == -1);
        REQUIRE(book->live_orders() == 0);
        REQUIRE(book->cum_volume() == 0);
        REQUIRE(resting_volume_in_band(*book) == 0);
        check_self_consistency(*book);
    }
}
