// Unit tests for the matching engine, exercised directly in C++ rather than
// through the Python bindings. The Python suite covers the same ground from the
// other side of pybind11; these catch anything the binding layer would mask.
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

#include "lob/book.hpp"

using namespace lob;

namespace {

// The arena is several megabytes of members, so never put a book on the stack.
std::unique_ptr<OrderBook> make_book() { return std::make_unique<OrderBook>(); }

// Sums resting volume across the whole grid. Only used in tests.
std::int64_t resting_volume(const OrderBook& b) {
    std::int64_t total = 0;
    for (std::size_t p = 0; p < kMaxTicks; ++p) {
        total += b.volume_at(kBid, static_cast<Tick>(p));
        total += b.volume_at(kAsk, static_cast<Tick>(p));
    }
    return total;
}

}  // namespace

// ---------------------------------------------------------------- adding ---

TEST_CASE("an empty book reports no touch", "[add]") {
    auto b = make_book();
    CHECK(b->best_bid() == -1);
    CHECK(b->best_ask() == -1);
    CHECK(b->spread() == -1);
    CHECK(b->live_orders() == 0);
    CHECK(b->cum_volume() == 0);
}

TEST_CASE("a resting order sets the touch and its level volume", "[add]") {
    auto b = make_book();
    OrderId id = b->limit(kBid, 10000, 50, kNoise);
    REQUIRE(id > 0);
    CHECK(b->best_bid() == 10000);
    CHECK(b->volume_at(kBid, 10000) == 50);
    CHECK(b->count_at(kBid, 10000) == 1);
    CHECK(b->qty_of(id) == 50);
    CHECK(b->is_live(id));
    CHECK(b->live_orders() == 1);
}

TEST_CASE("the touch tracks the best of several levels", "[add]") {
    auto b = make_book();
    b->limit(kBid, 9998, 10, kNoise);
    b->limit(kBid, 10000, 10, kNoise);
    b->limit(kBid, 9999, 10, kNoise);
    b->limit(kAsk, 10005, 10, kNoise);
    b->limit(kAsk, 10003, 10, kNoise);
    CHECK(b->best_bid() == 10000);
    CHECK(b->best_ask() == 10003);
    CHECK(b->spread() == 3);
    CHECK(b->mid() == 10001.5);
}

TEST_CASE("orders at the price grid boundaries are accepted", "[add][edge]") {
    auto b = make_book();
    CHECK(b->limit(kBid, 0, 5, kNoise) > 0);
    CHECK(b->limit(kAsk, static_cast<Tick>(kMaxTicks) - 1, 5, kNoise) > 0);
    CHECK(b->best_bid() == 0);
    CHECK(b->best_ask() == static_cast<Tick>(kMaxTicks) - 1);
}

TEST_CASE("orders outside the grid or with no size are rejected", "[add][edge]") {
    auto b = make_book();
    CHECK(b->limit(kBid, -1, 10, kNoise) == -1);
    CHECK(b->limit(kBid, static_cast<Tick>(kMaxTicks), 10, kNoise) == -1);
    CHECK(b->limit(kBid, 10000, 0, kNoise) == -1);
    CHECK(b->limit(kBid, 10000, -5, kNoise) == -1);
    CHECK(b->rejects() == 4);
    CHECK(b->live_orders() == 0);
}

// ------------------------------------------------------------- priority ----

TEST_CASE("the earlier order at a price fills first", "[priority]") {
    auto b = make_book();
    OrderId first = b->limit(kBid, 10000, 10, kNoise);
    OrderId second = b->limit(kBid, 10000, 10, kNoise);
    CHECK(b->queue_ahead(first) == 0);
    CHECK(b->queue_ahead(second) == 10);

    b->market(kAsk, 10, kNoise);
    CHECK_FALSE(b->is_live(first));
    CHECK(b->is_live(second));
    CHECK(b->qty_of(second) == 10);
}

TEST_CASE("queue position accounts for every order ahead", "[priority]") {
    auto b = make_book();
    std::vector<OrderId> ids;
    for (int i = 0; i < 5; ++i) ids.push_back(b->limit(kAsk, 10010, 7, kNoise));
    for (int i = 0; i < 5; ++i) CHECK(b->queue_ahead(ids[i]) == 7 * i);

    // Cancelling from the middle moves everyone behind it forward.
    REQUIRE(b->cancel(ids[1]));
    CHECK(b->queue_ahead(ids[2]) == 7);
    CHECK(b->queue_ahead(ids[4]) == 21);
}

TEST_CASE("a better price jumps the whole queue", "[priority]") {
    auto b = make_book();
    OrderId behind = b->limit(kBid, 10000, 10, kNoise);
    OrderId ahead = b->limit(kBid, 10001, 10, kNoise);
    b->market(kAsk, 10, kNoise);
    CHECK_FALSE(b->is_live(ahead));
    CHECK(b->is_live(behind));
}

// ---------------------------------------------------------------- fills ----

TEST_CASE("a partial fill leaves the remainder at the queue head", "[fill]") {
    auto b = make_book();
    OrderId id = b->limit(kAsk, 10000, 10, kNoise);
    Qty filled = b->market(kBid, 4, kNoise);
    CHECK(filled == 4);
    CHECK(b->qty_of(id) == 6);
    CHECK(b->volume_at(kAsk, 10000) == 6);
    CHECK(b->queue_ahead(id) == 0);
    CHECK(b->cum_volume() == 4);
    CHECK(b->last_trade_price() == 10000);
}

TEST_CASE("an aggressive order sweeps levels in price order", "[fill][cross]") {
    auto b = make_book();
    b->limit(kAsk, 10000, 5, kNoise);
    b->limit(kAsk, 10001, 5, kNoise);
    b->limit(kAsk, 10002, 5, kNoise);
    CHECK(b->market(kBid, 12, kNoise) == 12);
    CHECK(b->best_ask() == 10002);
    CHECK(b->volume_at(kAsk, 10002) == 3);
    CHECK(b->cum_volume() == 12);
    CHECK(b->last_trade_price() == 10002);
}

TEST_CASE("a market order larger than the book fills what exists", "[fill][edge]") {
    auto b = make_book();
    b->limit(kAsk, 10000, 5, kNoise);
    CHECK(b->market(kBid, 500, kNoise) == 5);
    CHECK(b->best_ask() == -1);
    CHECK(b->live_orders() == 0);
}

TEST_CASE("a market order on an empty book fills nothing", "[fill][edge]") {
    auto b = make_book();
    CHECK(b->market(kBid, 100, kNoise) == 0);
    CHECK(b->market(kAsk, 100, kNoise) == 0);
    CHECK(b->cum_volume() == 0);
    CHECK(b->last_trade_price() == -1);
}

TEST_CASE("a non-positive market order is a no-op", "[fill][edge]") {
    auto b = make_book();
    b->limit(kAsk, 10000, 5, kNoise);
    CHECK(b->market(kBid, 0, kNoise) == 0);
    CHECK(b->market(kBid, -3, kNoise) == 0);
    CHECK(b->volume_at(kAsk, 10000) == 5);
}

// -------------------------------------------------------------- crossing ---

TEST_CASE("a marketable limit crosses then rests the remainder", "[cross]") {
    auto b = make_book();
    b->limit(kAsk, 10000, 5, kNoise);
    OrderId id = b->limit(kBid, 10000, 12, kNoise);
    REQUIRE(id > 0);
    CHECK(b->qty_of(id) == 7);
    CHECK(b->best_bid() == 10000);
    CHECK(b->best_ask() == -1);
    CHECK(b->cum_volume() == 5);
}

TEST_CASE("a limit that is exactly consumed rests nothing", "[cross]") {
    auto b = make_book();
    b->limit(kAsk, 10000, 10, kNoise);
    CHECK(b->limit(kBid, 10000, 10, kNoise) == 0);   // 0 means fully filled
    CHECK(b->live_orders() == 0);
    CHECK(b->best_bid() == -1);
    CHECK(b->best_ask() == -1);
}

TEST_CASE("a limit priced away from the touch does not cross", "[cross]") {
    auto b = make_book();
    b->limit(kAsk, 10005, 5, kNoise);
    OrderId id = b->limit(kBid, 10000, 5, kNoise);
    CHECK(b->qty_of(id) == 5);
    CHECK(b->volume_at(kAsk, 10005) == 5);
    CHECK(b->cum_volume() == 0);
    CHECK(b->best_bid() < b->best_ask());
}

TEST_CASE("the book is never left crossed after a sweep", "[cross]") {
    auto b = make_book();
    for (Tick p = 10000; p < 10005; ++p) b->limit(kAsk, p, 4, kNoise);
    b->limit(kBid, 10003, 100, kNoise);      // sweeps 10000..10003, rests above
    REQUIRE(b->has_bid());
    REQUIRE(b->has_ask());
    CHECK(b->best_bid() < b->best_ask());
}

// -------------------------------------------------------------- cancels ----

TEST_CASE("a cancelled order leaves the book", "[cancel]") {
    auto b = make_book();
    OrderId id = b->limit(kBid, 10000, 10, kNoise);
    CHECK(b->cancel(id));
    CHECK_FALSE(b->is_live(id));
    CHECK(b->best_bid() == -1);
    CHECK(b->volume_at(kBid, 10000) == 0);
    CHECK(b->count_at(kBid, 10000) == 0);
    CHECK(b->live_orders() == 0);
}

TEST_CASE("cancelling twice, or cancelling nothing, fails cleanly", "[cancel][edge]") {
    auto b = make_book();
    OrderId id = b->limit(kBid, 10000, 10, kNoise);
    REQUIRE(b->cancel(id));
    CHECK_FALSE(b->cancel(id));
    CHECK_FALSE(b->cancel(0));
    CHECK_FALSE(b->cancel(-1));
    CHECK_FALSE(b->cancel(999999999));
}

TEST_CASE("cancelling a filled order fails", "[cancel][edge]") {
    auto b = make_book();
    OrderId id = b->limit(kAsk, 10000, 10, kNoise);
    b->market(kBid, 10, kNoise);
    CHECK_FALSE(b->is_live(id));
    CHECK_FALSE(b->cancel(id));
}

TEST_CASE("the touch walks back when its level empties", "[cancel]") {
    auto b = make_book();
    OrderId top = b->limit(kBid, 10000, 10, kNoise);
    b->limit(kBid, 9995, 10, kNoise);
    REQUIRE(b->cancel(top));
    CHECK(b->best_bid() == 9995);

    OrderId low = b->limit(kAsk, 10010, 10, kNoise);
    b->limit(kAsk, 10020, 10, kNoise);
    REQUIRE(b->cancel(low));
    CHECK(b->best_ask() == 10020);
}

TEST_CASE("a recycled arena slot does not answer to the old id", "[cancel][id]") {
    auto b = make_book();
    OrderId first = b->limit(kBid, 10000, 10, kNoise);
    REQUIRE(b->cancel(first));
    OrderId second = b->limit(kBid, 10000, 10, kNoise);
    CHECK(second != first);
    CHECK_FALSE(b->is_live(first));
    CHECK(b->is_live(second));
    CHECK(b->qty_of(first) == 0);
}

TEST_CASE("ids stay distinct across heavy recycling", "[id]") {
    // The slot index repeats constantly here; only the generation counter keeps
    // the ids apart.
    auto b = make_book();
    OrderId previous = 0;
    for (int i = 0; i < 20000; ++i) {
        OrderId id = b->limit(kBid, 10000, 1, kNoise);
        REQUIRE(id > 0);
        CHECK(id != previous);
        if (previous != 0) CHECK_FALSE(b->is_live(previous));
        REQUIRE(b->cancel(id));
        previous = id;
    }
}

// ------------------------------------------------------- owners and self ---

TEST_CASE("fills record both sides' owners", "[owner]") {
    auto b = make_book();
    b->limit(kBid, 10000, 10, 7);
    b->market(kAsk, 10, 42);
    const auto& ring = b->ring();
    bool saw_fill = false;
    for (std::size_t i = 0; i < ring.size(); ++i) {
        const Event& e = ring.data()[(ring.head() + i) % ring.capacity()];
        if (e.type == kEvFill) {
            saw_fill = true;
            CHECK(e.owner == 7);
            CHECK(e.aggr_owner == 42);
            CHECK(e.side == kBid);
        }
    }
    CHECK(saw_fill);
}

TEST_CASE("the engine does NOT prevent self-trading", "[owner][known-limitation]") {
    // Characterising current behaviour, not endorsing it. The engine matches on
    // price and time alone and never compares owners, so one owner can trade
    // with itself. Callers that need self-trade prevention must enforce it
    // above the engine, which is what the environment does by clamping its own
    // quotes so they cannot cross.
    auto b = make_book();
    OrderId resting = b->limit(kBid, 10000, 10, 5);
    Qty filled = b->market(kAsk, 10, 5);        // same owner on both sides
    CHECK(filled == 10);
    CHECK_FALSE(b->is_live(resting));
    CHECK(b->cum_volume() == 10);
}

// ------------------------------------------------------------- snapshot ----

TEST_CASE("a snapshot is ordered outward from the touch and padded", "[snapshot]") {
    auto b = make_book();
    for (int d = 0; d < 3; ++d) {
        b->limit(kBid, static_cast<Tick>(10000 - d), 10 + d, kNoise);
        b->limit(kAsk, static_cast<Tick>(10010 + d), 20 + d, kNoise);
    }
    constexpr int K = 5;
    Tick bid_px[K], ask_px[K];
    std::int64_t bid_vol[K], ask_vol[K];
    b->snapshot(K, bid_px, bid_vol, ask_px, ask_vol);

    CHECK(bid_px[0] == 10000);
    CHECK(bid_px[1] == 9999);
    CHECK(bid_px[2] == 9998);
    CHECK(ask_px[0] == 10010);
    CHECK(ask_px[2] == 10012);
    CHECK(bid_vol[0] == 10);
    CHECK(ask_vol[2] == 22);
    for (int i = 3; i < K; ++i) {
        CHECK(bid_px[i] == -1);
        CHECK(bid_vol[i] == 0);
        CHECK(ask_px[i] == -1);
        CHECK(ask_vol[i] == 0);
    }
}

TEST_CASE("a snapshot of an empty book is all padding", "[snapshot][edge]") {
    auto b = make_book();
    constexpr int K = 3;
    Tick bid_px[K], ask_px[K];
    std::int64_t bid_vol[K], ask_vol[K];
    b->snapshot(K, bid_px, bid_vol, ask_px, ask_vol);
    for (int i = 0; i < K; ++i) {
        CHECK(bid_px[i] == -1);
        CHECK(ask_px[i] == -1);
        CHECK(bid_vol[i] == 0);
        CHECK(ask_vol[i] == 0);
    }
}

// ------------------------------------------------------------ ring buffer --

TEST_CASE("the event ring wraps and keeps the newest records", "[events]") {
    auto b = make_book();
    const std::size_t cap = kRingSize;
    for (std::size_t i = 0; i < cap + 100; ++i) {
        OrderId id = b->limit(kBid, 10000, 1, kNoise);
        REQUIRE(b->cancel(id));                   // two events per iteration
    }
    const auto& ring = b->ring();
    CHECK(ring.size() == cap);
    CHECK(ring.total() == static_cast<std::int64_t>((cap + 100) * 2));
    // The oldest readable record is total - capacity.
    const Event& oldest = ring.data()[ring.head()];
    CHECK(oldest.seq == ring.total() - static_cast<std::int64_t>(cap));
}

// ----------------------------------------------------------------- reset ---

TEST_CASE("reset returns the book to its initial state", "[reset]") {
    auto b = make_book();
    b->limit(kBid, 10000, 10, kNoise);
    b->limit(kAsk, 10010, 10, kNoise);
    b->market(kBid, 5, kNoise);
    b->limit(kBid, -1, 5, kNoise);               // a reject, to dirty the counter
    REQUIRE(b->cum_volume() > 0);

    b->reset();
    CHECK(b->best_bid() == -1);
    CHECK(b->best_ask() == -1);
    CHECK(b->live_orders() == 0);
    CHECK(b->cum_volume() == 0);
    CHECK(b->rejects() == 0);
    CHECK(b->last_trade_price() == -1);
    CHECK(b->ring().total() == 0);
    CHECK(resting_volume(*b) == 0);
}

TEST_CASE("the book is reusable after reset", "[reset]") {
    auto b = make_book();
    for (int round = 0; round < 3; ++round) {
        b->reset();
        OrderId id = b->limit(kBid, 10000, 10, kNoise);
        REQUIRE(id > 0);
        CHECK(b->best_bid() == 10000);
        CHECK(b->live_orders() == 1);
    }
}

// ------------------------------------------------------------- exhaustion --

TEST_CASE("arena exhaustion rejects rather than allocating", "[edge][arena]") {
    auto b = make_book();
    for (std::size_t i = 0; i < kMaxOrders; ++i) {
        REQUIRE(b->limit(kBid, 10000, 1, kNoise) > 0);
    }
    CHECK(b->live_orders() == static_cast<Idx>(kMaxOrders));
    CHECK(b->limit(kBid, 10000, 1, kNoise) == -1);
    CHECK(b->rejects() == 1);
    CHECK(b->volume_at(kBid, 10000) == static_cast<std::int64_t>(kMaxOrders));
}
