#pragma once
#include <cstdint>

namespace lob {

using Tick   = std::int32_t;   // price in ticks (cents)
using Qty    = std::int32_t;
using OrderId = std::int64_t;
using Idx    = std::int32_t;

inline constexpr Idx kNullIdx = -1;

enum Side : std::uint8_t { kBid = 0, kAsk = 1 };

// Owner tag: lets the Python layer separate agent flow from synthetic flow
// without a side table.
enum Owner : std::uint8_t { kNoise = 0, kAgent = 1 };

enum EventType : std::int32_t {
    kEvNew     = 0,
    kEvFill    = 1,   // one fill leg (emitted once per resting order touched)
    kEvCancel  = 2,
    kEvReject  = 3,
};

// Flat POD event record. Layout is mirrored by a NumPy structured dtype in
// Python, so the ring buffer is read with zero copies.
struct Event {
    std::int64_t seq;
    std::int32_t type;
    std::int32_t side;        // side of the resting/placed order
    std::int64_t order_id;    // resting order for fills
    std::int64_t aggressor_id;
    std::int32_t price;
    std::int32_t qty;
    std::uint8_t owner;       // owner of order_id
    std::uint8_t aggr_owner;  // owner of the aggressor (fills only)
    std::uint8_t _pad[6];
};
static_assert(sizeof(Event) == 48, "Event layout must stay packed for NumPy");

}  // namespace lob
