#pragma once
#include <algorithm>
#include <array>
#include <cstring>
#include "lob/pool.hpp"
#include "lob/ring.hpp"
#include "lob/types.hpp"

namespace lob {

inline constexpr std::size_t kMaxTicks   = 1u << 17;   // direct-indexed price grid
inline constexpr std::size_t kMaxOrders  = 1u << 18;   // arena capacity
inline constexpr std::size_t kRingSize   = 1u << 14;   // event ring
inline constexpr int         kSlotBits   = 20;

struct Level {
    Idx          head  = kNullIdx;
    Idx          tail  = kNullIdx;
    std::int64_t volume = 0;
    std::int32_t count  = 0;
    std::int32_t _pad   = 0;
};

// Price-time priority matching engine.
//
// Hot path invariants: no allocation, no hashing, no pointer chasing outside
// the arena. Price lookup is a single indexed load into a flat level array;
// queue traversal walks arena indices.
class OrderBook {
  public:
    OrderBook() { reset(); }

    void reset() {
        pool_.reset();
        bid_levels_.fill(Level{});
        ask_levels_.fill(Level{});
        ring_.clear();
        best_bid_ = -1;
        best_ask_ = static_cast<Tick>(kMaxTicks);
        seq_ = 0;
        last_trade_price_ = -1;
        cum_volume_ = 0;
        rejects_ = 0;
    }

    // ---- queries -------------------------------------------------------

    Tick best_bid() const { return best_bid_; }
    Tick best_ask() const { return best_ask_ == static_cast<Tick>(kMaxTicks) ? -1 : best_ask_; }
    bool has_bid()  const { return best_bid_ >= 0; }
    bool has_ask()  const { return best_ask_ < static_cast<Tick>(kMaxTicks); }
    double mid() const {
        if (has_bid() && has_ask()) return 0.5 * (best_bid_ + best_ask_);
        if (has_bid()) return best_bid_;
        if (has_ask()) return best_ask_;
        return last_trade_price_ >= 0 ? last_trade_price_ : 0.0;
    }
    Tick spread() const { return (has_bid() && has_ask()) ? best_ask_ - best_bid_ : -1; }
    Tick last_trade_price() const { return last_trade_price_; }
    std::int64_t cum_volume() const { return cum_volume_; }
    std::int64_t rejects() const { return rejects_; }
    Idx live_orders() const { return pool_.in_use(); }

    std::int64_t volume_at(Side s, Tick p) const {
        if (p < 0 || static_cast<std::size_t>(p) >= kMaxTicks) return 0;
        return lvl(s, p).volume;
    }
    std::int32_t count_at(Side s, Tick p) const {
        if (p < 0 || static_cast<std::size_t>(p) >= kMaxTicks) return 0;
        return lvl(s, p).count;
    }
    // Newest resting order at a level -- the natural cancel candidate for the
    // synthetic flow model, and O(1) to reach.
    OrderId tail_id_at(Side s, Tick p) const {
        if (p < 0 || static_cast<std::size_t>(p) >= kMaxTicks) return -1;
        Idx t = lvl(s, p).tail;
        return t == kNullIdx ? -1 : make_id(t, pool_[t].gen);
    }
    Qty qty_of(OrderId id) const {
        Idx i = resolve(id);
        return i == kNullIdx ? 0 : pool_[i].qty;
    }
    bool is_live(OrderId id) const { return resolve(id) != kNullIdx; }

    // Number of shares ahead of `id` in its own queue (queue position).
    std::int64_t queue_ahead(OrderId id) const {
        Idx i = resolve(id);
        if (i == kNullIdx) return -1;
        const Order& o = pool_[i];
        std::int64_t ahead = 0;
        for (Idx j = lvl(static_cast<Side>(o.side), o.price).head; j != kNullIdx && j != i;
             j = pool_[j].next) {
            ahead += pool_[j].qty;
        }
        return ahead;
    }

    // ---- mutations -----------------------------------------------------

    // Marketable limit order. Sweeps the opposing book, rests the remainder.
    // Returns the resting order id, 0 if fully filled, -1 on reject.
    OrderId limit(Side side, Tick price, Qty qty, Owner owner) {
        if (qty <= 0 || price < 0 || static_cast<std::size_t>(price) >= kMaxTicks) {
            ++rejects_;
            emit(kEvReject, side, -1, -1, price, qty, owner, owner);
            return -1;
        }
        Qty remaining = sweep(side, price, qty, /*aggressor=*/-1, owner);
        if (remaining == 0) return 0;
        return rest(side, price, remaining, owner);
    }

    // Pure taker order. Returns filled quantity.
    Qty market(Side side, Qty qty, Owner owner) {
        if (qty <= 0) return 0;
        Tick limit_price = (side == kBid) ? static_cast<Tick>(kMaxTicks - 1) : 0;
        return qty - sweep(side, limit_price, qty, /*aggressor=*/-1, owner);
    }

    bool cancel(OrderId id) {
        Idx i = resolve(id);
        if (i == kNullIdx) return false;
        Order& o = pool_[i];
        Side s = static_cast<Side>(o.side);
        Tick p = o.price;
        Qty q = o.qty;
        Owner ow = static_cast<Owner>(o.owner);
        unlink(s, p, i);
        recycle(i);
        emit(kEvCancel, s, id, -1, p, q, ow, ow);
        return true;
    }

    // ---- market data ---------------------------------------------------

    // Writes K levels per side into caller-owned buffers. Empty levels are
    // padded with price=-1, volume=0.
    void snapshot(int K, Tick* bid_px, std::int64_t* bid_vol,
                  Tick* ask_px, std::int64_t* ask_vol) const {
        int n = 0;
        for (Tick p = best_bid_; p >= 0 && n < K; --p) {
            const Level& l = bid_levels_[static_cast<std::size_t>(p)];
            if (l.volume > 0) { bid_px[n] = p; bid_vol[n] = l.volume; ++n; }
        }
        for (; n < K; ++n) { bid_px[n] = -1; bid_vol[n] = 0; }

        n = 0;
        for (Tick p = best_ask_; static_cast<std::size_t>(p) < kMaxTicks && n < K; ++p) {
            const Level& l = ask_levels_[static_cast<std::size_t>(p)];
            if (l.volume > 0) { ask_px[n] = p; ask_vol[n] = l.volume; ++n; }
        }
        for (; n < K; ++n) { ask_px[n] = -1; ask_vol[n] = 0; }
    }

    EventRing<kRingSize>&       ring()       { return ring_; }
    const EventRing<kRingSize>& ring() const { return ring_; }

  private:
    static OrderId make_id(Idx slot, std::int32_t gen) {
        return (static_cast<OrderId>(gen) << kSlotBits) | static_cast<OrderId>(slot);
    }

    Idx resolve(OrderId id) const {
        if (id <= 0) return kNullIdx;
        Idx slot = static_cast<Idx>(id & ((OrderId{1} << kSlotBits) - 1));
        if (slot < 0 || static_cast<std::size_t>(slot) >= kMaxOrders) return kNullIdx;
        const Order& o = pool_[slot];
        if (!o.live) return kNullIdx;
        if (o.gen != static_cast<std::int32_t>(id >> kSlotBits)) return kNullIdx;
        return slot;
    }

    Level&       lvl(Side s, Tick p)       { return (s == kBid ? bid_levels_ : ask_levels_)[static_cast<std::size_t>(p)]; }
    const Level& lvl(Side s, Tick p) const { return (s == kBid ? bid_levels_ : ask_levels_)[static_cast<std::size_t>(p)]; }

    void emit(EventType t, Side s, OrderId id, OrderId aggr, Tick p, Qty q,
              Owner owner, Owner aggr_owner) {
        Event e{};
        e.seq = seq_++;
        e.type = t;
        e.side = s;
        e.order_id = id;
        e.aggressor_id = aggr;
        e.price = p;
        e.qty = q;
        e.owner = owner;
        e.aggr_owner = aggr_owner;
        ring_.push(e);
    }

    void unlink(Side s, Tick p, Idx i) {
        Level& L = lvl(s, p);
        Order& o = pool_[i];
        if (o.prev != kNullIdx) pool_[o.prev].next = o.next; else L.head = o.next;
        if (o.next != kNullIdx) pool_[o.next].prev = o.prev; else L.tail = o.prev;
        L.volume -= o.qty;
        L.count  -= 1;
        if (L.count == 0) {
            L.volume = 0;
            L.head = L.tail = kNullIdx;
            if (s == kBid && p == best_bid_) {
                while (best_bid_ >= 0 && bid_levels_[static_cast<std::size_t>(best_bid_)].count == 0) --best_bid_;
            } else if (s == kAsk && p == best_ask_) {
                while (static_cast<std::size_t>(best_ask_) < kMaxTicks &&
                       ask_levels_[static_cast<std::size_t>(best_ask_)].count == 0) ++best_ask_;
            }
        }
    }

    void recycle(Idx i) {
        pool_[i].live = 0;
        pool_[i].gen += 1;
        pool_.release(i);
    }

    OrderId rest(Side side, Tick price, Qty qty, Owner owner) {
        Idx i = pool_.acquire();
        if (i == kNullIdx) {
            ++rejects_;
            emit(kEvReject, side, -1, -1, price, qty, owner, owner);
            return -1;
        }
        Order& o = pool_[i];
        o.qty = qty;
        o.price = price;
        o.side = side;
        o.owner = owner;
        o.live = 1;
        o.next = kNullIdx;

        Level& L = lvl(side, price);
        o.prev = L.tail;
        if (L.tail != kNullIdx) pool_[L.tail].next = i; else L.head = i;
        L.tail = i;
        L.volume += qty;
        L.count  += 1;

        if (side == kBid) { if (price > best_bid_) best_bid_ = price; }
        else              { if (price < best_ask_) best_ask_ = price; }

        OrderId id = make_id(i, o.gen);
        emit(kEvNew, side, id, -1, price, qty, owner, owner);
        return id;
    }

    // Consumes liquidity up to `price`; returns the unfilled remainder.
    Qty sweep(Side side, Tick price, Qty qty, OrderId aggressor_id, Owner aggr_owner) {
        Qty remaining = qty;
        if (side == kBid) {
            while (remaining > 0 && has_ask() && best_ask_ <= price) {
                remaining = consume_level(kAsk, best_ask_, remaining, aggressor_id, aggr_owner);
            }
        } else {
            while (remaining > 0 && has_bid() && best_bid_ >= price) {
                remaining = consume_level(kBid, best_bid_, remaining, aggressor_id, aggr_owner);
            }
        }
        return remaining;
    }

    Qty consume_level(Side book_side, Tick price, Qty remaining, OrderId aggressor_id,
                      Owner aggr_owner) {
        Level& L = lvl(book_side, price);
        while (remaining > 0 && L.head != kNullIdx) {
            Idx i = L.head;
            Order& o = pool_[i];
            Qty traded = std::min(remaining, o.qty);
            remaining -= traded;
            o.qty     -= traded;
            L.volume  -= traded;
            last_trade_price_ = price;
            cum_volume_ += traded;
            emit(kEvFill, book_side, make_id(i, o.gen), aggressor_id, price, traded,
                 static_cast<Owner>(o.owner), aggr_owner);
            if (o.qty == 0) {
                L.head = o.next;
                if (L.head != kNullIdx) pool_[L.head].prev = kNullIdx; else L.tail = kNullIdx;
                L.count -= 1;
                recycle(i);
            }
        }
        if (L.count == 0) {
            L.volume = 0;
            L.head = L.tail = kNullIdx;
            if (book_side == kBid) {
                while (best_bid_ >= 0 && bid_levels_[static_cast<std::size_t>(best_bid_)].count == 0) --best_bid_;
            } else {
                while (static_cast<std::size_t>(best_ask_) < kMaxTicks &&
                       ask_levels_[static_cast<std::size_t>(best_ask_)].count == 0) ++best_ask_;
            }
        }
        return remaining;
    }

    StaticObjectPool<Order, kMaxOrders> pool_{};
    std::array<Level, kMaxTicks> bid_levels_{};
    std::array<Level, kMaxTicks> ask_levels_{};
    EventRing<kRingSize> ring_{};
    Tick best_bid_ = -1;
    Tick best_ask_ = static_cast<Tick>(kMaxTicks);
    std::int64_t seq_ = 0;
    Tick last_trade_price_ = -1;
    std::int64_t cum_volume_ = 0;
    std::int64_t rejects_ = 0;
};

}  // namespace lob
