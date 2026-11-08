#pragma once
// A deliberately naive order book: std::map of price to a std::deque of orders.
// It is the obvious implementation, written for clarity with no regard for
// speed, and it exists for two reasons:
//
//   1. as the reference side of a differential fuzz test, so the fast engine
//      has to agree with something simple on every single operation;
//   2. as the baseline in the benchmark, so "4 ns" has something to be 4 ns
//      *against*.
//
// It must mirror lob::OrderBook's semantics exactly, including that a limit
// order sweeps the opposing side before resting its remainder.
#include <cstdint>
#include <deque>
#include <map>
#include <unordered_map>

#include "lob/types.hpp"

namespace ref {

struct Order {
    lob::OrderId id;
    lob::Qty qty;
    lob::OwnerId owner;
};

class ReferenceBook {
  public:
    // Returns the resting id, 0 if fully filled, -1 on reject -- as the engine.
    lob::OrderId limit(lob::Side side, lob::Tick price, lob::Qty qty, lob::OwnerId owner) {
        if (qty <= 0 || price < 0 || static_cast<std::size_t>(price) >= lob::kMaxTicks) {
            ++rejects_;
            return -1;
        }
        lob::Qty remaining = sweep(side, price, qty);
        if (remaining == 0) return 0;
        lob::OrderId id = next_id_++;
        auto& side_map = (side == lob::kBid) ? bids_ : asks_;
        side_map[price].push_back(Order{id, remaining, owner});
        where_[id] = {side, price};
        return id;
    }

    lob::Qty market(lob::Side side, lob::Qty qty, lob::OwnerId) {
        if (qty <= 0) return 0;
        lob::Tick limit_price =
            (side == lob::kBid) ? static_cast<lob::Tick>(lob::kMaxTicks - 1) : 0;
        return qty - sweep(side, limit_price, qty);
    }

    bool cancel(lob::OrderId id) {
        auto it = where_.find(id);
        if (it == where_.end()) return false;
        auto [side, price] = it->second;
        auto& side_map = (side == lob::kBid) ? bids_ : asks_;
        auto level = side_map.find(price);
        if (level == side_map.end()) return false;
        auto& queue = level->second;
        for (auto q = queue.begin(); q != queue.end(); ++q) {
            if (q->id == id) {
                queue.erase(q);
                break;
            }
        }
        if (queue.empty()) side_map.erase(level);
        where_.erase(it);
        return true;
    }

    lob::Tick best_bid() const { return bids_.empty() ? -1 : bids_.rbegin()->first; }
    lob::Tick best_ask() const { return asks_.empty() ? -1 : asks_.begin()->first; }

    std::int64_t volume_at(lob::Side side, lob::Tick price) const {
        const auto& side_map = (side == lob::kBid) ? bids_ : asks_;
        auto level = side_map.find(price);
        if (level == side_map.end()) return 0;
        std::int64_t total = 0;
        for (const Order& o : level->second) total += o.qty;
        return total;
    }

    std::int32_t count_at(lob::Side side, lob::Tick price) const {
        const auto& side_map = (side == lob::kBid) ? bids_ : asks_;
        auto level = side_map.find(price);
        return level == side_map.end() ? 0 : static_cast<std::int32_t>(level->second.size());
    }

    // Shares resting ahead of `id` in its own queue; -1 if it is not resting.
    std::int64_t queue_ahead(lob::OrderId id) const {
        auto it = where_.find(id);
        if (it == where_.end()) return -1;
        auto [side, price] = it->second;
        const auto& side_map = (side == lob::kBid) ? bids_ : asks_;
        auto level = side_map.find(price);
        if (level == side_map.end()) return -1;
        std::int64_t ahead = 0;
        for (const Order& o : level->second) {
            if (o.id == id) return ahead;
            ahead += o.qty;
        }
        return -1;
    }

    bool is_live(lob::OrderId id) const { return where_.count(id) != 0; }
    std::size_t live_orders() const { return where_.size(); }
    std::int64_t cum_volume() const { return cum_volume_; }
    std::int64_t rejects() const { return rejects_; }

    std::int64_t resting_volume() const {
        std::int64_t total = 0;
        for (const auto& [price, queue] : bids_)
            for (const Order& o : queue) total += o.qty;
        for (const auto& [price, queue] : asks_)
            for (const Order& o : queue) total += o.qty;
        return total;
    }

  private:
    // Consumes liquidity up to `price`; returns the unfilled remainder.
    lob::Qty sweep(lob::Side side, lob::Tick price, lob::Qty qty) {
        lob::Qty remaining = qty;
        if (side == lob::kBid) {
            while (remaining > 0 && !asks_.empty() && asks_.begin()->first <= price) {
                remaining = consume(asks_, asks_.begin()->first, remaining);
            }
        } else {
            while (remaining > 0 && !bids_.empty() && bids_.rbegin()->first >= price) {
                remaining = consume(bids_, bids_.rbegin()->first, remaining);
            }
        }
        return remaining;
    }

    lob::Qty consume(std::map<lob::Tick, std::deque<Order>>& side_map, lob::Tick price,
                     lob::Qty remaining) {
        auto& queue = side_map[price];
        while (remaining > 0 && !queue.empty()) {
            Order& head = queue.front();
            lob::Qty traded = remaining < head.qty ? remaining : head.qty;
            remaining -= traded;
            head.qty -= traded;
            cum_volume_ += traded;
            if (head.qty == 0) {
                where_.erase(head.id);
                queue.pop_front();
            }
        }
        if (queue.empty()) side_map.erase(price);
        return remaining;
    }

    std::map<lob::Tick, std::deque<Order>> bids_;
    std::map<lob::Tick, std::deque<Order>> asks_;
    std::unordered_map<lob::OrderId, std::pair<lob::Side, lob::Tick>> where_;
    lob::OrderId next_id_ = 1;
    std::int64_t cum_volume_ = 0;
    std::int64_t rejects_ = 0;
};

}  // namespace ref
