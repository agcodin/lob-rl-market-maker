#pragma once
#include <array>
#include <cassert>
#include "lob/types.hpp"

namespace lob {

// Order node living inside the arena. `next`/`prev` are arena indices, not
// pointers: the whole FIFO queue of a price level is one contiguous region of
// the same cache-friendly array.
struct Order {
    Qty          qty;
    Tick         price;
    Idx          next;
    Idx          prev;
    std::int32_t gen = 1;  // bumped on recycle so stale ids cannot alias;
                           // starts at 1 so no valid id is ever 0 or negative
    std::uint8_t side;
    std::uint8_t owner;
    std::uint8_t live;
    std::uint8_t _pad;
};

// Fixed-capacity arena with an O(1) index-based free stack. No allocation ever
// happens after construction; exhaustion is reported, not grown around.
template <typename T, std::size_t N>
class StaticObjectPool {
  public:
    StaticObjectPool() { reset(); }

    // Returns every slot to the free stack in place; never touches the heap.
    void reset() {
        for (std::size_t i = 0; i < N; ++i) {
            slots_[i] = T{};
            free_stack_[i] = static_cast<Idx>(N - 1 - i);
        }
        free_top_ = static_cast<Idx>(N);
    }

    static constexpr std::size_t capacity() { return N; }
    Idx in_use() const { return static_cast<Idx>(N) - free_top_; }

    // Returns kNullIdx when the arena is exhausted.
    Idx acquire() {
        if (free_top_ == 0) return kNullIdx;
        return free_stack_[--free_top_];
    }

    void release(Idx idx) {
        assert(idx >= 0 && static_cast<std::size_t>(idx) < N);
        free_stack_[free_top_++] = idx;
    }

    T&       operator[](Idx i)       { return slots_[static_cast<std::size_t>(i)]; }
    const T& operator[](Idx i) const { return slots_[static_cast<std::size_t>(i)]; }

  private:
    std::array<T, N>   slots_{};
    std::array<Idx, N> free_stack_{};
    Idx                free_top_ = 0;
};

}  // namespace lob
