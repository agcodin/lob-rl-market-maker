#pragma once
#include <array>
#include <cstddef>
#include "lob/types.hpp"

namespace lob {

// Single-producer ring of POD events. Overwrites the oldest record when full;
// the reader tracks `head`/`count` to detect drops.
template <std::size_t N>
class EventRing {
  public:
    static_assert((N & (N - 1)) == 0, "ring capacity must be a power of two");

    void push(const Event& e) {
        buf_[write_ & (N - 1)] = e;
        ++write_;
    }

    // Records currently readable, oldest first.
    std::size_t size() const { return static_cast<std::size_t>(write_) < N ? static_cast<std::size_t>(write_) : N; }
    std::int64_t total() const { return write_; }
    std::size_t head() const { return static_cast<std::size_t>(write_ - static_cast<std::int64_t>(size())) & (N - 1); }
    static constexpr std::size_t capacity() { return N; }

    Event*       data()       { return buf_.data(); }
    const Event* data() const { return buf_.data(); }

    void clear() { write_ = 0; }

  private:
    std::array<Event, N> buf_{};
    std::int64_t         write_ = 0;
};

}  // namespace lob
