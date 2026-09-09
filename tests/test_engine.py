import numpy as np
import pytest

from lobrl import EventType, OrderBook, Owner, Side
from lobrl._lobcore import MAX_ORDERS, RING_SIZE


@pytest.fixture
def book():
    return OrderBook()


def test_empty_book(book):
    assert book.best_bid() == -1
    assert book.best_ask() == -1
    assert book.spread() == -1


def test_rest_and_touch(book):
    book.limit(Side.BID, 100, 10, Owner.NOISE)
    book.limit(Side.BID, 99, 5, Owner.NOISE)
    book.limit(Side.ASK, 102, 7, Owner.NOISE)
    assert (book.best_bid(), book.best_ask()) == (100, 102)
    assert book.mid() == 101.0
    assert book.spread() == 2
    assert book.volume_at(Side.BID, 100) == 10


def test_price_time_priority(book):
    first = book.limit(Side.BID, 100, 10, Owner.NOISE)
    second = book.limit(Side.BID, 100, 10, Owner.AGENT)
    assert book.queue_ahead(first) == 0
    assert book.queue_ahead(second) == 10
    book.market(Side.ASK, 10, Owner.NOISE)
    assert not book.is_live(first)      # oldest filled first
    assert book.is_live(second)


def test_partial_fill_keeps_queue_head(book):
    oid = book.limit(Side.ASK, 100, 10, Owner.NOISE)
    book.market(Side.BID, 4, Owner.AGENT)
    assert book.qty_of(oid) == 6
    assert book.volume_at(Side.ASK, 100) == 6


def test_sweep_multiple_levels(book):
    for px, q in ((100, 5), (101, 5), (102, 5)):
        book.limit(Side.ASK, px, q, Owner.NOISE)
    filled = book.market(Side.BID, 12, Owner.AGENT)
    assert filled == 12
    assert book.best_ask() == 102
    assert book.volume_at(Side.ASK, 102) == 3


def test_market_order_beyond_book(book):
    book.limit(Side.ASK, 100, 5, Owner.NOISE)
    assert book.market(Side.BID, 50, Owner.AGENT) == 5
    assert book.best_ask() == -1


def test_marketable_limit_rests_remainder(book):
    book.limit(Side.ASK, 100, 5, Owner.NOISE)
    oid = book.limit(Side.BID, 100, 12, Owner.AGENT)
    assert oid > 0
    assert book.qty_of(oid) == 7
    assert book.best_bid() == 100
    assert book.best_ask() == -1


def test_limit_does_not_cross_worse_price(book):
    book.limit(Side.ASK, 105, 5, Owner.NOISE)
    oid = book.limit(Side.BID, 100, 5, Owner.AGENT)
    assert book.qty_of(oid) == 5
    assert book.volume_at(Side.ASK, 105) == 5


def test_cancel_and_stale_id(book):
    oid = book.limit(Side.BID, 100, 10, Owner.NOISE)
    assert book.cancel(oid) is True
    assert book.cancel(oid) is False       # already gone
    assert book.best_bid() == -1
    # A recycled slot must not answer to the old id.
    new = book.limit(Side.BID, 100, 10, Owner.NOISE)
    assert new != oid
    assert book.is_live(oid) is False


def test_best_price_walks_back_on_empty_level(book):
    top = book.limit(Side.BID, 100, 10, Owner.NOISE)
    book.limit(Side.BID, 98, 10, Owner.NOISE)
    book.cancel(top)
    assert book.best_bid() == 98


def test_snapshot_padding_and_ordering(book):
    for d in range(3):
        book.limit(Side.BID, 100 - d, 10 + d, Owner.NOISE)
        book.limit(Side.ASK, 110 + d, 20 + d, Owner.NOISE)
    K = 5
    bp, bv = np.zeros(K, np.int32), np.zeros(K, np.int64)
    ap, av = np.zeros(K, np.int32), np.zeros(K, np.int64)
    book.snapshot_into(bp, bv, ap, av)
    assert list(bp[:3]) == [100, 99, 98]
    assert list(ap[:3]) == [110, 111, 112]
    assert list(bv[:3]) == [10, 11, 12]
    assert list(bp[3:]) == [-1, -1] and list(bv[3:]) == [0, 0]


def test_snapshot_rejects_mismatched_buffers(book):
    with pytest.raises(ValueError):
        book.snapshot_into(np.zeros(5, np.int32), np.zeros(4, np.int64),
                           np.zeros(5, np.int32), np.zeros(5, np.int64))


def test_event_buffer_is_a_zero_copy_view(book):
    ev = book.events_buffer()
    assert ev.shape == (RING_SIZE,)
    assert ev.base is not None            # shares the C++ ring's memory
    book.limit(Side.BID, 100, 10, Owner.AGENT)
    assert book.events_total() == 1
    rec = ev[0]
    assert rec["type"] == int(EventType.NEW)
    assert rec["price"] == 100 and rec["qty"] == 10
    assert rec["owner"] == int(Owner.AGENT)


def test_fill_events_carry_both_owners(book):
    book.limit(Side.BID, 100, 10, Owner.AGENT)
    book.market(Side.ASK, 10, Owner.NOISE)
    ev = book.events_buffer()[: book.events_size()]
    fills = ev[ev["type"] == int(EventType.FILL)]
    assert len(fills) == 1
    assert fills[0]["owner"] == int(Owner.AGENT)
    assert fills[0]["aggr_owner"] == int(Owner.NOISE)
    assert fills[0]["side"] == int(Side.BID)


def test_reset_clears_everything(book):
    book.limit(Side.BID, 100, 10, Owner.NOISE)
    book.reset()
    assert book.best_bid() == -1
    assert book.live_orders() == 0
    assert book.events_total() == 0


def test_invalid_orders_are_rejected(book):
    assert book.limit(Side.BID, 100, 0, Owner.NOISE) == -1
    assert book.limit(Side.BID, -5, 10, Owner.NOISE) == -1
    assert book.rejects() == 2


def test_arena_exhaustion_is_reported_not_crashed(book):
    # Filling the arena must reject cleanly rather than allocate.
    for i in range(MAX_ORDERS):
        if book.limit(Side.BID, 100, 1, Owner.NOISE) == -1:
            pytest.fail("rejected before capacity")
    assert book.limit(Side.BID, 100, 1, Owner.NOISE) == -1
    assert book.rejects() == 1


def test_cumulative_volume_and_last_trade(book):
    book.limit(Side.ASK, 100, 10, Owner.NOISE)
    book.market(Side.BID, 6, Owner.AGENT)
    assert book.cum_volume() == 6
    assert book.last_trade_price() == 100
