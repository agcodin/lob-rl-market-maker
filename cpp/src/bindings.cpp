#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lob/book.hpp"

namespace py = pybind11;
using namespace lob;

namespace {

// Mirrors lob::Event field-for-field so the ring can be read without a copy.
py::dtype event_dtype() {
    py::dict spec;
    spec["names"] = py::make_tuple("seq", "type", "side", "order_id", "aggressor_id",
                                   "price", "qty", "owner", "aggr_owner");
    spec["formats"] = py::make_tuple("<i8", "<i4", "<i4", "<i8", "<i8", "<i4", "<i4", "u1", "u1");
    spec["offsets"] = py::make_tuple(0, 8, 12, 16, 24, 32, 36, 40, 41);
    spec["itemsize"] = py::int_(static_cast<int>(sizeof(Event)));
    return py::dtype::from_args(spec);
}

// Writes a K-level snapshot straight into caller-provided NumPy buffers.
void snapshot_into(const OrderBook& b, py::array_t<std::int32_t, py::array::c_style> bid_px,
                   py::array_t<std::int64_t, py::array::c_style> bid_vol,
                   py::array_t<std::int32_t, py::array::c_style> ask_px,
                   py::array_t<std::int64_t, py::array::c_style> ask_vol) {
    const py::ssize_t K = bid_px.size();
    if (bid_vol.size() != K || ask_px.size() != K || ask_vol.size() != K) {
        throw std::invalid_argument("snapshot buffers must all have the same length");
    }
    b.snapshot(static_cast<int>(K), bid_px.mutable_data(), bid_vol.mutable_data(),
               ask_px.mutable_data(), ask_vol.mutable_data());
}

}  // namespace

PYBIND11_MODULE(_lobcore, m) {
    m.doc() = "Zero-allocation limit order book matching engine";

    py::enum_<Side>(m, "Side")
        .value("BID", kBid)
        .value("ASK", kAsk)
        .export_values();
    py::enum_<Owner>(m, "Owner")
        .value("NOISE", kNoise)
        .value("AGENT", kAgent)
        .export_values();
    py::enum_<EventType>(m, "EventType")
        .value("NEW", kEvNew)
        .value("FILL", kEvFill)
        .value("CANCEL", kEvCancel)
        .value("REJECT", kEvReject)
        .export_values();

    m.attr("MAX_TICKS")  = py::int_(static_cast<std::int64_t>(kMaxTicks));
    m.attr("MAX_ORDERS") = py::int_(static_cast<std::int64_t>(kMaxOrders));
    m.attr("RING_SIZE")  = py::int_(static_cast<std::int64_t>(kRingSize));
    m.attr("EVENT_ITEMSIZE") = py::int_(static_cast<std::int64_t>(sizeof(Event)));
    m.attr("MAX_OWNERS") = py::int_(static_cast<std::int64_t>(kMaxOwners));

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("reset", &OrderBook::reset)
        .def("limit", &OrderBook::limit, py::arg("side"), py::arg("price"), py::arg("qty"),
             py::arg("owner") = OwnerId{kNoise},
             "Submit a marketable limit order; returns resting id, 0 if fully filled, -1 on reject.")
        .def("market", &OrderBook::market, py::arg("side"), py::arg("qty"),
             py::arg("owner") = OwnerId{kNoise})
        .def("cancel", &OrderBook::cancel, py::arg("order_id"))
        .def("best_bid", &OrderBook::best_bid)
        .def("best_ask", &OrderBook::best_ask)
        .def("mid", &OrderBook::mid)
        .def("spread", &OrderBook::spread)
        .def("last_trade_price", &OrderBook::last_trade_price)
        .def("cum_volume", &OrderBook::cum_volume)
        .def("rejects", &OrderBook::rejects)
        .def("live_orders", &OrderBook::live_orders)
        .def("volume_at", &OrderBook::volume_at, py::arg("side"), py::arg("price"))
        .def("count_at", &OrderBook::count_at, py::arg("side"), py::arg("price"))
        .def("tail_id_at", &OrderBook::tail_id_at, py::arg("side"), py::arg("price"))
        .def("qty_of", &OrderBook::qty_of, py::arg("order_id"))
        .def("is_live", &OrderBook::is_live, py::arg("order_id"))
        .def("queue_ahead", &OrderBook::queue_ahead, py::arg("order_id"))
        .def("snapshot_into", &snapshot_into, py::arg("bid_px"), py::arg("bid_vol"),
             py::arg("ask_px"), py::arg("ask_vol"))
        .def("events_total", [](const OrderBook& b) { return b.ring().total(); })
        .def("events_size", [](const OrderBook& b) { return b.ring().size(); })
        .def("events_head", [](const OrderBook& b) { return b.ring().head(); })
        .def("clear_events", [](OrderBook& b) { b.ring().clear(); })
        // Zero-copy view over the whole ring as a structured array. The book is
        // held as the array's base, so the buffer cannot outlive it.
        .def("events_buffer", [](py::object self) {
            OrderBook& b = self.cast<OrderBook&>();
            py::dtype dt = event_dtype();
            return py::array(dt, {static_cast<py::ssize_t>(kRingSize)},
                             {static_cast<py::ssize_t>(sizeof(Event))},
                             b.ring().data(), self);
        });
}
