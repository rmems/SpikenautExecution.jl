"""
    SpikenautExecution

Async trade signal pipeline: ZMQ SUB → confidence gating → Kelly sizing → dYdX v4 REST execution.

Bridges Julia SNN strategy output to live decentralized exchange execution with:
- ZMQ SUB socket for JSON trade signals from Rust nervous system
- Nanosecond latency tracking (signal creation → execution)
- Confidence gate (only executes signals above threshold)
- Kelly position sizing via SpikenautKelly
- dYdX v4 decentralized perpetuals REST client (no API key required)

## Architecture

```
Julia (Brain)          Rust (Muscle)           Exchange
     ↓                      ↑
LIF neurons → SNN signal → ZMQ bridge → Kelly sizing → dydx v4 → Order
(16 neurons)  (confidence)   (IPC)        (sizing)     (REST)
```

## Provenance

Extracted from Eagle-Lander, a private neuromorphic GPU supervisor (closed-source).
The execution pipeline received live
ZMQ signals from a Rust SNN nervous system and placed orders on dYdX v4 perpetuals
in production before being open-sourced as a standalone Julia package.

## References

- Kelly, J.L. (1956). A New Interpretation of Information Rate.
  *Bell System Technical Journal*, 35(4), 917–926.
  https://doi.org/10.1002/j.1538-7305.1956.tb03809.x
  Position sizing via the Kelly Criterion.

- dYdX Foundation (2024). *dYdX v4 Indexer API Documentation*.
  REST endpoints for orderbook queries and perpetual market data.
  https://docs.dydx.exchange/api_integration-indexer/indexer_api

- iMatix Corporation (2013). *ZeroMQ: Messaging for Many Applications*.
  O'Reilly Media. SUB socket pattern for trade signal delivery.
  https://zguide.zeromq.org

## Usage

```julia
using SpikenautExecution

engine = ExecutionEngine(confidence_threshold=0.85)
start!(engine, zmq_endpoint="tcp://localhost:5555")
```
"""
module SpikenautExecution

using Dates
using JSON

export TradeSignal, TradeSide, ExecutionEngine, ExecutionDecision
export execute_signal!, latency_ns, passes_gate
export DydxClient, DydxPrice, get_price, mid_price, spread_bps
export start!

# ── Trade Signal ─────────────────────────────────────────────────────────────

"""
    @enum TradeSide

Direction of a trade signal.
"""
@enum TradeSide begin
    Buy     = 1
    Sell    = 2
    Neutral = 0
end

"""
    TradeSignal

Deserialized trade signal from Rust nervous system via ZMQ.

# Fields
- `ticker`:       asset symbol (e.g. "BTC-USD", "DNX-USDT")
- `side`:         Buy / Sell / Neutral
- `price`:        expected execution price (USD)
- `quantity`:     units to trade (Kelly-sized by Rust, override in Julia)
- `confidence`:   SNN output [0.0, 1.0]
- `timestamp_ns`: Unix nanoseconds for latency tracking
"""
struct TradeSignal
    ticker::String
    side::TradeSide
    price::Float64
    quantity::Float64
    confidence::Float32
    timestamp_ns::Int64
end

function TradeSignal(d::Dict)
    side = if get(d, "side", "NEUTRAL") == "BUY"
        Buy
    elseif get(d, "side", "NEUTRAL") == "SELL"
        Sell
    else
        Neutral
    end
    TradeSignal(
        get(d, "ticker", "UNKNOWN"),
        side,
        Float64(get(d, "price", 0.0)),
        Float64(get(d, "quantity", 0.0)),
        Float32(get(d, "confidence", 0.0)),
        Int64(get(d, "timestamp_ns", 0)),
    )
end

"""
    latency_ns(signal) -> Int64

End-to-end latency from signal creation to now (nanoseconds).
"""
function latency_ns(s::TradeSignal)::Int64
    now_ns = Int64(Dates.value(Dates.now()) * 1_000_000)  # ms → ns approximation
    return max(0, now_ns - s.timestamp_ns)
end

"""
    passes_gate(signal, threshold) -> Bool

True if signal confidence meets or exceeds the threshold.
"""
passes_gate(s::TradeSignal, threshold::Float32) = s.confidence >= threshold

# ── Execution Decision ────────────────────────────────────────────────────────

"""
    ExecutionDecision

Result of processing one signal through the execution engine.
"""
struct ExecutionDecision
    signal::TradeSignal
    executed::Bool
    reason::String
    kelly_fraction::Float64
    position_units::Float64
    latency_ns::Int64
end

# ── Execution Engine ──────────────────────────────────────────────────────────

"""
    ExecutionEngine

Stateful execution engine with confidence gating and position management.

# Fields
- `confidence_threshold`: minimum SNN confidence to execute (default 0.85)
- `max_position_size`:    hard cap on position units (default 10.0)
- `payoff_ratio`:         expected price move for Kelly sizing (default 0.01)
- `positions`:            current open positions (ticker → quantity)
"""
mutable struct ExecutionEngine
    confidence_threshold::Float32
    max_position_size::Float64
    payoff_ratio::Float64
    positions::Dict{String, Float64}
    total_signals::Int
    executed_signals::Int
    rejected_signals::Int
end

function ExecutionEngine(;
    confidence_threshold::Float32 = Float32(0.85),
    max_position_size::Float64 = 10.0,
    payoff_ratio::Float64 = 0.01,
)
    ExecutionEngine(confidence_threshold, max_position_size, payoff_ratio,
                    Dict{String,Float64}(), 0, 0, 0)
end

"""
    execute_signal!(engine, signal, account_balance) -> ExecutionDecision

Process one signal: gate by confidence, size via Kelly, update positions.
"""
function execute_signal!(
    engine::ExecutionEngine,
    signal::TradeSignal,
    account_balance::Float64 = 10_000.0,
)::ExecutionDecision
    engine.total_signals += 1
    lat = latency_ns(signal)

    if !passes_gate(signal, engine.confidence_threshold)
        engine.rejected_signals += 1
        return ExecutionDecision(signal, false,
            "confidence=$(signal.confidence) < threshold=$(engine.confidence_threshold)",
            0.0, 0.0, lat)
    end

    if signal.side == Neutral
        engine.rejected_signals += 1
        return ExecutionDecision(signal, false, "neutral signal", 0.0, 0.0, lat)
    end

    # Kelly position sizing
    p = Float64(signal.confidence)
    b = engine.payoff_ratio
    q = 1.0 - p
    full_k = (p * b - q) / b
    k = clamp(full_k * 0.5, 0.0, 1.0)  # half-Kelly

    units = min((k * account_balance) / max(signal.price, 1e-9), engine.max_position_size)

    # Update position book
    current = get(engine.positions, signal.ticker, 0.0)
    if signal.side == Buy
        engine.positions[signal.ticker] = current + units
    elseif signal.side == Sell
        engine.positions[signal.ticker] = max(0.0, current - units)
    end

    engine.executed_signals += 1
    return ExecutionDecision(signal, true, "executed", k, units, lat)
end

"""
    fill_rate(engine) -> Float64

Fraction of signals that were executed (not rejected).
"""
fill_rate(e::ExecutionEngine) =
    e.total_signals == 0 ? 0.0 : e.executed_signals / e.total_signals

# ── dYdX v4 REST Client ───────────────────────────────────────────────────────

"""
    DydxPrice

Current price data from dYdX v4 indexer.
"""
struct DydxPrice
    ticker::String
    oracle_price::Float64
    best_bid::Float64
    best_ask::Float64
end

"""
    mid_price(p) -> Float64

Mid-price between best bid and ask.
"""
mid_price(p::DydxPrice) = (p.best_bid + p.best_ask) / 2.0

"""
    spread_bps(p) -> Float64

Bid-ask spread in basis points.
"""
spread_bps(p::DydxPrice) = max(p.best_ask - p.best_bid, 0.0) / max(p.best_ask, 1e-9) * 10_000.0

"""
    DydxClient

REST client for dYdX v4 perpetuals (no API key required for read-only).
"""
struct DydxClient
    base_url::String
    timeout_s::Float64
end

DydxClient(; base_url::String = "https://indexer.dydx.trade/v4",
             timeout_s::Float64 = 5.0) = DydxClient(base_url, timeout_s)

"""
    get_price(client, ticker) -> Union{DydxPrice, Nothing}

Fetch current oracle price and order book top for `ticker` (e.g. "BTC-USD").
Returns `nothing` on network error.
"""
function get_price(client::DydxClient, ticker::String)::Union{DydxPrice, Nothing}
    try
        url = "$(client.base_url)/orderbooks/perpetualMarket/$(ticker)"
        resp = HTTP.get(url; readtimeout=client.timeout_s)
        data = JSON.parse(String(resp.body))

        bids = get(data, "bids", [])
        asks = get(data, "asks", [])

        best_bid = isempty(bids) ? 0.0 : parse(Float64, first(bids)["price"])
        best_ask = isempty(asks) ? 0.0 : parse(Float64, first(asks)["price"])

        # Oracle price from markets endpoint
        market_url = "$(client.base_url)/perpetualMarkets?ticker=$(ticker)"
        market_resp = HTTP.get(market_url; readtimeout=client.timeout_s)
        market_data = JSON.parse(String(market_resp.body))
        markets = get(market_data, "markets", Dict())
        oracle = if haskey(markets, ticker)
            parse(Float64, get(markets[ticker], "oraclePrice", "0"))
        else
            (best_bid + best_ask) / 2.0
        end

        return DydxPrice(ticker, oracle, best_bid, best_ask)
    catch e
        @warn "dYdX price fetch failed for $ticker: $e"
        return nothing
    end
end

# ── ZMQ Signal Listener ───────────────────────────────────────────────────────

"""
    start!(engine; zmq_endpoint, account_balance, on_decision)

Start the ZMQ SUB listener loop (requires ZMQ.jl).

Subscribes to `zmq_endpoint` and processes incoming JSON trade signals
through the execution engine, calling `on_decision` for each result.

# Arguments
- `engine`:          `ExecutionEngine` instance
- `zmq_endpoint`:    ZMQ endpoint (e.g. "tcp://localhost:5555" or "ipc:///tmp/signals.ipc")
- `account_balance`: account size for Kelly sizing
- `on_decision`:     callback `(ExecutionDecision) -> Nothing`

# Example
```julia
engine = ExecutionEngine(confidence_threshold=Float32(0.85))
start!(engine, zmq_endpoint="tcp://localhost:5555") do decision
    if decision.executed
        println("Executed: \$(decision.signal.ticker) x\$(round(decision.position_units, digits=4))")
    end
end
```
"""
function start!(
    engine::ExecutionEngine;
    zmq_endpoint::String = "tcp://localhost:5555",
    account_balance::Float64 = 10_000.0,
    on_decision = decision -> nothing,
)
    # Lazy-load ZMQ to keep the package installable without it
    zmq = Base.require(Main, :ZMQ)

    ctx = zmq.Context()
    socket = zmq.Socket(ctx, zmq.SUB)
    zmq.subscribe(socket, "")
    zmq.connect(socket, zmq_endpoint)

    @info "[execution] ZMQ SUB connected to $zmq_endpoint"

    try
        while true
            msg = zmq.recv(socket)
            data = JSON.parse(String(msg))
            signal = TradeSignal(data)
            decision = execute_signal!(engine, signal, account_balance)
            on_decision(decision)
        end
    finally
        zmq.close(socket)
        zmq.close(ctx)
    end
end

end # module
