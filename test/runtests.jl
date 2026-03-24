using Test
using SpikenautExecution

@testset "SpikenautExecution" begin
    @testset "TradeSignal" begin
        d = Dict("ticker"=>"BTC-USD","side"=>"BUY","price"=>65000.0,"quantity"=>0.01,"confidence"=>0.9,"timestamp_ns"=>0)
        s = TradeSignal(d)
        @test s.ticker == "BTC-USD"
        @test s.side == Buy
        @test s.confidence ≈ 0.9f0
        @test passes_gate(s, 0.85f0)
        @test !passes_gate(s, 0.95f0)
    end

    @testset "ExecutionEngine — confidence gate" begin
        engine = ExecutionEngine(confidence_threshold=0.85f0)

        # Signal above threshold → executed
        sig_hi = TradeSignal(Dict("ticker"=>"SOL-USD","side"=>"BUY","price"=>90.0,"quantity"=>1.0,"confidence"=>0.90,"timestamp_ns"=>0))
        dec = execute_signal!(engine, sig_hi, 10_000.0)
        @test dec.executed
        @test dec.position_units > 0.0
        @test dec.kelly_fraction > 0.0

        # Signal below threshold → rejected
        sig_lo = TradeSignal(Dict("ticker"=>"SOL-USD","side"=>"BUY","price"=>90.0,"quantity"=>1.0,"confidence"=>0.70,"timestamp_ns"=>0))
        dec_lo = execute_signal!(engine, sig_lo, 10_000.0)
        @test !dec_lo.executed
    end

    @testset "ExecutionEngine — position tracking" begin
        engine = ExecutionEngine()
        sig = TradeSignal(Dict("ticker"=>"DNX-USDT","side"=>"BUY","price"=>0.03,"quantity"=>100.0,"confidence"=>0.92,"timestamp_ns"=>0))
        execute_signal!(engine, sig, 500.0)
        @test get(engine.positions, "DNX-USDT", 0.0) > 0.0
    end

    @testset "fill_rate" begin
        engine = ExecutionEngine(confidence_threshold=0.85f0)
        for conf in [0.90, 0.70, 0.88, 0.60]
            s = TradeSignal(Dict("ticker"=>"BTC-USD","side"=>"BUY","price"=>65000.0,"quantity"=>0.01,"confidence"=>conf,"timestamp_ns"=>0))
            execute_signal!(engine, s, 10_000.0)
        end
        @test 0.0 < fill_rate(engine) < 1.0
    end

    @testset "DydxPrice" begin
        p = DydxPrice("BTC-USD", 65000.0, 64990.0, 65010.0)
        @test mid_price(p) ≈ 65000.0
        @test spread_bps(p) > 0.0
    end
end
