from __future__ import annotations

from barrick_unified.lse_adapter import LSEMarketDataProvider, TREASURY_TENORS


class FakeClient:
    def __init__(self) -> None:
        self.candle_calls: list[tuple] = []
        self.yield_calls: list[tuple] = []

    def catalog(self, category):
        assert category == "stocks"
        return [{"symbol": item, "name": item} for item in ("B", "GOLD")]

    def candles(self, *args, **kwargs):
        self.candle_calls.append((args, kwargs))
        return [{"symbol": args[0]}]

    def options(self, *args, **kwargs):
        return [{"ticker": "GLD-test"}]

    def options_underlyings(self):
        return [{"underlying": "GLD"}]

    def bond_yields(self, *args, **kwargs):
        self.yield_calls.append((args, kwargs))
        return [{"symbol": args[0]}]


def test_adapter_validates_catalog_and_fetches_once_per_symbol() -> None:
    client = FakeClient()
    provider = LSEMarketDataProvider(client=client)
    result = provider.fetch_daily_stock_candles(["B", "GOLD"], "2026-01-01", "2026-02-01")
    assert set(result) == {"B", "GOLD"}
    assert len(client.candle_calls) == 2
    assert all(call[1]["dataset"] == "stocks" for call in client.candle_calls)


def test_option_input_fetch_is_one_chain_plus_all_tenors() -> None:
    client = FakeClient()
    provider = LSEMarketDataProvider(client=client)
    options, yields = provider.fetch_gld_option_inputs()
    assert len(options) == 1
    assert len(yields) == len(TREASURY_TENORS)
    assert [call[0][0] for call in client.yield_calls] == list(TREASURY_TENORS)


def test_market_fetch_uses_stock_catalog_and_options_underlying_catalog() -> None:
    client = FakeClient()
    provider = LSEMarketDataProvider(client=client)
    result, unavailable, catalog = provider.fetch_daily_market_candles(
        ["B", "GOLD", "GLD"], "2026-01-01", "2026-02-01"
    )
    assert set(result) == {"B", "GOLD", "GLD"}
    assert unavailable == []
    assert set(catalog) == {"B", "GOLD"}
    b_call = next(call for call in client.candle_calls if call[0][0] == "B")
    gld_call = next(call for call in client.candle_calls if call[0][0] == "GLD")
    assert b_call[1]["dataset"] == "stocks"
    assert "dataset" not in gld_call[1]
