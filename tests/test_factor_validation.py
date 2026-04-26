from __future__ import annotations

import unittest

import pandas as pd

from analysis.validate_chapter3_factors import (
    attach_future_returns,
    compute_direction_validation,
    compute_time_series_validation,
)


class FactorValidationTest(unittest.TestCase):
    def test_attach_future_returns_uses_next_trading_day(self) -> None:
        profile_df = pd.DataFrame(
            [
                {
                    "publish_date": "2023-11-03",
                    "ts_code": "300750.SZ",
                    "stock_name": "宁德时代",
                    "net_sentiment_factor": 0.4,
                    "negative_shock_factor": 0.0,
                    "attention_factor": 1.0,
                    "sentiment_momentum_factor": 0.3,
                    "composite_score": 0.5,
                    "ema3_sentiment_state": 0.4,
                    "ema5_sentiment_state": 0.35,
                    "negative_shock_carry": 0.0,
                    "event_novelty_factor": 0.1,
                    "earnings_event_state": 0.2,
                    "operations_event_state": 0.3,
                    "market_buzz_event_state": 0.1,
                    "risk_event_state": 0.0,
                    "state_composite_factor": 0.4,
                },
                {
                    "publish_date": "2023-11-04",
                    "ts_code": "300750.SZ",
                    "stock_name": "宁德时代",
                    "net_sentiment_factor": 0.5,
                    "negative_shock_factor": 0.0,
                    "attention_factor": 1.1,
                    "sentiment_momentum_factor": 0.4,
                    "composite_score": 0.55,
                    "ema3_sentiment_state": 0.45,
                    "ema5_sentiment_state": 0.40,
                    "negative_shock_carry": 0.0,
                    "event_novelty_factor": 0.2,
                    "earnings_event_state": 0.25,
                    "operations_event_state": 0.35,
                    "market_buzz_event_state": 0.15,
                    "risk_event_state": 0.0,
                    "state_composite_factor": 0.45,
                },
            ]
        )
        prices_df = pd.DataFrame(
            [
                {"trade_date": "2023-11-03", "ts_code": "300750.SZ", "close": 100.0, "fwd_return_1d": 0.02, "fwd_return_3d": 0.03, "fwd_return_5d": 0.04},
                {"trade_date": "2023-11-06", "ts_code": "300750.SZ", "close": 102.0, "fwd_return_1d": -0.01, "fwd_return_3d": 0.01, "fwd_return_5d": 0.02},
                {"trade_date": "2023-11-07", "ts_code": "300750.SZ", "close": 101.0, "fwd_return_1d": 0.00, "fwd_return_3d": 0.02, "fwd_return_5d": 0.03},
            ]
        )
        prices_df["trade_date"] = pd.to_datetime(prices_df["trade_date"])

        merged = attach_future_returns(profile_df, prices_df)

        self.assertEqual(str(merged.loc[0, "anchor_trade_date"].date()), "2023-11-06")
        self.assertAlmostEqual(merged.loc[0, "fwd_return_1d"], -0.01)
        self.assertEqual(str(merged.loc[1, "anchor_trade_date"].date()), "2023-11-06")

    def test_time_series_validation_uses_expected_direction(self) -> None:
        panel_df = pd.DataFrame(
            [
                {"net_sentiment_factor": 0.9, "negative_shock_factor": 0.1, "attention_factor": 1.4, "sentiment_momentum_factor": 0.7, "composite_score": 0.8, "ema3_sentiment_state": 0.7, "ema5_sentiment_state": 0.6, "negative_shock_carry": 0.1, "event_novelty_factor": 0.3, "earnings_event_state": 0.4, "operations_event_state": 0.5, "market_buzz_event_state": 0.2, "risk_event_state": 0.0, "state_composite_factor": 0.6, "fwd_return_1d": 0.05, "fwd_return_3d": 0.06, "fwd_return_5d": 0.08},
                {"net_sentiment_factor": 0.7, "negative_shock_factor": 0.2, "attention_factor": 1.3, "sentiment_momentum_factor": 0.5, "composite_score": 0.7, "ema3_sentiment_state": 0.6, "ema5_sentiment_state": 0.55, "negative_shock_carry": 0.2, "event_novelty_factor": 0.2, "earnings_event_state": 0.3, "operations_event_state": 0.4, "market_buzz_event_state": 0.2, "risk_event_state": 0.0, "state_composite_factor": 0.5, "fwd_return_1d": 0.03, "fwd_return_3d": 0.04, "fwd_return_5d": 0.05},
                {"net_sentiment_factor": 0.5, "negative_shock_factor": 0.3, "attention_factor": 1.1, "sentiment_momentum_factor": 0.2, "composite_score": 0.5, "ema3_sentiment_state": 0.4, "ema5_sentiment_state": 0.4, "negative_shock_carry": 0.3, "event_novelty_factor": 0.0, "earnings_event_state": 0.2, "operations_event_state": 0.2, "market_buzz_event_state": 0.1, "risk_event_state": 0.1, "state_composite_factor": 0.3, "fwd_return_1d": 0.01, "fwd_return_3d": 0.02, "fwd_return_5d": 0.02},
                {"net_sentiment_factor": 0.2, "negative_shock_factor": 0.5, "attention_factor": 0.9, "sentiment_momentum_factor": -0.1, "composite_score": 0.1, "ema3_sentiment_state": 0.2, "ema5_sentiment_state": 0.25, "negative_shock_carry": 0.5, "event_novelty_factor": -0.2, "earnings_event_state": 0.0, "operations_event_state": -0.1, "market_buzz_event_state": 0.0, "risk_event_state": 0.2, "state_composite_factor": 0.1, "fwd_return_1d": -0.02, "fwd_return_3d": -0.03, "fwd_return_5d": -0.04},
                {"net_sentiment_factor": -0.1, "negative_shock_factor": 0.7, "attention_factor": 0.8, "sentiment_momentum_factor": -0.3, "composite_score": -0.1, "ema3_sentiment_state": 0.0, "ema5_sentiment_state": 0.1, "negative_shock_carry": 0.7, "event_novelty_factor": -0.4, "earnings_event_state": -0.1, "operations_event_state": -0.2, "market_buzz_event_state": -0.1, "risk_event_state": 0.3, "state_composite_factor": -0.1, "fwd_return_1d": -0.04, "fwd_return_3d": -0.05, "fwd_return_5d": -0.06},
            ]
        )

        result = compute_time_series_validation(panel_df)
        net_5d = result[(result["factor"] == "net_sentiment_factor") & (result["horizon"] == 5)].iloc[0]
        neg_5d = result[(result["factor"] == "negative_shock_factor") & (result["horizon"] == 5)].iloc[0]

        self.assertGreater(net_5d["expected_top_bottom_gap"], 0)
        self.assertGreater(neg_5d["expected_top_bottom_gap"], 0)

    def test_direction_validation_reports_market_baseline(self) -> None:
        panel_df = pd.DataFrame(
            [
                {"net_sentiment_factor": 0.8, "negative_shock_factor": 0.2, "attention_factor": 1.0, "sentiment_momentum_factor": 0.5, "composite_score": 0.6, "ema3_sentiment_state": 0.5, "ema5_sentiment_state": 0.5, "negative_shock_carry": 0.1, "event_novelty_factor": 0.2, "earnings_event_state": 0.4, "operations_event_state": 0.3, "market_buzz_event_state": 0.2, "risk_event_state": 0.0, "state_composite_factor": 0.5, "fwd_return_1d": 0.03, "fwd_return_3d": -0.02, "fwd_return_5d": -0.04},
                {"net_sentiment_factor": 0.7, "negative_shock_factor": 0.3, "attention_factor": 1.0, "sentiment_momentum_factor": 0.4, "composite_score": 0.5, "ema3_sentiment_state": 0.4, "ema5_sentiment_state": 0.4, "negative_shock_carry": 0.2, "event_novelty_factor": 0.1, "earnings_event_state": 0.3, "operations_event_state": 0.2, "market_buzz_event_state": 0.1, "risk_event_state": 0.0, "state_composite_factor": 0.4, "fwd_return_1d": -0.01, "fwd_return_3d": -0.03, "fwd_return_5d": -0.05},
                {"net_sentiment_factor": 0.6, "negative_shock_factor": 0.4, "attention_factor": 1.0, "sentiment_momentum_factor": 0.3, "composite_score": 0.4, "ema3_sentiment_state": 0.3, "ema5_sentiment_state": 0.3, "negative_shock_carry": 0.3, "event_novelty_factor": 0.0, "earnings_event_state": 0.2, "operations_event_state": 0.1, "market_buzz_event_state": 0.0, "risk_event_state": 0.1, "state_composite_factor": 0.3, "fwd_return_1d": -0.02, "fwd_return_3d": -0.01, "fwd_return_5d": -0.02},
                {"net_sentiment_factor": -0.2, "negative_shock_factor": 0.6, "attention_factor": 1.0, "sentiment_momentum_factor": -0.2, "composite_score": -0.2, "ema3_sentiment_state": 0.0, "ema5_sentiment_state": 0.1, "negative_shock_carry": 0.4, "event_novelty_factor": -0.1, "earnings_event_state": -0.1, "operations_event_state": -0.2, "market_buzz_event_state": -0.1, "risk_event_state": 0.2, "state_composite_factor": 0.0, "fwd_return_1d": -0.03, "fwd_return_3d": -0.04, "fwd_return_5d": -0.06},
                {"net_sentiment_factor": -0.4, "negative_shock_factor": 0.8, "attention_factor": 1.0, "sentiment_momentum_factor": -0.4, "composite_score": -0.4, "ema3_sentiment_state": -0.1, "ema5_sentiment_state": 0.0, "negative_shock_carry": 0.5, "event_novelty_factor": -0.2, "earnings_event_state": -0.2, "operations_event_state": -0.3, "market_buzz_event_state": -0.2, "risk_event_state": 0.3, "state_composite_factor": -0.1, "fwd_return_1d": -0.04, "fwd_return_3d": -0.05, "fwd_return_5d": -0.07},
            ]
        )

        result = compute_direction_validation(panel_df)
        row = result[(result["factor"] == "net_sentiment_factor") & (result["horizon"] == 5)].iloc[0]
        self.assertGreater(row["market_majority_hit_rate"], 0.5)
        self.assertIn("excess_hit_rate_vs_market", result.columns)


if __name__ == "__main__":
    unittest.main()
