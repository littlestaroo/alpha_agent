from __future__ import annotations

import unittest

import pandas as pd

from news_quant.ranking import (
    FactorSpec,
    add_derived_ranking_features,
    build_rankings,
    cross_sectional_percentile,
)


class RankingModelTest(unittest.TestCase):
    def test_percentile_handles_constant_cross_section(self) -> None:
        values = pd.Series([1.0, 1.0, 1.0])
        result = cross_sectional_percentile(values)
        self.assertEqual(result.tolist(), [0.5, 0.5, 0.5])

    def test_build_rankings_selects_top_k_with_negative_direction(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "publish_date": "2023-11-01",
                    "ts_code": "000001.SZ",
                    "stock_name": "A",
                    "net_sentiment_factor": 0.8,
                    "negative_shock_factor": 0.1,
                },
                {
                    "publish_date": "2023-11-01",
                    "ts_code": "000002.SZ",
                    "stock_name": "B",
                    "net_sentiment_factor": 0.4,
                    "negative_shock_factor": 0.7,
                },
                {
                    "publish_date": "2023-11-01",
                    "ts_code": "000003.SZ",
                    "stock_name": "C",
                    "net_sentiment_factor": 0.6,
                    "negative_shock_factor": 0.2,
                },
            ]
        )
        specs = [
            FactorSpec("net_sentiment_factor", "净情绪", "text", 0.6, 1),
            FactorSpec("negative_shock_factor", "负面冲击", "text", 0.4, -1),
        ]

        ranked = build_rankings(df, top_k=2, factor_specs=specs)
        selected = ranked[ranked["selected_topk"] == 1]["stock_name"].tolist()

        self.assertEqual(ranked.loc[0, "stock_name"], "A")
        self.assertEqual(selected, ["A", "C"])
        self.assertLess(
            ranked.loc[ranked["stock_name"] == "B", "negative_shock_factor__signal"].iloc[0],
            ranked.loc[ranked["stock_name"] == "C", "negative_shock_factor__signal"].iloc[0],
        )

    def test_custom_existing_stock_factor_can_join_text_factor(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "publish_date": "2023-11-01",
                    "ts_code": "000001.SZ",
                    "stock_name": "A",
                    "net_sentiment_factor": 0.9,
                    "momentum_20d": -0.1,
                },
                {
                    "publish_date": "2023-11-01",
                    "ts_code": "000002.SZ",
                    "stock_name": "B",
                    "net_sentiment_factor": 0.3,
                    "momentum_20d": 0.8,
                },
            ]
        )
        specs = [
            FactorSpec("net_sentiment_factor", "净情绪", "text", 0.4, 1),
            FactorSpec("momentum_20d", "20日动量", "market", 0.6, 1),
        ]

        ranked = build_rankings(df, top_k=1, factor_specs=specs)

        self.assertEqual(ranked.loc[0, "stock_name"], "B")
        self.assertEqual(ranked.loc[0, "selected_topk"], 1)
        self.assertIn("text_group_score", ranked.columns)
        self.assertIn("market_group_score", ranked.columns)

    def test_derived_reliability_features_are_created(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "publish_date": "2023-11-01",
                    "ts_code": "000001.SZ",
                    "stock_name": "A",
                    "article_count": 3,
                    "mention_count": 3,
                    "attention_factor": 0.6,
                    "sentiment_dispersion_factor": 0.1,
                    "net_sentiment_factor": 0.8,
                    "ema5_sentiment_state": 0.7,
                }
            ]
        )

        enriched = add_derived_ranking_features(df)
        self.assertIn("sentiment_confirmation_factor", enriched.columns)
        self.assertIn("coverage_reliability_factor", enriched.columns)
        self.assertGreater(enriched.loc[0, "sentiment_confirmation_factor"], 0.6)
        self.assertGreater(enriched.loc[0, "coverage_reliability_factor"], 0.7)

    def test_reliability_multiplier_shrinks_noisy_high_score(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "publish_date": "2023-11-01",
                    "ts_code": "000001.SZ",
                    "stock_name": "稳健高分",
                    "net_sentiment_factor": 0.9,
                    "negative_shock_factor": 0.0,
                    "attention_factor": 0.62,
                    "composite_score": 0.8,
                    "ema5_sentiment_state": 0.82,
                    "negative_shock_carry": 0.0,
                    "state_composite_factor": 0.78,
                    "operations_event_state": 0.5,
                    "risk_event_state": 0.0,
                    "event_novelty_factor": 0.4,
                    "sentiment_dispersion_factor": 0.05,
                    "event_density_factor": 0.76,
                    "article_count": 3,
                    "mention_count": 3,
                    "earnings_event_state": 0.2,
                    "market_buzz_event_state": 0.1,
                },
                {
                    "publish_date": "2023-11-01",
                    "ts_code": "000002.SZ",
                    "stock_name": "噪声高分",
                    "net_sentiment_factor": 0.92,
                    "negative_shock_factor": 0.0,
                    "attention_factor": 0.25,
                    "composite_score": 0.82,
                    "ema5_sentiment_state": 0.1,
                    "negative_shock_carry": 0.1,
                    "state_composite_factor": 0.15,
                    "operations_event_state": 0.1,
                    "risk_event_state": 0.0,
                    "event_novelty_factor": 0.02,
                    "sentiment_dispersion_factor": 0.7,
                    "event_density_factor": 0.32,
                    "article_count": 1,
                    "mention_count": 1,
                    "earnings_event_state": 0.0,
                    "market_buzz_event_state": 0.0,
                },
            ]
        )
        specs = [
            FactorSpec("net_sentiment_factor", "净情绪", "direct", 0.06, 1),
            FactorSpec("negative_shock_factor", "负面冲击", "direct", 0.06, -1),
            FactorSpec("attention_factor", "关注度", "direct", 0.06, 1),
            FactorSpec("composite_score", "综合分", "direct", 0.06, 1),
            FactorSpec("event_novelty_factor", "新颖性", "direct", 0.05, 1),
            FactorSpec("ema5_sentiment_state", "状态", "state", 0.10, 1),
            FactorSpec("negative_shock_carry", "负面记忆", "state", 0.08, -1),
            FactorSpec("state_composite_factor", "状态综合", "state", 0.10, 1),
            FactorSpec("sentiment_confirmation_factor", "确认", "state", 0.12, 1),
            FactorSpec("coverage_reliability_factor", "覆盖可靠性", "reliability", 0.09, 1),
            FactorSpec("sentiment_dispersion_factor", "分歧", "reliability", 0.08, -1),
            FactorSpec("operations_event_state", "经营事件", "event", 0.12, 1),
            FactorSpec("risk_event_state", "风险事件", "event", 0.10, -1),
            FactorSpec("event_density_factor", "事件密度", "event", 0.04, 1),
            FactorSpec("earnings_event_state", "业绩事件", "event", 0.04, 1),
            FactorSpec("market_buzz_event_state", "市场舆情", "event", 0.04, 1),
        ]

        ranked = build_rankings(df, top_k=1, factor_specs=specs)
        self.assertEqual(ranked.loc[0, "stock_name"], "稳健高分")
        self.assertGreater(ranked.loc[0, "reliability_multiplier"], ranked.loc[1, "reliability_multiplier"])
        self.assertGreater(ranked.loc[0, "ranking_score"], ranked.loc[1, "ranking_score"])


if __name__ == "__main__":
    unittest.main()
