from __future__ import annotations

import unittest

import pandas as pd

from news_quant.ranking import FactorSpec, build_rankings, cross_sectional_percentile


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


if __name__ == "__main__":
    unittest.main()
