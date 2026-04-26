from __future__ import annotations

import unittest

import pandas as pd

from news_quant.baseline import PROFILE_COLUMNS, build_daily_profiles


class BuildDailyProfilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mentions_df = pd.DataFrame(
            [
                {
                    "news_id": "n1",
                    "publish_date": "2023-11-01",
                    "publish_time": "2023-11-01 00:00:00",
                    "ts_code": "000001.SZ",
                    "stock_name": "测试股票",
                    "sentiment": 0.6,
                    "relevance": 1.0,
                    "confidence": 1.0,
                    "event_importance": 1.0,
                    "sentiment_strength": 1.0,
                    "risk_flag": 0,
                    "event_type": "业绩",
                    "keywords": "业绩|盈利",
                },
                {
                    "news_id": "n2",
                    "publish_date": "2023-11-02",
                    "publish_time": "2023-11-02 00:00:00",
                    "ts_code": "000001.SZ",
                    "stock_name": "测试股票",
                    "sentiment": -0.4,
                    "relevance": 1.0,
                    "confidence": 1.0,
                    "event_importance": 1.0,
                    "sentiment_strength": 1.0,
                    "risk_flag": 1,
                    "event_type": "风险",
                    "keywords": "风险|下滑",
                },
                {
                    "news_id": "n3",
                    "publish_date": "2023-11-03",
                    "publish_time": "2023-11-03 00:00:00",
                    "ts_code": "000001.SZ",
                    "stock_name": "测试股票",
                    "sentiment": 0.2,
                    "relevance": 1.0,
                    "confidence": 1.0,
                    "event_importance": 1.0,
                    "sentiment_strength": 1.0,
                    "risk_flag": 0,
                    "event_type": "经营",
                    "keywords": "订单|合作",
                },
            ]
        )
        self.universe_df = pd.DataFrame(
            [{"ts_code": "000001.SZ", "name": "测试股票", "industry": "测试行业"}]
        )

    def test_profiles_include_enhanced_columns(self) -> None:
        profiles_df = build_daily_profiles(self.mentions_df, self.universe_df)
        expected_columns = {
            "ema3_sentiment_state",
            "ema5_sentiment_state",
            "negative_shock_carry",
            "event_novelty_factor",
            "earnings_event_factor",
            "operations_event_factor",
            "financing_event_factor",
            "market_buzz_event_factor",
            "risk_event_factor",
            "earnings_event_state",
            "operations_event_state",
            "financing_event_state",
            "market_buzz_event_state",
            "risk_event_state",
            "state_composite_factor",
        }
        self.assertTrue(expected_columns.issubset(set(PROFILE_COLUMNS)))
        self.assertTrue(expected_columns.issubset(set(profiles_df.columns)))

    def test_time_state_and_carry_values(self) -> None:
        profiles_df = build_daily_profiles(self.mentions_df, self.universe_df)

        self.assertEqual(len(profiles_df), 3)
        self.assertAlmostEqual(profiles_df.loc[0, "net_sentiment_factor"], 0.6, places=4)
        self.assertAlmostEqual(profiles_df.loc[1, "net_sentiment_factor"], -0.4, places=4)
        self.assertAlmostEqual(profiles_df.loc[2, "net_sentiment_factor"], 0.2, places=4)

        self.assertAlmostEqual(profiles_df.loc[0, "ema3_sentiment_state"], 0.6, places=4)
        self.assertAlmostEqual(profiles_df.loc[1, "ema3_sentiment_state"], 0.1, places=4)
        self.assertAlmostEqual(profiles_df.loc[2, "ema3_sentiment_state"], 0.15, places=4)

        self.assertAlmostEqual(profiles_df.loc[0, "negative_shock_carry"], 0.0, places=4)
        self.assertAlmostEqual(profiles_df.loc[1, "negative_shock_carry"], 0.4, places=4)
        self.assertAlmostEqual(profiles_df.loc[2, "negative_shock_carry"], 0.26, places=4)

    def test_event_factor_and_state_values(self) -> None:
        profiles_df = build_daily_profiles(self.mentions_df, self.universe_df)

        self.assertAlmostEqual(profiles_df.loc[0, "earnings_event_factor"], 0.6, places=4)
        self.assertAlmostEqual(profiles_df.loc[1, "risk_event_factor"], -0.4, places=4)
        self.assertAlmostEqual(profiles_df.loc[2, "operations_event_factor"], 0.2, places=4)

        self.assertAlmostEqual(profiles_df.loc[0, "earnings_event_state"], 0.6, places=4)
        self.assertAlmostEqual(profiles_df.loc[1, "risk_event_state"], -0.1333, places=4)
        self.assertAlmostEqual(profiles_df.loc[2, "operations_event_state"], 0.0667, places=4)

        for value in profiles_df["state_composite_factor"].tolist():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
