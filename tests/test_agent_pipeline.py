from __future__ import annotations

import unittest

from news_quant.agent_pipeline import build_markdown_report


class AgentPipelineReportTest(unittest.TestCase):
    def test_build_markdown_report_contains_key_sections(self) -> None:
        summary = {
            "news_path": "data/sample.jsonl",
            "universe_path": "data/universe.csv",
            "ranking_preset": "optimized",
            "top_k": 3,
            "steps": [
                {"agent_name": "analysis_agent", "status": "ok", "message": "完成文章级结构化"},
                {"agent_name": "ranking_agent", "status": "ok", "message": "完成 top k 排序"},
            ],
            "ranking_summary": {
                "date_count": 5,
                "stock_count": 10,
                "avg_cross_section_size": 8.2,
                "topk_by_date": [
                    {
                        "publish_date": "2023-11-01",
                        "selected_stocks": [{"stock_name": "A"}, {"stock_name": "B"}],
                    }
                ],
            },
            "performance_summary": {
                "horizons": {
                    "1": {"avg_excess_return": 0.01, "hit_rate_vs_universe": 0.6},
                    "3": {"avg_excess_return": 0.02, "hit_rate_vs_universe": 0.7},
                    "5": {"avg_excess_return": -0.01, "hit_rate_vs_universe": 0.4},
                }
            },
        }

        report = build_markdown_report(summary)
        self.assertIn("第四章 Agent 系统运行报告", report)
        self.assertIn("analysis_agent", report)
        self.assertIn("2023-11-01", report)
        self.assertIn("future 3D", report)
        self.assertIn("top3", report)


if __name__ == "__main__":
    unittest.main()
