from __future__ import annotations

import argparse
import sys
from pathlib import Path

from news_quant.baseline import run_baseline
from news_quant.config import (
    DEFAULT_BODY_CHARS,
    MOCK_LLM,
    OPENNEWSARCHIVE_EXPERIMENT_PATH,
    OPENNEWSARCHIVE_PREPARED_PATH,
    OPENNEWSARCHIVE_RAW_DIR,
    ROOT,
)
from news_quant.dataset import prepare_opennewsarchive_dataset
from news_quant.experiment import build_experiment_subset
from news_quant.experiment import build_experiment_subset_from_raw
from news_quant.experiment import build_daily_topk_per_stock
from news_quant.experiment import refine_thesis_subset
from news_quant.experiment import select_thesis_llm_set
from news_quant.filtering import filter_stock_news_file


def run_main() -> None:
    parser = argparse.ArgumentParser(
        description="第三章 baseline：新闻 -> 股票情绪/事件结构化 -> 日频舆情画像 -> 因子"
    )
    parser.add_argument(
        "--news",
        type=Path,
        default=OPENNEWSARCHIVE_EXPERIMENT_PATH,
        help="新闻文件或目录，支持 csv/jsonl/parquet",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "data" / "stock_universe_sample.csv",
        help="股票池 CSV，建议包含 ts_code,name,industry,aliases",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "chapter3_baseline",
        help="输出前缀，将生成 article_mentions、daily_profiles、skipped_news 三个 CSV",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="关键词过滤，可重复传入，也可用逗号分隔",
    )
    parser.add_argument("--date-from", type=str, default="", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--date-to", type=str, default="", help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理多少条新闻；传 0 表示不限制",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="每条新闻最多保留多少只候选股票",
    )
    parser.add_argument(
        "--body-chars",
        type=int,
        default=DEFAULT_BODY_CHARS,
        help="送入 LLM 的正文最大字符数",
    )
    args = parser.parse_args()

    mentions_df, profiles_df, skipped_df = run_baseline(
        news_path=args.news,
        universe_path=args.universe,
        output_path=args.out,
        keywords=args.keyword,
        date_from=args.date_from or None,
        date_to=args.date_to or None,
        limit=args.limit,
        max_candidates=args.max_candidates,
        max_body_chars=args.body_chars,
    )

    mode = "MOCK_LLM" if MOCK_LLM else "API"
    print(f"模式: {mode}")
    print(f"文章级结果: {len(mentions_df)} 行")
    print(f"日频画像结果: {len(profiles_df)} 行")
    print(f"跳过新闻: {len(skipped_df)} 行")

    if not profiles_df.empty:
        print("\n日频画像预览：")
        print(profiles_df.head(10).to_string(index=False))


def prepare_data_main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m news_quant prepare-data",
        description="将 OpenNewsArchive 原始文件目录整理为本工程使用的规范化 JSONL",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=OPENNEWSARCHIVE_RAW_DIR,
        help="OpenNewsArchive 原始数据目录或单个文件",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OPENNEWSARCHIVE_PREPARED_PATH,
        help="规范化后的 JSONL 输出路径",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅调试时限制处理条数")
    parser.add_argument(
        "--languages",
        type=str,
        default="zh",
        help="保留的语言，逗号分隔；为空表示不过滤，例如 zh 或 zh,en",
    )
    args = parser.parse_args()

    language_set = {
        part.strip().lower()
        for part in args.languages.split(",")
        if part.strip()
    }

    df = prepare_opennewsarchive_dataset(
        raw_path=args.raw,
        output_path=args.out,
        limit=args.limit or None,
        languages=language_set or None,
    )
    print(f"已生成规范化数据: {args.out}")
    print(f"记录数: {len(df)}")
    if not df.empty:
        print(df.head(5).to_string(index=False))


def filter_data_main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m news_quant filter-data",
        description="对规范化新闻做股票/金融相关性轻筛选",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=OPENNEWSARCHIVE_PREPARED_PATH,
        help="规范化后的 JSONL 输入路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_stock_news.jsonl",
        help="筛选后的股票相关新闻 JSONL 输出路径",
    )
    parser.add_argument(
        "--rejected-out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_nonstock_news.jsonl",
        help="未通过筛选的新闻输出路径",
    )
    parser.add_argument(
        "--mode",
        choices=("broad", "strict"),
        default="broad",
        help="broad 为宽松筛选，strict 为中度严格筛选",
    )
    args = parser.parse_args()

    summary = filter_stock_news_file(
        input_path=args.input_path,
        output_path=args.out,
        rejected_path=args.rejected_out,
        mode=args.mode,
    )
    print(f"输入新闻数: {summary['total']}")
    print(f"保留新闻数: {summary['kept']}")
    print(f"剔除新闻数: {summary['rejected']}")
    print(f"筛选后文件: {args.out}")
    print(f"剔除文件: {args.rejected_out}")


def build_experiment_main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m news_quant build-experiment-set",
        description="从严格筛选后的新闻中抽取适合 baseline 的股票实验子集",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_stock_news_strict.jsonl",
        help="严格筛选后的股票相关新闻 JSONL",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "data" / "stock_universe_sample.csv",
        help="股票池 CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_experiment_set.jsonl",
        help="实验子集输出路径",
    )
    parser.add_argument(
        "--rejected-out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_experiment_rejected.jsonl",
        help="未进入实验子集的记录输出路径",
    )
    parser.add_argument(
        "--max-per-stock",
        type=int,
        default=500,
        help="每只股票最多保留多少条新闻",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=2500,
        help="实验子集最多保留多少条新闻",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default="2020-01-01",
        help="仅保留不早于该日期的新闻",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default="",
        help="仅保留不晚于该日期的新闻",
    )
    args = parser.parse_args()

    summary = build_experiment_subset(
        input_path=args.input_path,
        universe_path=args.universe,
        output_path=args.out,
        rejected_path=args.rejected_out,
        max_per_stock=args.max_per_stock,
        max_total=args.max_total,
        date_from=args.date_from or None,
        date_to=args.date_to or None,
    )
    print(f"输入新闻数: {summary['total']}")
    print(f"实验子集条数: {summary['kept']}")
    print(f"剔除条数: {summary['rejected']}")
    print(f"输出文件: {args.out}")
    print(f"各股票样本数: {summary['counts_by_stock']}")


def build_experiment_from_raw_main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m news_quant build-experiment-set-from-raw",
        description="直接从 OpenNewsArchive 原始 jsonl 分片中抽取目标股票实验子集",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=OPENNEWSARCHIVE_RAW_DIR / "1623-0000001" / "zh",
        help="原始 jsonl 目录或单个 jsonl 文件",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "data" / "stock_universe_thesis_2stocks.csv",
        help="目标股票池 CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_fullrange.jsonl",
        help="实验子集输出路径",
    )
    parser.add_argument(
        "--rejected-out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_fullrange_rejected.jsonl",
        help="未命中股票池的记录输出路径",
    )
    parser.add_argument(
        "--max-per-stock",
        type=int,
        default=0,
        help="每只股票最多保留多少条；0 表示不限",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="总共最多保留多少条；0 表示不限",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default="2023-01-01",
        help="仅保留不早于该日期的新闻",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default="2023-12-31",
        help="仅保留不晚于该日期的新闻",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="语言过滤，默认 zh",
    )
    args = parser.parse_args()

    summary = build_experiment_subset_from_raw(
        raw_path=args.raw,
        universe_path=args.universe,
        output_path=args.out,
        rejected_path=args.rejected_out,
        max_per_stock=args.max_per_stock,
        max_total=args.max_total,
        date_from=args.date_from or None,
        date_to=args.date_to or None,
        language=args.language.strip().lower(),
    )
    print(f"扫描文件数: {summary['files_scanned']}")
    print(f"输入新闻数: {summary['total']}")
    print(f"实验子集条数: {summary['kept']}")
    print(f"剔除条数: {summary['rejected']}")
    print(f"输出文件: {args.out}")
    print(f"各股票样本数: {summary['counts_by_stock']}")


def refine_thesis_set_main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m news_quant refine-thesis-set",
        description="对两股票候选新闻做二次强相关过滤，保留更适合论文实验的公司舆情新闻",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_2023_fullraw.jsonl",
        help="两股票候选新闻输入路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_2023_refined.jsonl",
        help="二次过滤后的论文实验集输出路径",
    )
    parser.add_argument(
        "--rejected-out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_2023_refined_rejected.jsonl",
        help="被二次过滤剔除的记录输出路径",
    )
    args = parser.parse_args()

    summary = refine_thesis_subset(
        input_path=args.input_path,
        output_path=args.out,
        rejected_path=args.rejected_out,
    )
    print(f"输入新闻数: {summary['total']}")
    print(f"保留新闻数: {summary['kept']}")
    print(f"剔除新闻数: {summary['rejected']}")
    print(f"输出文件: {args.out}")
    print(f"各股票样本数: {summary['counts_by_stock']}")


def select_thesis_llm_set_main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m news_quant select-thesis-llm-set",
        description="从二次过滤结果中选出最终用于论文 LLM baseline 的实验集",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_2023_refined.jsonl",
        help="二次过滤后的输入路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_2023_llmset.jsonl",
        help="最终论文实验集输出路径",
    )
    parser.add_argument(
        "--rejected-out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_2023_llmset_rejected.jsonl",
        help="未进入最终实验集的记录输出路径",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=16.0,
        help="最小相关性分数阈值",
    )
    parser.add_argument(
        "--max-per-stock",
        type=int,
        default=700,
        help="每只股票最多保留多少条",
    )
    args = parser.parse_args()

    summary = select_thesis_llm_set(
        input_path=args.input_path,
        output_path=args.out,
        rejected_path=args.rejected_out,
        min_score=args.min_score,
        max_per_stock=args.max_per_stock,
    )
    print(f"输入新闻数: {summary['total']}")
    print(f"最终实验集条数: {summary['kept']}")
    print(f"剔除条数: {summary['rejected']}")
    print(f"输出文件: {args.out}")
    print(f"各股票样本数: {summary['counts_by_stock']}")


def build_daily_topk_set_main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m news_quant build-daily-topk-set",
        description="从论文实验候选集中按日期和股票保留每天 topK 条相关新闻",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_2023_refined.jsonl",
        help="论文实验候选输入路径，建议使用 refined 数据集",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_daily_topk.jsonl",
        help="按日 topK 采样后的输出路径",
    )
    parser.add_argument(
        "--rejected-out",
        type=Path,
        default=ROOT / "data" / "prepared" / "opennewsarchive_thesis_2stocks_daily_topk_rejected.jsonl",
        help="同日未进入 topK 的记录输出路径",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每天每只股票最多保留多少条，允许少于 K 条",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default="2023-11-01",
        help="仅保留不早于该日期的新闻",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default="2023-11-30",
        help="仅保留不晚于该日期的新闻",
    )
    args = parser.parse_args()

    summary = build_daily_topk_per_stock(
        input_path=args.input_path,
        output_path=args.out,
        rejected_path=args.rejected_out,
        top_k=args.top_k,
        date_from=args.date_from or None,
        date_to=args.date_to or None,
    )
    print(f"输入新闻数: {summary['total']}")
    print(f"输出新闻数: {summary['kept']}")
    print(f"同日被截断条数: {summary['rejected']}")
    print(f"交易日-股票分组数: {summary['group_count']}")
    print(f"输出文件: {args.out}")
    print(f"各股票样本数: {summary['counts_by_stock']}")
    print(f"各股票覆盖天数: {summary['days_by_stock']}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "prepare-data":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        prepare_data_main()
    elif len(sys.argv) >= 2 and sys.argv[1] == "filter-data":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        filter_data_main()
    elif len(sys.argv) >= 2 and sys.argv[1] == "build-experiment-set":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        build_experiment_main()
    elif len(sys.argv) >= 2 and sys.argv[1] == "build-experiment-set-from-raw":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        build_experiment_from_raw_main()
    elif len(sys.argv) >= 2 and sys.argv[1] == "refine-thesis-set":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        refine_thesis_set_main()
    elif len(sys.argv) >= 2 and sys.argv[1] == "select-thesis-llm-set":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        select_thesis_llm_set_main()
    elif len(sys.argv) >= 2 and sys.argv[1] == "build-daily-topk-set":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        build_daily_topk_set_main()
    else:
        run_main()
