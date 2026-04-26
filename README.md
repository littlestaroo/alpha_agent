# 第三章 Baseline：基于情绪类文本另类数据的股票舆情因子挖掘

这个工程是按你论文第三章的思路做的一个最小可运行 baseline，重点打通下面这条链路：

`OpenNewsArchive 风格新闻 -> 股票对齐 -> LLM 情绪/事件结构化 -> 股票级日频舆情画像 -> 趋势因子输出`

它不追求复杂模型，而是作为第四章智能体方案之前的对比基线，便于你先完成第二周和第三章前半部分实验。

当前默认运行入口已经切到“实验子集”：

- 原始规范化数据：`data/prepared/opennewsarchive_news.jsonl`
- 严格筛选数据：`data/prepared/opennewsarchive_stock_news_strict.jsonl`
- baseline 默认输入：`data/prepared/opennewsarchive_experiment_set.jsonl`

## 1. 方法定位

本 baseline 对应你报告中的“基于大语言模型 API 的文本情绪识别”方案，但输出不只停留在正负面分类，而是同时给出：

- 股票对象
- 事件类别
- 情绪得分
- 情绪强度
- 事件重要度
- 置信度
- 风险标记
- 趋势信号

之后将文本级结果聚合为股票级日频画像，并构造三个最核心的第三章因子：

- `net_sentiment_factor`：净情绪因子
- `negative_shock_factor`：负面冲击因子
- `sentiment_momentum_factor`：情绪动量因子

同时保留一些扩展字段，便于后续写论文或继续增强：

- `attention_factor`
- `sentiment_dispersion_factor`
- `event_density_factor`
- `composite_score`
- `profile_label`

## 2. 数据格式

### 推荐目录结构

```text
data/
  raw/
    opennewsarchive/      # 放原始下载文件
  prepared/
    opennewsarchive_news.jsonl
```

建议先把 `OpenNewsArchive` 的原始文件放到 `data/raw/opennewsarchive/`，再运行一次规范化准备命令，把它统一整理成工程内部使用的标准 JSONL。

### 新闻输入

支持以下格式：

- `csv`
- `jsonl`
- `parquet`
- 目录（递归读取目录下的上述文件）

程序会自动识别常见字段名，并统一映射为：

- `id`
- `title`
- `body`
- `publish_time`
- `publish_date`
- `source`
- `url`

可兼容的典型新闻字段包括：

- `headline/title/news_title`
- `content/body/text/full_text/description`
- `published_at/publish_time/date/datetime`

如果原始 JSON/JSONL 中存在嵌套结构，例如 `article.title`、`data.content` 这类字段，准备脚本也会尝试自动展开并提取。

### 股票池输入

股票池 CSV 至少需要：

- `ts_code`
- `name`

建议同时提供：

- `industry`
- `aliases`

其中 `aliases` 用 `|` 分隔，例如：

```csv
ts_code,name,industry,aliases
300750.SZ,宁德时代,电力设备,宁德时代|CATL|宁王
601398.SH,工商银行,银行,工商银行|工行
```

## 3. 快速开始

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 先配置数据集

如果你已经从 `OpenNewsArchive` 下载了原始文件，把它们放进：

```bash
data/raw/opennewsarchive/
```

然后执行：

```bash
python -m news_quant prepare-data
```

或显式指定路径：

```bash
python -m news_quant prepare-data \
  --raw data/raw/opennewsarchive \
  --out data/prepared/opennewsarchive_news.jsonl
```

这一步会把原始 `csv/json/jsonl/parquet` 文件规范化为统一字段：

- `id`
- `title`
- `body`
- `publish_time`
- `publish_date`
- `source`
- `url`

后续 baseline 默认就直接读取这个规范化后的文件。

### 直接跑样例

```bash
python -m news_quant
```

这条命令默认会读取：

```bash
data/prepared/opennewsarchive_experiment_set.jsonl
```

或显式指定参数：

```bash
python -m news_quant \
  --news data/prepared/opennewsarchive_experiment_set.jsonl \
  --universe data/stock_universe_sample.csv \
  --out output/chapter3_baseline
```

### 用真实 API 跑

1. 复制 `.env.example` 为 `.env`
2. 填写：

```bash
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=可选，兼容 OpenAI 的中转或国产模型服务地址
OPENAI_MODEL=gpt-4o-mini
MOCK_LLM=0
```

### 常用过滤参数

```bash
python -m news_quant \
  --news data/prepared/opennewsarchive_news.jsonl \
  --universe data/stock_universe_sample.csv \
  --keyword 银行,电池,业绩 \
  --date-from 2025-03-01 \
  --date-to 2025-03-31 \
  --limit 200
```

## 4. 输出文件

以 `--out output/chapter3_baseline` 为例，会生成：

- `output/chapter3_baseline_article_mentions.csv`
- `output/chapter3_baseline_daily_profiles.csv`
- `output/chapter3_baseline_skipped_news.csv`

其中：

- `article_mentions` 是文本级结构化结果
- `daily_profiles` 是股票级日频画像与因子表
- `skipped_news` 记录了未进入后续分析的新闻及原因

## 5. 因子解释

### 净情绪因子

对某股票某日的全部相关新闻，按“相关度 × 置信度 × 事件重要度 × 情绪强度”加权聚合：

`net_sentiment_factor = weighted_mean(sentiment)`

### 负面冲击因子

只关注负面部分，用于刻画风险暴露强度：

`negative_shock_factor = weighted_mean(max(-sentiment, 0))`

### 情绪动量因子

衡量当前日情绪相对过去几日均值的变化：

`sentiment_momentum_factor = current_net_sentiment - previous_3day_mean`

## 6. 与论文第三章的对应关系

你可以直接把这套 baseline 写成第三章前半部分的“对照算法”：

1. 文本与股票对齐：基于股票名称/别名匹配相关新闻
2. 文本结构化分析：调用 LLM 输出事件类别与情绪信息
3. 股票级舆情画像：按股票和日期聚合文本级结果
4. 因子构造：生成净情绪、负面冲击、情绪动量等因子
5. 趋势分析：观察股票舆情画像与因子的时间变化

后面你再做第四章智能体时，可以把这套 baseline 作为“无智能体、直接 LLM API”的比较对象。

## 7. 排序模型与 top k 选股

项目当前已经在第三章 baseline 基础上实现了横截面排序模型，可将股票日频因子进一步映射为每日 `top k` 股票。

核心模块：

- `news_quant/ranking.py`
- `analysis/build_stock_rankings.py`
- `analysis/compare_ranking_presets.py`

支持的 preset 包括：

- `direct`：只使用当天直接型因子
- `state`：加入时序状态因子
- `event`：加入事件状态因子
- `optimized`：加入可靠性优化后的增强排序模型

示例：

```bash
.venv_news/bin/python analysis/build_stock_rankings.py \
  --in output/q4_20stocks_batches/chapter3_baseline_q4_20stocks_batches_000_004_diverse_top3_daily_profiles_global.csv \
  --out /tmp/chapter3_ranked.csv \
  --topk-out /tmp/chapter3_top5.csv \
  --summary-out /tmp/chapter3_rank_summary.json \
  --top-k 5 \
  --preset optimized
```

## 8. 第四章多智能体工作流

项目已经实现第四章对应的多智能体工作流，将第三章方法组织为：

`数据输入 agent -> 舆情分析 agent -> 因子构建 agent -> 排序选股 agent -> 评估报告 agent`

核心模块：

- `news_quant/agent_pipeline.py`
- `analysis/run_agent_pipeline.py`

示例：

```bash
MOCK_LLM=1 .venv_news/bin/python analysis/run_agent_pipeline.py \
  --news data/prepared/opennewsarchive_thesis_2stocks_2023-11_daily1_perstock.jsonl \
  --universe data/stock_universe_thesis_2stocks.csv \
  --out-dir /tmp/chapter4_agent_demo \
  --ranking-preset optimized \
  --top-k 1 \
  --price-cache data/market/chapter3_2stocks_prices_20231020_20231231.csv
```

输出包括：

- 文章级结构化结果
- 股票级日频因子
- 排序结果与 top k 股票
- 收益评估结果
- Markdown 系统运行报告

## 9. 论文文档入口

当前仓库中可直接用于论文写作的核心文档包括：

- `chapter3_proposed_factor_ranking_model.md`
- `chapter3_methodology_argument.md`
- `chapter3_ranking_validation_report.md`
- `chapter4_agent_design_and_implementation.md`
- `thesis_writing_blueprint.md`
