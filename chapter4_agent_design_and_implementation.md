# 第四章 面向另类数据挖掘因子的智能体设计与实现

## 4.1 本章定位

第三章解决的是“模型怎么设计”的问题，即如何将新闻/舆情文本转化为股票级因子，并进一步组织成横截面排序模型。第四章解决的是“系统怎么实现”的问题，即如何将数据输入、文本分析、因子构建、排序选股和报告输出封装为一条可自动执行的多智能体工作流。

因此，第四章的目标不是重新提出新的因子模型，而是围绕第三章的方法，将其实现为一个具备自动化、模块化和可扩展性的智能体系统。

## 4.2 系统目标

本系统围绕“新闻/舆情文本 -> 因子 -> 股票排序 -> 报告输出”的链路设计，目标包括：

1. 自动读取原始或预处理后的新闻数据；
2. 自动完成新闻与股票对象匹配、事件识别和情绪结构化；
3. 自动生成股票级日频舆情因子；
4. 自动调用排序模型完成 top k 股票选择；
5. 自动输出评估结果和文字化报告；
6. 支持后续扩展更多因子、更多股票池和更多市场因子。

## 4.3 多智能体架构设计

本文采用“功能分工型多智能体”设计，将整个流程拆分为若干职责清晰的 agent。

### 4.3.1 数据输入 Agent

数据输入 agent 负责读取新闻文件和股票池文件，并对输入参数进行统一管理。其主要职责包括：

- 加载新闻文件或目录；
- 加载股票池 CSV；
- 读取日期范围、样本条数、排序 preset 等运行参数；
- 组织整个工作流的输入上下文。

### 4.3.2 舆情分析 Agent

舆情分析 agent 负责完成单条新闻的股票候选匹配和 LLM 结构化分析。主要输出为文章级结构化结果，字段包括：

- `sentiment`
- `sentiment_strength`
- `confidence`
- `relevance`
- `event_importance`
- `risk_flag`
- `event_type`

这一 agent 对应第三章中的“新闻级数值化”步骤。

### 4.3.3 因子构建 Agent

因子构建 agent 负责将文章级结构化结果聚合为股票-日期级因子，包括：

- 直接型因子；
- 时序状态因子；
- 事件状态因子；
- 可靠性增强因子。

这一 agent 的输出构成第三章排序模型的主要输入。

### 4.3.4 排序选股 Agent

排序选股 agent 负责完成以下任务：

- 因子方向对齐；
- 横截面百分位标准化；
- 加权打分；
- 可靠性收缩；
- 每日 top k 股票选择。

该 agent 对应第三章中的横截面排序模型。

### 4.3.5 评估报告 Agent

评估报告 agent 负责将排序结果与未来收益对齐，计算：

- top k 平均收益；
- 相对股票池超额收益；
- top-bottom spread；
- 命中率；
- 文字化运行报告。

这一 agent 将算法输出转化为论文展示和系统展示都易于使用的结果。

## 4.4 系统工作流

系统工作流可概括为：

```text
数据输入 Agent
  -> 舆情分析 Agent
  -> 因子构建 Agent
  -> 排序选股 Agent
  -> 评估报告 Agent
```

对应到实现层，可以进一步写为：

```text
新闻文件 / 股票池
  -> run_baseline()
  -> build_daily_profiles()
  -> build_rankings()
  -> evaluate_topk()
  -> Markdown / JSON 报告
```

## 4.5 关键实现模块

### 4.5.1 Agent 工作流核心模块

本系统已实现统一工作流模块：

[agent_pipeline.py](/Users/star/Desktop/agent/news_quant/agent_pipeline.py)

该模块以 `AgentPipelineConfig` 统一管理运行参数，并通过 `run_agent_pipeline()` 顺序调度各个 agent。

### 4.5.2 第四章工作流入口脚本

本系统已实现第四章的独立运行脚本：

[run_agent_pipeline.py](/Users/star/Desktop/agent/analysis/run_agent_pipeline.py)

也已接入主 CLI：

```bash
python -m news_quant run-agent-pipeline ...
```

### 4.5.3 排序模型复用

排序 agent 复用了第三章的核心模块：

[ranking.py](/Users/star/Desktop/agent/news_quant/ranking.py)

其中支持：

- `direct`
- `state`
- `event`
- `optimized`

等多种排序 preset，便于第四章展示“系统可切换不同策略配置”。

## 4.6 系统输入输出设计

### 4.6.1 输入

系统的主要输入包括：

- 新闻文件路径；
- 股票池路径；
- 排序模型 preset；
- top k 参数；
- 日期范围参数；
- 最大处理条数和 LLM 输入限制。

### 4.6.2 输出

系统输出包括：

- `article_mentions.csv`：文章级结构化结果；
- `daily_profiles.csv`：股票级日频因子；
- `agent_ranked.csv`：完整排序结果；
- `agent_topk.csv`：每日 top k 股票；
- `agent_topk_performance.csv`：收益评估结果；
- `agent_workflow_report.md`：系统运行报告；
- `agent_workflow_summary.json`：系统摘要。

## 4.7 运行示例

第四章系统已支持如下运行方式：

```bash
.venv_news/bin/python analysis/run_agent_pipeline.py \
  --news data/prepared/opennewsarchive_thesis_20stocks_2023Q4_top10_perstock.jsonl \
  --universe data/stock_universe_thesis_20stocks.csv \
  --out-dir output/chapter4_agent_demo \
  --ranking-preset optimized \
  --top-k 5
```

或者：

```bash
.venv_news/bin/python -m news_quant run-agent-pipeline \
  --news data/prepared/opennewsarchive_thesis_20stocks_2023Q4_top10_perstock.jsonl \
  --universe data/stock_universe_thesis_20stocks.csv \
  --out-dir output/chapter4_agent_demo \
  --ranking-preset optimized \
  --top-k 5
```

## 4.8 系统特点

该多智能体系统具备以下特点：

1. **模块解耦**  
   新闻分析、因子构建、排序和评估各自独立，便于替换和扩展。

2. **可复用性强**  
   第三章的因子模型可直接被第四章系统调用，不需要重复实现。

3. **支持多策略切换**  
   系统可以切换不同 ranking preset，便于对比不同模型。

4. **结果可追踪**  
   每一步输出都保留 CSV、JSON 和 Markdown 报告，适合论文展示和调试。

5. **便于后续扩展**  
   后续可以继续接入传统股票因子、来源质量因子、事件聚类模块和更复杂的权重学习模块。

## 4.9 与论文整体结构的衔接

第四章与第三章之间的关系可以概括为：

- 第三章：提出方法、构造因子、设计排序模型；
- 第四章：将该方法实现为多智能体工作流系统，并验证系统能够自动完成从新闻到 top k 股票选择的全过程。

因此，第四章的价值不在于再次证明因子有效性，而在于说明：

> 本文不仅提出了基于文本另类数据的股票排序模型，而且已经将其实现为一个可自动执行、可复用和可扩展的智能体系统。

## 4.10 本章小结

本章围绕第三章提出的文本另类数据因子排序方法，设计并实现了一套面向股票排序任务的多智能体工作流系统。系统通过数据输入 agent、舆情分析 agent、因子构建 agent、排序选股 agent 和评估报告 agent 的分工协作，实现了从新闻文本到 top k 股票选择结果的自动化处理。该系统不仅验证了第三章方法在工程上的可落地性，也为后续扩展为更完整的投研辅助系统提供了实现基础。
