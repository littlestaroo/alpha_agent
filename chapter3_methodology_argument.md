# 第三章方法论证：基于多因子横截面排序的 top k 股票选择

## 1. 为什么将问题建模为排序而不是涨跌预测

本文的目标不是判断某一只股票明天一定上涨或下跌，而是在给定股票池中找出相对更值得关注的股票。因此，更合适的问题形式是横截面排序：

\[
\mathcal{S}_t=\{s_1,s_2,\dots,s_n\}
\]

给定交易日 \(t\) 的股票池 \(\mathcal{S}_t\)，以及每只股票对应的因子向量：

\[
\mathbf{F}_{s,t}=(f_{1,s,t},f_{2,s,t},\dots,f_{m,s,t})
\]

构造排序函数：

\[
Score_{s,t}=g(\mathbf{F}_{s,t})
\]

并选择得分最高的前 \(k\) 只股票：

\[
\mathcal{P}_{t}^{k}=TopK(\{Score_{s,t}:s\in\mathcal{S}_t\})
\]

这种建模方式有三个优点：

1. 更符合实际选股任务，投资者通常关心“股票池中谁更好”，而不是孤立预测单只股票；
2. 可以同时融合文本舆情因子和已有股票因子；
3. 评价时可以使用 top k 收益、相对股票池收益、top-bottom spread 和 Rank IC 等排序指标。

## 2. 因子输入

本文的因子输入分为两类。

第一类是文本另类数据因子，由新闻/舆情文本经 LLM 结构化后生成：

- `net_sentiment_factor`：净情绪因子；
- `negative_shock_factor`：负面冲击因子；
- `attention_factor`：关注度因子；
- `composite_score`：基础综合舆情分；
- `ema5_sentiment_state`：时序情绪状态；
- `negative_shock_carry`：负面冲击记忆；
- `state_composite_factor`：状态综合因子；
- `earnings_event_state`、`operations_event_state`、`market_buzz_event_state`、`risk_event_state`：事件状态因子。

第二类是已有股票相关因子，可包括但不限于：

- 动量因子；
- 波动率因子；
- 成交量或换手率因子；
- 估值因子；
- 行业相对收益因子；
- 基本面质量因子。

在当前项目实现中，文本因子已经完整打通；已有股票因子可以通过外部 JSON 配置加入排序模型。

## 3. 因子方向对齐

不同因子的经济含义不同。有些因子越大越好，例如净情绪、情绪状态、经营事件状态；有些因子越大越差，例如负面冲击、风险事件状态、波动率或风险暴露。

因此，本文为每个因子定义方向参数：

\[
d_f \in \{+1,-1\}
\]

方向对齐后：

\[
\tilde{x}_{s,t}^{f}=d_f x_{s,t}^{f}
\]

其中 \(d_f=+1\) 表示因子越大越好，\(d_f=-1\) 表示因子越小越好。经过方向对齐后，所有因子都满足“值越大，排序含义越好”。

## 4. 横截面标准化

由于不同因子的量纲不同，不能直接加权相加。例如情绪因子通常在 \([-1,1]\)，关注度因子在 \([0,1]\)，成交量类因子可能是很大的数值。

本文采用交易日内横截面百分位标准化：

\[
q_{s,t}^{f}=PctRank(\tilde{x}_{s,t}^{f})
\]

其中 \(q_{s,t}^{f}\in[0,1]\)。该方法的含义是：在同一天的股票池中，该股票在某个因子上处于什么相对位置。

这样做的优点是：

1. 不依赖正态分布假设；
2. 能处理不同量纲的因子；
3. 输出天然适合横截面排序；
4. 对极端值更稳健。

当某个交易日只有一只股票，或某因子在当天所有股票上取值完全相同，本文将该因子的百分位信号设为 \(0.5\)，表示中性贡献。

## 5. 加权排序模型

完成方向对齐和横截面标准化后，本文采用加权综合得到排序分数：

\[
Score_{s,t}=\frac{\sum_{f\in\mathcal{F}}w_f q_{s,t}^{f}}{\sum_{f\in\mathcal{F}}|w_f|}
\]

其中：

- \(w_f\)：第 \(f\) 个因子的权重；
- \(q_{s,t}^{f}\)：第 \(f\) 个因子的横截面百分位信号；
- \(Score_{s,t}\)：最终排序分数。

在当前第三章主模型中，权重分为三组。

| 因子组 | 含义 | 作用 |
| --- | --- | --- |
| 直接型因子 | 当日净情绪、负面冲击、关注度、基础综合分 | 捕捉当日新闻冲击 |
| 状态型因子 | 5日情绪状态、负面冲击记忆、状态综合分 | 降低单日噪声，刻画近期状态 |
| 事件状态因子 | 业绩、经营、市场舆情、风险事件状态 | 保留事件解释性，识别持续发酵主题 |

该模型体现的核心观点是：

> 单条新闻的情绪识别可以由 LLM 完成，但真正用于选股排序的信号，需要在股票层面经过时序累计和事件拆分。

## 6. top k 选择规则

对每个交易日 \(t\)，按 \(Score_{s,t}\) 降序排列：

\[
s_{(1),t},s_{(2),t},\dots,s_{(n),t}
\]

取前 \(k\) 只股票：

\[
\mathcal{P}_{t}^{k}=\{s_{(1),t},s_{(2),t},\dots,s_{(k),t}\}
\]

如果当天股票池数量小于 \(k\)，则选择当天全部股票。

## 7. 与 baseline 的递进关系

为了证明主模型的必要性，本文可以设计四个递进模型。

| 模型 | 因子输入 | 论证目的 |
| --- | --- | --- |
| M0：直接情绪模型 | 只使用 `net_sentiment_factor` | 作为最简单 baseline |
| M1：当日综合模型 | 加入负面冲击、关注度、基础综合分 | 验证多维当日舆情优于单一情绪 |
| M2：状态累计模型 | 加入情绪状态、负面冲击记忆 | 验证时序累计能降低噪声 |
| M3：事件状态模型 | 加入事件状态因子 | 验证事件拆分能提升解释性 |
| M4：融合模型 | 加入已有股票因子 | 验证文本另类数据的增量价值 |

当前项目中，`direct` preset 对应 M1，`state` preset 对应 M2，`event/all` preset 对应 M3；M4 可通过外部因子配置实现。

## 8. 方法有效性验证

排序模型的评价不宜只使用涨跌方向准确率，因为在单边市场中“总猜上涨”或“总猜下跌”可能获得较高准确率，但没有真正排序能力。

本文建议使用以下指标：

1. top k 平均未来收益：

\[
\overline{R}_{topk}^{(h)}
=\frac{1}{k}\sum_{s\in \mathcal{P}_{t}^{k}}R_{s,t}^{(h)}
\]

2. top k 相对股票池平均超额收益：

\[
Excess_{topk}^{(h)}
=\overline{R}_{topk}^{(h)}-\overline{R}_{universe}^{(h)}
\]

3. top-bottom spread：

\[
Spread^{(h)}
=\overline{R}_{topk}^{(h)}-\overline{R}_{bottomk}^{(h)}
\]

4. Rank IC：排序分数与未来收益的 Spearman 秩相关；
5. top k 命中率：top k 平均收益是否超过股票池平均收益。

其中 \(h\) 可取 1、3、5 个交易日。

## 9. 避免未来信息泄露

为了保证方法严谨，新闻日期与收益计算需要错位对齐。本文采用保守设定：

1. 新闻在自然日 \(t\) 发布；
2. 因子在下一交易日 \(\tau(t)\) 生效；
3. 未来收益从 \(\tau(t)\) 的收盘价开始计算。

这样可以避免使用新闻发布后同日无法实际交易的信息。

## 10. 工程实现对应关系

当前项目已经将排序模型实现为可复用模块：

- `news_quant/ranking.py`：核心排序模型；
- `analysis/build_stock_rankings.py`：命令行脚本；
- `tests/test_ranking_model.py`：排序模型单元测试。

核心命令示例：

```bash
python analysis/build_stock_rankings.py \
  --in output/q4_20stocks_batches/chapter3_baseline_q4_20stocks_batches_000_004_diverse_top3_daily_profiles_global.csv \
  --out output/q4_20stocks_batches/chapter3_ranked.csv \
  --topk-out output/q4_20stocks_batches/chapter3_top5.csv \
  --summary-out output/q4_20stocks_batches/chapter3_rank_summary.json \
  --top-k 5 \
  --preset event
```

若要加入已有股票因子，可提供 JSON 配置：

```json
{
  "factors": [
    {"name": "net_sentiment_factor", "label": "净情绪", "group": "text", "weight": 0.2, "direction": 1},
    {"name": "operations_event_state", "label": "经营事件状态", "group": "text", "weight": 0.3, "direction": 1},
    {"name": "momentum_20d", "label": "20日动量", "group": "market", "weight": 0.3, "direction": 1},
    {"name": "volatility_20d", "label": "20日波动率", "group": "market", "weight": 0.2, "direction": -1}
  ]
}
```

## 11. 可写入论文的总结表述

本文将基于情绪类文本另类数据的因子挖掘进一步建模为股票横截面排序问题。具体而言，首先利用大语言模型对新闻文本进行事件识别和情绪数值化，再在股票-日期层面构造直接型因子、时序状态因子和事件状态因子。随后，对所有因子进行方向对齐和横截面百分位标准化，并通过加权融合得到股票排序分数。最后，按照排序分数选取每日 top k 股票，形成从文本另类数据到股票选择结果的完整方法链路。该方法既保留了 LLM 在单条文本理解上的优势，又通过时序累计和事件拆分提升了因子的稳定性与解释性。
