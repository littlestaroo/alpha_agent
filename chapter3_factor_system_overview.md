# 第三章从数值化参数到因子体系总表

## 1. 总体链路

本文第三章中，新闻舆情数据不是直接拿来做涨跌预测，而是先经过大语言模型完成单条新闻的结构化数值化，再按“股票-日期”进行聚合，最终形成股票级日频因子。其整体链路可以概括为：

**新闻文本 -> 单条新闻数值化 -> 新闻权重计算 -> 股票日频聚合 -> 基础因子 -> 时序累计因子 -> 事件状态因子 -> 综合因子**

需要特别说明的是：

- 单条新闻的情绪与事件参数还不属于因子，它们只是因子生成的前置输入。
- 只有在按“股票-日期”聚合后得到的日频结果，才属于本文真正意义上的股票舆情因子。
- 本文已经实现了时序累计因子，即考虑了新闻舆情在时间上的持续影响，而不是只使用当天的新闻值。

## 2. 新闻级数值化参数

这一层解决的问题是：**一条新闻如何被 LLM 理解并转化为可计算的结构化字段。**

| 参数名 | 含义 | 取值范围/方向 | 作用 |
| --- | --- | --- | --- |
| `sentiment` | 情绪方向值 | `[-1, 1]`，越大越正面 | 反映新闻整体偏利好还是偏利空 |
| `sentiment_strength` | 情绪强度 | `[0, 1]`，越大越强烈 | 区分“轻微偏正面”和“强烈利好/利空” |
| `confidence` | 模型把握度 | `[0, 1]` | 表示 LLM 对判断结果的信心 |
| `relevance` | 与股票的相关度 | `[0, 1]` | 判断这条新闻是否真正在讲该股票 |
| `event_importance` | 事件重要度 | `[0, 1]` | 反映该事件对市场预期和股价可能的影响程度 |
| `risk_flag` | 风险标记 | `0/1` | 是否含有明显风险提示 |
| `event_type` | 事件类型 | 类别型 | 将新闻划分为业绩、经营、投融资、风险、市场舆情等类型 |

这一层的输出是“新闻级参数”，它们会作为后续聚合和加权的输入。

## 3. 新闻权重的形成

为了避免所有新闻在聚合时“权重相同”，本文对每条新闻先计算权重：

$$
w_i = relevance_i \times confidence_i \times event\_importance_i \times (0.5 + 0.5 \times sentiment\_strength_i) \times event\_type\_weight_i
$$

其经济含义是：

- 与股票越相关的新闻，权重越高；
- 模型越有把握的新闻，权重越高；
- 事件越重要的新闻，权重越高；
- 情绪越强烈的新闻，权重越高；
- 业绩、风险等更重要事件类型，会给予更高类别权重。

因此，本文后续形成的因子不是简单平均，而是**加权聚合结果**。

## 4. 基础日频因子

这一层解决的问题是：**同一天关于同一只股票的多条新闻，如何合并成日频股票因子。**

| 因子名 | 中文名称 | 形成方式 | 经济含义 |
| --- | --- | --- | --- |
| `net_sentiment_factor` | 净情绪因子 | 对单日新闻情绪按权重加权平均 | 反映当天整体舆情偏正面还是偏负面 |
| `negative_shock_factor` | 负面冲击因子 | 只提取负面部分并按权重聚合 | 反映当天利空或风险压力有多强 |
| `attention_factor` | 关注度因子 | 基于新闻数量与相关度形成 | 反映当天市场对该股票的关注热度 |
| `sentiment_dispersion_factor` | 情绪分歧因子 | 统计同日新闻情绪分散程度 | 反映市场观点是否一致 |
| `event_density_factor` | 事件密度因子 | 基于当天事件数量形成 | 反映当天新闻事件是否集中爆发 |
| `risk_ratio` | 风险占比 | 统计风险新闻占比 | 反映风险类新闻在当天舆情中的比例 |
| `positive_ratio` | 正面新闻占比 | 统计正面新闻比例 | 辅助刻画舆情结构 |
| `neutral_ratio` | 中性新闻占比 | 统计中性新闻比例 | 辅助刻画舆情结构 |
| `negative_ratio` | 负面新闻占比 | 统计负面新闻比例 | 辅助刻画舆情结构 |

其中，第三章最核心的基础因子是：

- `net_sentiment_factor`
- `negative_shock_factor`
- `attention_factor`

## 5. 时间变化因子

这一层解决的问题是：**新闻舆情不是静态的，情绪是否在变化、改善或恶化。**

| 因子名 | 中文名称 | 形成方式 | 经济含义 |
| --- | --- | --- | --- |
| `sentiment_delta_1d` | 单日情绪变化量 | 当日净情绪减去前一日净情绪 | 反映相邻两天舆情变化幅度 |
| `sentiment_trend_factor` | 情绪趋势因子 | 当前净情绪相对数日前的变化 | 反映情绪变化方向 |
| `sentiment_momentum_factor` | 情绪动量因子 | 当前净情绪相对近期均值的偏离 | 反映情绪是否持续上行或下行 |

这一层仍然属于“日频动态因子”，但还不是严格意义上的时序累计因子。

## 6. 时序累计因子

这一层是你当前最关心的部分，解决的问题是：**新闻影响是否会在时间上延续，而不是只停留在当天。**

答案是：**有，而且已经实现了。**

| 因子名 | 中文名称 | 形成方式 | 经济含义 |
| --- | --- | --- | --- |
| `ema3_sentiment_state` | 3日情绪状态 | 对净情绪做 3 日指数加权平均 | 反映最近 3 天的累计情绪状态 |
| `ema5_sentiment_state` | 5日情绪状态 | 对净情绪做 5 日指数加权平均 | 反映最近 5 天更平滑的累计情绪状态 |
| `negative_shock_carry` | 负面冲击记忆 | 对负面冲击做衰减累计 | 反映近期利空压力是否持续存在 |
| `event_novelty_factor` | 事件新颖性因子 | 当前值减去过去 5 日均值 | 反映今天是否出现了“新冲击” |

其中：

- `ema3_sentiment_state` 和 `ema5_sentiment_state` 是典型的**时序累计因子**；
- `negative_shock_carry` 是典型的**负面影响持续性因子**；
- `event_novelty_factor` 用于衡量“新信息冲击”，不是简单累计，而是相对历史均值的偏离。

因此，本文并不只是使用“当天舆情值”，而是已经把新闻影响的时间持续性考虑进去了。

## 7. 事件因子

这一层解决的问题是：**不同类型的新闻不能全部压缩成一个总情绪值，而要区分是哪类事件在驱动舆情。**

| 因子名 | 中文名称 | 形成方式 | 经济含义 |
| --- | --- | --- | --- |
| `earnings_event_factor` | 业绩事件因子 | 对业绩类新闻情绪按权重聚合 | 反映业绩类信息的舆情方向 |
| `operations_event_factor` | 经营事件因子 | 对经营类新闻情绪按权重聚合 | 反映经营变化对舆情的影响 |
| `financing_event_factor` | 投融资事件因子 | 对投融资类新闻情绪按权重聚合 | 反映资本运作信息的市场态度 |
| `market_buzz_event_factor` | 市场舆情事件因子 | 对市场舆情类新闻情绪按权重聚合 | 反映市场讨论和热点情绪 |
| `risk_event_factor` | 风险事件因子 | 对风险类新闻情绪按权重聚合 | 反映处罚、调查、承压等风险事件影响 |

这些因子属于**当日事件拆分因子**，重点回答“当天到底是哪类事件在推动舆情变化”。

## 8. 事件状态因子

这一层解决的问题是：**某一类事件是否在最近几天持续发酵。**

| 因子名 | 中文名称 | 形成方式 | 经济含义 |
| --- | --- | --- | --- |
| `earnings_event_state` | 业绩事件状态 | 对业绩事件因子做 5 日指数加权平均 | 反映业绩信息是否持续发酵 |
| `operations_event_state` | 经营事件状态 | 对经营事件因子做 5 日指数加权平均 | 反映经营类信息是否持续影响市场 |
| `financing_event_state` | 投融资事件状态 | 对投融资事件因子做 5 日指数加权平均 | 反映资本运作事件的持续影响 |
| `market_buzz_event_state` | 市场舆情事件状态 | 对市场舆情事件因子做 5 日指数加权平均 | 反映市场讨论热度是否持续 |
| `risk_event_state` | 风险事件状态 | 对风险事件因子做 5 日指数加权平均 | 反映风险舆情是否持续升温 |

这类因子是本文目前最有研究价值的一类，因为它兼顾了：

- 事件解释性；
- 时间持续性；
- 股票级聚合。

例如，`operations_event_state` 就表示：最近几天经营类信息是持续改善，还是持续恶化。

## 9. 综合因子

这一层解决的问题是：**如何将多个单项因子进一步压缩为一个总分，用于排序、展示和综合分析。**

| 因子名 | 中文名称 | 形成方式 | 经济含义 |
| --- | --- | --- | --- |
| `composite_score` | 基础综合分 | 基于净情绪、负面冲击、关注度、情绪动量、分歧等线性组合 | 反映当日股票舆情的综合表现 |
| `state_composite_factor` | 状态综合分 | 基于情绪状态、负面记忆、事件状态等线性组合 | 反映最近几天股票舆情的综合状态 |

两者的区别是：

- `composite_score` 更偏“当天值”
- `state_composite_factor` 更偏“时间累计后的状态值”

## 10. 当前因子体系的结构总结

从“数值化参数”到“因子体系”的完整结构可以概括为：

1. **新闻级参数**  
   `sentiment`、`sentiment_strength`、`confidence`、`relevance`、`event_importance`、`risk_flag`

2. **基础日频因子**  
   `net_sentiment_factor`、`negative_shock_factor`、`attention_factor`

3. **时间变化因子**  
   `sentiment_delta_1d`、`sentiment_trend_factor`、`sentiment_momentum_factor`

4. **时序累计因子**  
   `ema3_sentiment_state`、`ema5_sentiment_state`、`negative_shock_carry`、`event_novelty_factor`

5. **事件因子**  
   `earnings_event_factor`、`operations_event_factor`、`financing_event_factor`、`market_buzz_event_factor`、`risk_event_factor`

6. **事件状态因子**  
   `earnings_event_state`、`operations_event_state`、`financing_event_state`、`market_buzz_event_state`、`risk_event_state`

7. **综合因子**  
   `composite_score`、`state_composite_factor`

## 11. 最适合向老师重点展示的因子

如果第三章汇报不想展开过多细节，建议重点讲三组：

### 1. 基础因子

- `net_sentiment_factor`
- `negative_shock_factor`
- `attention_factor`

### 2. 时序累计因子

- `ema3_sentiment_state`
- `ema5_sentiment_state`
- `negative_shock_carry`

### 3. 事件状态因子

- `earnings_event_state`
- `operations_event_state`
- `risk_event_state`

这样可以非常清晰地说明：

- 本文不是只做单条新闻情绪判断；
- 也不是只用当天舆情值；
- 而是已经形成了“当天值 + 时序累计状态 + 事件状态”的完整因子体系。
