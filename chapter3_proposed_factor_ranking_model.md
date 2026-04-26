# 第三章主模型设计：基于事件-情绪状态因子的股票排序模型

## 1. 研究问题建模

本文第三章的核心问题可以定义为：

> 给定股票池、新闻/舆情文本以及已有市场因子，如何将文本另类数据转化为可计算的情绪事件因子，并据此对股票进行横截面排序，选出得分最高的 top k 只股票。

设交易日为 \(t\)，股票池为：

\[
\mathcal{S}_t=\{s_1,s_2,\dots,s_n\}
\]

对每只股票 \(s\)，本文希望从新闻文本中构造舆情因子向量：

\[
\mathbf{F}_{s,t}^{text}
\]

并与已有股票因子向量：

\[
\mathbf{F}_{s,t}^{base}
\]

共同输入排序函数：

\[
Score_{s,t}=g(\mathbf{F}_{s,t}^{text},\mathbf{F}_{s,t}^{base})
\]

最终按 \(Score_{s,t}\) 从高到低排序，选择前 \(k\) 只股票：

\[
\mathcal{P}_{t}^{k}=TopK(\{Score_{s,t}:s\in \mathcal{S}_t\})
\]

因此，本文不是直接预测单只股票涨跌，而是将任务建模为一个更适合多股票场景的横截面排序问题。

## 2. Baseline 模型：直接 LLM 情绪因子排序

为了形成对比基准，本文首先构建一个最简单的 baseline：

1. 使用 LLM API 对单条新闻进行股票相关性判断、事件识别和情绪评分；
2. 将单条新闻输出为结构化字段，包括情绪得分、情绪强度、相关度、置信度、事件重要度和风险标记；
3. 按“股票-日期”聚合，形成日频净情绪因子；
4. 直接使用当日舆情因子或基础综合分进行股票排序。

baseline 的典型形式为：

\[
Score_{s,t}^{base}=NetSent_{s,t}
\]

或：

\[
Score_{s,t}^{base}
=0.45NetSent_{s,t}
-0.25NegShock_{s,t}
+0.15Attention_{s,t}
+0.15Momentum_{s,t}
-0.10Dispersion_{s,t}
\]

该方法的优点是链路清晰、实现简单、解释性强，适合作为论文中的对比算法。但它的不足也很明显：只使用当天舆情值时，容易受到单日新闻噪声影响；同时，将所有新闻压缩为一个总情绪值，会丢失事件类型差异。

## 3. 主模型：事件-情绪状态因子排序模型

基于 baseline 的不足，本文设计主模型为：

> 事件-情绪状态因子排序模型，即 Event-Sentiment State Factor Ranking Model。

模型核心思想是：

1. 单条新闻层面继续使用 LLM 完成文本理解；
2. 股票日频层面不直接使用单日情绪值，而是构造“直接因子 + 时序状态因子 + 事件状态因子”；
3. 排序层面对不同因子做方向对齐、横截面标准化和加权融合；
4. 输出每日 top k 股票。

该模型可以写成三层结构：

```text
新闻文本
  -> LLM 事件情绪结构化
  -> 股票日频基础因子
  -> 时序累计与事件状态因子
  -> 横截面排序分数
  -> top k 股票选择
```

## 4. 单条新闻数值化层

对新闻 \(i\)，LLM 输出以下结构化参数：

- `sentiment_i`：情绪方向，取值 \([-1,1]\)
- `sentiment_strength_i`：情绪强度，取值 \([0,1]\)
- `relevance_i`：与股票相关度，取值 \([0,1]\)
- `confidence_i`：模型置信度，取值 \([0,1]\)
- `event_importance_i`：事件重要度，取值 \([0,1]\)
- `event_type_i`：事件类型
- `risk_flag_i`：风险标记

单条新闻权重定义为：

\[
w_i = relevance_i \times confidence_i \times event\_importance_i
\times (0.5+0.5sentiment\_strength_i)
\times \omega(event\_type_i)
\]

其中 \(\omega(event\_type_i)\) 表示事件类型权重。这样可以避免所有新闻等权进入聚合。

## 5. 股票日频基础因子层

对股票 \(s\) 在日期 \(t\) 的相关新闻集合 \(N_{s,t}\)，构造基础因子。

净情绪因子：

\[
NetSent_{s,t}
=\frac{\sum_{i\in N_{s,t}} sentiment_i w_i}{\sum_{i\in N_{s,t}} w_i}
\]

负面冲击因子：

\[
NegShock_{s,t}
=\frac{\sum_{i\in N_{s,t}} \max(-sentiment_i,0)w_i}{\sum_{i\in N_{s,t}} w_i}
\]

关注度因子：

\[
Attention_{s,t}
=\tanh\left(
\frac{\ln(1+ArticleCount_{s,t}+MentionCount_{s,t}/2)\times \overline{relevance}_{s,t}}{2}
\right)
\]

基础综合因子：

\[
Composite_{s,t}
=0.45NetSent_{s,t}
-0.25NegShock_{s,t}
+0.15Attention_{s,t}
+0.15Momentum_{s,t}
-0.10Dispersion_{s,t}
\]

这一层对应 baseline 的主要输出，回答“今天这只股票的舆情怎么样”。

## 6. 时序状态因子层

直接使用当天舆情值容易受到单日噪声影响，因此本文进一步引入时序累计。

情绪状态因子：

\[
SentState_{s,t}^{(h)}
=\alpha_h NetSent_{s,t}
+(1-\alpha_h)SentState_{s,t-1}^{(h)}
\]

其中 \(h\) 可以取 3 日或 5 日窗口，\(\alpha_h=2/(h+1)\)。

负面冲击记忆因子：

\[
NegCarry_{s,t}
=\beta NegCarry_{s,t-1}+NegShock_{s,t}
\]

事件新颖性因子：

\[
Novelty_{s,t}
=Composite_{s,t}
-\frac{1}{m}\sum_{j=1}^{m}Composite_{s,t-j}
\]

这一层回答“近期舆情状态是否持续改善或恶化”。

## 7. 事件状态因子层

不同事件类型的市场含义不同，因此本文不只构造总情绪因子，还按事件类型拆分。

设事件类型集合为：

\[
K=\{业绩,经营,投融资,风险,市场舆情\}
\]

某类事件的当日情绪因子为：

\[
EventFactor_{s,t}^{k}
=\frac{\sum_{i\in N_{s,t},event_i=k} sentiment_i w_i}{\sum_{i\in N_{s,t},event_i=k} w_i}
\]

进一步构造事件状态因子：

\[
EventState_{s,t}^{k}
=\lambda EventState_{s,t-1}^{k}
+(1-\lambda)EventFactor_{s,t}^{k}
\]

例如：

- `earnings_event_state`：业绩事件状态
- `operations_event_state`：经营事件状态
- `financing_event_state`：投融资事件状态
- `market_buzz_event_state`：市场舆情事件状态
- `risk_event_state`：风险事件状态

这一层回答“哪一类事件正在持续发酵”。

## 8. 横截面排序模型

由于不同因子的量纲和方向不同，排序前需要做方向对齐与横截面标准化。

对因子 \(f\)，定义方向参数：

\[
d_f\in\{+1,-1\}
\]

其中正向因子取 \(+1\)，风险或负面冲击类因子取 \(-1\)。

方向对齐：

\[
\tilde{x}_{s,t}^{f}=d_f x_{s,t}^{f}
\]

在同一交易日内做百分位标准化：

\[
q_{s,t}^{f}=PctRank(\tilde{x}_{s,t}^{f})
\]

最终排序分数为：

\[
Score_{s,t}
=\sum_{f\in \mathcal{F}} \theta_f q_{s,t}^{f}
\]

其中 \(\theta_f\) 为因子权重。当前建议采用三组因子加权：

| 因子组 | 代表因子 | 建议权重 |
| --- | --- | --- |
| 直接型因子 | 净情绪、负面冲击、关注度、基础综合分 | 0.25-0.30 |
| 状态型因子 | 5日情绪状态、负面冲击记忆、状态综合分 | 0.35-0.40 |
| 事件状态因子 | 业绩、经营、市场舆情、风险事件状态 | 0.30-0.40 |

这种设计体现本文主张：

> 单条新闻情绪识别适合交给 LLM，但真正用于决策排序的信号，应来自股票级时序累计和事件状态因子。

## 9. 与已有股票因子的融合

若需要进一步结合“现有和股票相关的因子”，可以将已有因子作为基础因子组加入排序函数。

设已有因子包括动量、波动率、成交额、估值或行业相对收益等：

\[
\mathbf{F}_{s,t}^{base}
=\{Momentum,Volatility,Turnover,Valuation,IndustryReturn,\dots\}
\]

融合模型可以写为：

\[
Score_{s,t}
=\gamma Score_{s,t}^{text}
+(1-\gamma)Score_{s,t}^{market}
\]

其中：

- \(Score_{s,t}^{text}\)：本文构造的舆情事件因子得分；
- \(Score_{s,t}^{market}\)：已有市场因子得分；
- \(\gamma\)：文本另类数据权重。

论文中可以先以纯文本因子模型作为第三章主模型，再在扩展实验中加入已有市场因子，比较文本另类数据是否带来增量信息。

## 10. 对比实验设计

为了证明主模型相较 baseline 有改进，建议设计以下对比：

| 模型 | 使用信息 | 作用 |
| --- | --- | --- |
| M0：直接情绪 baseline | 当日净情绪 | 最简单对照 |
| M1：基础综合因子 | 当日净情绪、负面冲击、关注度、动量 | 验证多因子聚合 |
| M2：时序状态模型 | M1 + EMA 情绪状态 + 负面冲击记忆 | 验证时序累计价值 |
| M3：事件状态模型 | M2 + 分类事件状态因子 | 验证事件拆分价值 |
| M4：融合排序模型 | M3 + 传统股票因子 | 验证另类数据增量价值 |

评价指标建议采用：

1. `top k` 平均未来收益；
2. `top k` 相对股票池平均收益；
3. top-bottom spread；
4. Spearman Rank IC；
5. 横截面排序命中率。

## 11. 第三章可采用的模型命名

论文中建议将本文主模型命名为：

**基于事件-情绪状态因子的股票横截面排序模型**

英文可写为：

**Event-Sentiment State Factor Ranking Model**

该命名的优点是能够同时覆盖第三章的三个重点：

1. `Event`：不是只做情绪分类，而是区分事件类型；
2. `Sentiment State`：不是只看当天情绪，而是构造时序状态；
3. `Factor Ranking`：最终服务于因子排序和 top k 股票选择。

## 12. 第三章与第四章的边界

第三章重点写方法和模型：

- 新闻如何结构化；
- 情绪如何数值化；
- 数值如何聚合成因子；
- 因子如何进入排序模型；
- 模型相比 baseline 有哪些改进。

第四章重点写系统和智能体实现：

- 数据采集 agent；
- 新闻筛选 agent；
- LLM 情绪事件分析 agent；
- 因子计算 agent；
- 排序选股 agent；
- 可视化与报告生成 agent。

也就是说，第三章回答“模型怎么设计”，第四章回答“系统怎么实现并自动运行”。

## 13. 当前最适合写进论文的结论

当前实验已经支持如下结论：

1. 使用 LLM API 直接分析单条新闻，可以作为有效 baseline；
2. 单日净情绪因子能够形成股票舆情画像，但直接用于排序时稳定性不足；
3. 时序累计因子和事件状态因子更适合作为决策信号；
4. 股票选择问题可以自然建模为横截面 top k 排序问题；
5. 第三章主模型可以定位为“文本另类数据因子挖掘 + 横截面排序”，第四章再将其封装为智能体系统。
