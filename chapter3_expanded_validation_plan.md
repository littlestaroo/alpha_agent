# 第三章扩样本验证落地方案

## 1. 目标

当前第三章已经完成：

- 新闻文本到单条新闻数值化
- 股票级因子生成
- 小样本探索性验证

下一步的核心目标不再是证明“单条新闻能不能打分”，而是：

**在更大的股票池和更长的时间窗口下，比较不同因子设计方案的优劣。**

更具体地说，需要回答三个问题：

1. 直接型因子是否稳定有效？
2. 时序累计因子是否优于直接型因子？
3. 事件状态因子是否优于总情绪因子？

## 2. 推荐的实验扩样本路线

考虑到 API 成本、论文进度和样本代表性，建议采用“两阶段”路线。

如果从“本科论文更有说服力”的角度来讲，建议的股票数可以这样理解：

- `12` 只：适合作为过渡版、小规模扩样本验证；
- `20` 只左右：更适合作为论文正式比较不同因子设计的版本；
- `30` 只以上：说服力更强，但 LLM 成本和清洗工作量会明显上升。

因此，本文更推荐的正式扩样本方案是：

**20 只股票 + 2023Q4 或 2023H2 + 先多保留 `top10` 候选新闻，再做事件多样性筛选**

这样既能明显增强横截面比较能力，又不会把工作量一下拉得过大。

### 阶段 A：最小可行扩样本验证

- 股票数：`20` 只
- 时间范围：`2023-10-01` 到 `2023-12-31`
- 每日每股候选新闻条数：`top10`
- 目标：先验证因子设计方向，而不是追求最大样本量

为什么这样选：

- `20` 只股票已经足以显著改善当前只有 `2` 只股票时横截面太弱的问题；
- `2023Q4` 作为第一轮样本，更利于控制成本并快速得到结果；
- 如果一开始就直接截成 `top2` 或 `top3`，容易把同一天同一只股票的新闻压得过窄，甚至几条新闻属于同一种事件类型；
- 因此更合理的做法是：前面先多保留一些候选新闻，等 LLM 抽出事件类型后，再做二次筛选。

### 阶段 B：增强版验证

- 股票数：`20` 只左右
- 时间范围：`2023-07-01` 到 `2023-12-31`
- 每日每股候选新闻条数：`top10`，后续再压缩到 `2-3` 条不同事件类型新闻
- 目标：形成论文中更完整的因子检验结果

如果时间或成本受限，可以先完成阶段 A，再决定是否扩大到阶段 B。

## 3. 已经准备好的扩样本股票池

我已经准备好两份可直接使用的股票池：

`data/stock_universe_thesis_12stocks.csv`
`data/stock_universe_thesis_20stocks.csv`

其中，论文正式推荐使用的是：

`data/stock_universe_thesis_20stocks.csv`

包含：

- 宁德时代
- 贵州茅台
- 工商银行
- 平安银行
- 中芯国际
- 招商银行
- 比亚迪
- 恒瑞医药
- 长江电力
- 中国平安
- 紫金矿业
- 中国移动
- 中信证券
- 伊利股份
- 美的集团
- 三一重工
- 兴业银行
- 爱尔眼科
- 海康威视
- 中国神华

这份股票池的特点是：

- 行业分布相对均衡
- 新闻关注度普遍较高
- 公司简称/别名比较清晰
- 更适合构造新闻舆情实验集

## 4. 推荐的落地步骤

### 步骤 1：从原始新闻中抽取扩样本候选集

```bash
python -m news_quant build-experiment-set-from-raw \
  --raw data/raw/opennewsarchive \
  --universe data/stock_universe_thesis_20stocks.csv \
  --out data/prepared/opennewsarchive_thesis_20stocks_2023Q4_fullraw.jsonl \
  --rejected-out data/prepared/opennewsarchive_thesis_20stocks_2023Q4_fullraw_rejected.jsonl \
  --date-from 2023-10-01 \
  --date-to 2023-12-31
```

这一阶段的目的：

- 从全量原始新闻中先抽出和 20 只股票相关的候选新闻；
- 形成第一层股票相关候选池。

### 步骤 2：做二次强相关过滤

```bash
python -m news_quant refine-thesis-set \
  --in data/prepared/opennewsarchive_thesis_20stocks_2023Q4_fullraw.jsonl \
  --out data/prepared/opennewsarchive_thesis_20stocks_2023Q4_refined.jsonl \
  --rejected-out data/prepared/opennewsarchive_thesis_20stocks_2023Q4_refined_rejected.jsonl
```

这一阶段的目的：

- 保留更像“公司舆情新闻”的记录；
- 降低行业泛新闻、弱别名命中和噪声新闻对 LLM 成本的浪费。

### 步骤 3：构造 LLM 实验集

```bash
python -m news_quant select-thesis-llm-set \
  --in data/prepared/opennewsarchive_thesis_20stocks_2023Q4_refined.jsonl \
  --out data/prepared/opennewsarchive_thesis_20stocks_2023Q4_llmset.jsonl \
  --rejected-out data/prepared/opennewsarchive_thesis_20stocks_2023Q4_llmset_rejected.jsonl \
  --min-score 16 \
  --max-per-stock 250
```

这一阶段的目的：

- 从强相关候选新闻中选出更适合调用 LLM 的实验集；
- 控制每只股票样本上限，避免热门股票挤占样本。

### 步骤 4：按日每股先保留 `top10` 候选新闻

```bash
python -m news_quant build-daily-topk-set \
  --in data/prepared/opennewsarchive_thesis_20stocks_2023Q4_refined.jsonl \
  --out data/prepared/opennewsarchive_thesis_20stocks_2023Q4_top10_perstock.jsonl \
  --rejected-out data/prepared/opennewsarchive_thesis_20stocks_2023Q4_top10_perstock_rejected.jsonl \
  --top-k 10 \
  --date-from 2023-10-01 \
  --date-to 2023-12-31
```

这一阶段的目的：

- 先形成“日频候选新闻集”，而不是最终定稿实验集；
- 控制 API 成本；
- 保留“同日多新闻聚合”的特征；
- 避免一开始就因为 `top2` 过窄而丢掉事件多样性。

### 步骤 5：先在 `top10` 候选集上跑 LLM baseline

```bash
python -m news_quant \
  --news data/prepared/opennewsarchive_thesis_20stocks_2023Q4_top10_perstock.jsonl \
  --universe data/stock_universe_thesis_20stocks.csv \
  --out output/chapter3_baseline_thesis_20stocks_2023Q4_top10
```

这一阶段的目的：

- 得到文章级结构化结果；
- 得到股票级日频因子；
- 后续再接事件多样性筛选与因子检验。

### 步骤 6：基于事件类型做二次筛选

这一阶段是当前方案相较于“直接 top2”最重要的改进。

建议规则是：

1. 同一天、同一只股票，先允许保留较多候选新闻（例如 `top10`）；
2. 等 LLM 跑完后，拿到每条新闻的 `event_type`；
3. 再进行二次筛选：
   - 优先保留不同 `event_type` 的新闻；
   - 如果同一类事件有多条新闻，只保留得分最高的 1 条；
   - 最终同一天每股可以保留 `2-3` 条，但尽量保证类型不同。

这样做的好处是：

- 避免最终样本里同一天全是“同一类经营新闻”；
- 能更好体现“事件拆分因子”的价值；
- 更符合论文里“不同事件类型驱动不同市场反应”的研究逻辑。

## 5. 扩样本后重点比较什么

扩样本之后，建议不要把重点放在“哪一条新闻打得准”，而是比较以下三组因子：

### 5.1 直接型因子

- `net_sentiment_factor`
- `negative_shock_factor`
- `attention_factor`

### 5.2 时序累计因子

- `ema3_sentiment_state`
- `ema5_sentiment_state`
- `negative_shock_carry`

### 5.3 事件状态因子

- `earnings_event_state`
- `operations_event_state`
- `risk_event_state`

建议重点回答：

1. 时序累计因子是否优于直接型因子？
2. 事件状态因子是否优于总情绪因子？
3. 在“事件多样性筛选”之后，因子表现是否更稳？
4. 哪类因子在 3 日、5 日窗口上更有解释力？

## 6. 第三章后续最合理的章节表述

第三章接下来最合理的逻辑不是：

> “继续强调单条新闻情绪打分的准确率”

而应该是：

> “在完成新闻数值化之后，进一步研究股票级因子如何构造，以及不同因子设计方案在扩样本条件下的表现差异”

因此，后续第三章的重点应转向：

- 直接型因子设计
- 时序累计因子设计
- 事件状态因子设计
- 扩样本因子比较检验

## 7. 向老师汇报时可以怎么说

可以直接这样说：

> 老师，我现在第三章已经完成了 baseline 跑通，也已经把数值化之后的股票级因子分成了直接型因子、时序累计因子和事件状态因子三类。下一步我准备不再停留在两只股票的小样本上，而是扩展到 20 只左右股票和更长时间窗口。具体做法上，我不会一开始就把每天每股新闻压成 top2，因为这样容易保留同一种类型的新闻；我会先多保留一些候选新闻，例如每天每股先取 top10，等 LLM 抽出事件类型之后，再做二次筛选，尽量保证同一天样本具有事件多样性。这样更适合比较“当天值因子”和“时序累计/事件状态因子”在扩样本条件下的表现差异。

## 8. 现在最推荐的下一步

如果只做一件最重要的事，建议优先做：

**先跑阶段 A：20 只股票、2023Q4、每天每股先保留 `top10` 候选新闻，再按事件类型做二次筛选。**

原因是：

- 样本比当前更大，但仍可控；
- 能初步回答老师真正关心的问题；
- 成本和工作量不会一下子失控；
- 跑完这一版，第三章就能从“方法探索”进一步升级到“扩样本验证”。
