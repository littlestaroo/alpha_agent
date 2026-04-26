# 毕业论文写作蓝图

## 1. 论文题目方向

可采用如下题目风格：

**基于情绪类文本另类数据的股票因子挖掘与智能体选股系统设计**

也可以稍作变化：

- 基于新闻舆情另类数据的股票因子挖掘方法与智能体实现
- 面向股票排序任务的文本另类数据因子挖掘与多智能体系统设计

## 2. 章节结构

建议采用如下结构：

### 第一章 绪论

写三部分：

1. 研究背景  
   新闻、舆情、公告等文本另类数据对市场预期和投资者行为有影响。

2. 研究意义  
   传统量价因子难以充分反映非结构化信息，文本另类数据为选股提供了补充维度。

3. 本文工作  
   提出一套从文本到因子再到股票排序的完整方法，并实现为智能体系统。

### 第二章 相关工作

建议分三块：

1. 文本情绪分析与事件抽取研究；
2. 另类数据在金融因子和选股中的应用；
3. 智能体系统与自动化投研流程相关研究。

如果篇幅紧张，也可以把部分相关工作压缩并入第一章。

### 第三章 基于情绪类文本另类数据的因子挖掘方法

这一章已经有较完整的材料：

- [chapter3_baseline_method.md](/Users/star/Desktop/agent/chapter3_baseline_method.md)
- [chapter3_factor_system_overview.md](/Users/star/Desktop/agent/chapter3_factor_system_overview.md)
- [chapter3_proposed_factor_ranking_model.md](/Users/star/Desktop/agent/chapter3_proposed_factor_ranking_model.md)
- [chapter3_methodology_argument.md](/Users/star/Desktop/agent/chapter3_methodology_argument.md)
- [chapter3_ranking_validation_report.md](/Users/star/Desktop/agent/chapter3_ranking_validation_report.md)

第三章建议写法：

1. 问题定义；
2. baseline：LLM 单条新闻情绪结构化；
3. 股票级因子构建；
4. 排序模型设计；
5. 优化模型：时序状态、事件状态、可靠性增强；
6. 小样本和扩样本验证；
7. 本章小结。

### 第四章 面向另类数据挖掘因子的智能体设计与实现

这一章的核心文稿已经补好：

- [chapter4_agent_design_and_implementation.md](/Users/star/Desktop/agent/chapter4_agent_design_and_implementation.md)

建议写法：

1. 系统目标与需求；
2. 多智能体架构设计；
3. 工作流与模块实现；
4. 系统运行示例；
5. 工程实现特点；
6. 本章小结。

### 第五章 总结与展望

写三部分：

1. 本文完成了什么；
2. 当前结果的局限性；
3. 后续工作展望。

## 3. 第三章最核心的论证主线

第三章不要写成“我调用了 LLM 得到了分数”，而要写成：

1. 单条新闻数值化只是前置步骤；
2. 真正的研究对象是股票级、日频化因子；
3. 当天直接情绪值并不够，需要时序累计和事件状态建模；
4. 最终任务是股票横截面排序和 top k 选择。

一句话主线可以写成：

> 本文将文本舆情信息挖掘问题进一步建模为股票横截面排序问题，并提出从新闻文本到股票级情绪事件因子的结构化生成与排序方法。

## 4. 第四章最核心的论证主线

第四章不要只是列脚本，而要强调：

1. 第三章的方法已经被系统化封装；
2. 各步骤由不同 agent 承担；
3. 系统能够自动完成从新闻输入到 top k 股票输出的全过程；
4. 系统具备扩展空间。

一句话主线可以写成：

> 本文将第三章提出的因子挖掘与排序方法实现为一个多智能体协同工作流系统，从而提升了方法的自动化、模块化与可复用性。

## 5. 当前最真实、最稳的实验结论

你现在最适合写进论文的结论是：

1. 仅使用 LLM 对单条新闻进行情绪结构化，可以形成可运行的 baseline；
2. 在股票层面，需要将直接情绪、时序状态、事件状态和可靠性信息进一步组织为因子体系；
3. 文本另类数据因子已经具备支持横截面排序和 top k 股票选择的能力；
4. 当前小样本结果说明 baseline 在短周期上较稳，而优化模型增强了解释性与稳健性；
5. 但最终收益优势仍需要依赖更大样本、传统市场因子融合和权重优化进一步验证。

## 6. 论文里要主动说明的局限

这一部分很重要，提前写好会显得论文更成熟。

建议明确写：

1. 当前样本规模有限；
2. 新闻数据覆盖仍非全市场全量实时输入；
3. 排序权重主要基于经验设定；
4. 传统量价因子尚未充分融合；
5. 因子有效性结论目前仍属于探索性验证。

## 7. 后续最值得继续补的内容

如果你后面还有时间，优先补下面两项：

1. 传统股票因子融合  
   加入动量、波动率、换手率、行业相对收益等因子，形成更完整的排序模型。

2. 更长时间窗口扩样本  
   从当前小样本扩展到更长时间和更多股票，强化第三章结论。

## 8. 已有代码和文档的最小映射

如果你要快速写论文，可以直接按下面映射取材：

- 第三章方法：  
  [chapter3_proposed_factor_ranking_model.md](/Users/star/Desktop/agent/chapter3_proposed_factor_ranking_model.md)

- 第三章论证：  
  [chapter3_methodology_argument.md](/Users/star/Desktop/agent/chapter3_methodology_argument.md)

- 第三章结果：  
  [chapter3_ranking_validation_report.md](/Users/star/Desktop/agent/chapter3_ranking_validation_report.md)

- 第四章系统：  
  [chapter4_agent_design_and_implementation.md](/Users/star/Desktop/agent/chapter4_agent_design_and_implementation.md)

- 核心代码：  
  [ranking.py](/Users/star/Desktop/agent/news_quant/ranking.py)  
  [agent_pipeline.py](/Users/star/Desktop/agent/news_quant/agent_pipeline.py)

## 9. 当前最适合的收束方式

如果现在就开始正式写论文，最稳妥的策略是：

1. 先把第三章和第四章写实；
2. 不夸大收益结果；
3. 突出方法完整性、系统实现完整性和后续扩展价值；
4. 把论文定位成“从文本另类数据到股票排序系统的研究与实现”。
