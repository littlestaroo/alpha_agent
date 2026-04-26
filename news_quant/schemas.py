from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "业绩",
    "经营",
    "投融资",
    "政策",
    "风险",
    "市场舆情",
    "产品",
    "其他",
]

TrendSignal = Literal["上升", "下降", "震荡", "未知"]


class StockMention(BaseModel):
    ts_code: str = Field(description="股票代码，必须来自候选列表")
    stock_name: str = Field(description="股票名称，必须来自候选列表")
    event_type: EventType = Field(description="事件类别")
    event_summary: str = Field(description="与该股票相关的一句话事件摘要")
    sentiment: float = Field(
        ge=-1.0, le=1.0, description="情绪得分，负面为负，正面为正"
    )
    sentiment_strength: float = Field(
        ge=0.0, le=1.0, description="情绪强度，越大表示影响越强"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="模型置信度")
    relevance: float = Field(ge=0.0, le=1.0, description="新闻与股票的相关度")
    event_importance: float = Field(
        ge=0.0, le=1.0, description="事件重要度，用于后续聚合加权"
    )
    risk_flag: int = Field(ge=0, le=1, description="是否为风险类事件")
    trend_signal: TrendSignal = Field(description="对短期情绪趋势的判断")
    keywords: list[str] = Field(default_factory=list, description="关键词")


class ArticleAnalysis(BaseModel):
    article_summary: str = Field(description="文章摘要")
    overall_sentiment: float = Field(
        ge=-1.0, le=1.0, description="全文整体情绪得分"
    )
    mentions: list[StockMention] = Field(default_factory=list)
