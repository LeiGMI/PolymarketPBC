"""
AI Analysis Engine — LLM-powered news-to-market impact analysis.

Uses structured prompting to extract:
1. Relevance: Which markets does this news affect?
2. Direction: Does it push probability UP or DOWN?
3. Magnitude: How significant is the shift? (0-1 scale)
4. Confidence: How certain is the analysis?

Supports Claude, OpenAI, or any OpenAI-compatible API.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# System prompt for the analysis LLM
ANALYSIS_SYSTEM_PROMPT = """You are an expert prediction market analyst and news trader.

Your job: Given a news article and a list of active prediction markets, determine:
1. Which markets are affected by this news
2. Whether the news pushes the probability UP or DOWN for the "Yes" outcome
3. How large the probability shift should be
4. Your confidence in this assessment

You think like a superforecaster: you consider base rates, update incrementally,
account for the news already being priced in, and distinguish between noise and signal.

CRITICAL RULES:
- Most news is NOISE. Only flag markets where there is a CLEAR causal link.
- Markets often price in information quickly. Be conservative with magnitude.
- Consider whether this is genuinely NEW information or already known.
- Express probability shifts as decimals (e.g., 0.05 = 5 percentage point shift).
- Never suggest shifts > 0.20 unless the news is truly earth-shattering (election result, war declaration, etc.)

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

ANALYSIS_USER_TEMPLATE = """NEWS ARTICLE:
Title: {title}
Description: {description}
Source: {source}
Published: {published_at}

ACTIVE MARKETS (top candidates by relevance):
{markets_text}

Analyze the news and return a JSON array of affected markets:
{{
  "affected_markets": [
    {{
      "market_slug": "<slug>",
      "market_question": "<question>",
      "direction": "UP" | "DOWN" | "NEUTRAL",
      "probability_shift": <float 0.0-0.30>,
      "reasoning": "<1-2 sentence explanation>",
      "confidence": <float 0.0-1.0>,
      "is_new_information": true | false,
      "causal_chain": "<brief causal link from news to market>"
    }}
  ],
  "overall_signal_strength": "STRONG" | "MODERATE" | "WEAK" | "NOISE"
}}

If no markets are meaningfully affected, return: {{"affected_markets": [], "overall_signal_strength": "NOISE"}}"""


@dataclass
class MarketImpact:
    """Result of analyzing a news article's impact on a specific market."""
    market_slug: str
    market_question: str
    direction: str  # UP, DOWN, NEUTRAL
    probability_shift: float  # Magnitude of shift (always positive)
    reasoning: str
    confidence: float  # 0-1
    is_new_information: bool
    causal_chain: str
    
    # Metadata
    news_article_id: str = ""
    news_title: str = ""
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_latency_ms: float = 0.0
    
    @property
    def signed_shift(self) -> float:
        """Probability shift with direction applied."""
        if self.direction == "UP":
            return self.probability_shift
        elif self.direction == "DOWN":
            return -self.probability_shift
        return 0.0
    
    def to_dict(self) -> dict:
        return {
            "market_slug": self.market_slug,
            "market_question": self.market_question,
            "direction": self.direction,
            "probability_shift": self.probability_shift,
            "signed_shift": self.signed_shift,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "is_new_information": self.is_new_information,
            "causal_chain": self.causal_chain,
            "news_article_id": self.news_article_id,
            "news_title": self.news_title,
            "analyzed_at": self.analyzed_at.isoformat(),
            "analysis_latency_ms": round(self.analysis_latency_ms, 1),
        }


@dataclass
class AnalysisResult:
    """Full analysis result for a news article."""
    article_id: str
    article_title: str
    impacts: list[MarketImpact]
    signal_strength: str  # STRONG, MODERATE, WEAK, NOISE
    total_latency_ms: float
    
    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "article_title": self.article_title,
            "signal_strength": self.signal_strength,
            "num_affected_markets": len(self.impacts),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "impacts": [i.to_dict() for i in self.impacts],
        }


class AnalysisEngine:
    """
    LLM-powered news analysis engine.
    
    Supports multiple LLM backends via OpenAI-compatible API format:
    - Anthropic Claude (via API)
    - OpenAI GPT-4
    - Local models via Ollama/vLLM
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.anthropic.com/v1/messages",
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",  # "anthropic" or "openai"
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_url = api_url
        self.model = model
        self.provider = provider
        self._analysis_count = 0
        self._total_latency = 0.0
    
    async def analyze_news(
        self,
        article,  # NewsArticle
        markets: list,  # list[Market]
    ) -> AnalysisResult:
        """
        Analyze a news article against active markets.
        Returns structured impact assessments.
        """
        start = time.monotonic()
        
        # Format markets for the prompt
        markets_text = self._format_markets(markets)
        
        prompt = ANALYSIS_USER_TEMPLATE.format(
            title=article.title,
            description=article.description,
            source=article.source,
            published_at=article.published_at.isoformat() if article.published_at else "Unknown",
            markets_text=markets_text,
        )
        
        # Call LLM
        try:
            response_text = await self._call_llm(prompt)
            parsed = self._parse_response(response_text)
        except Exception as e:
            logger.error(f"Analysis failed for article '{article.title}': {e}")
            return AnalysisResult(
                article_id=article.id,
                article_title=article.title,
                impacts=[],
                signal_strength="NOISE",
                total_latency_ms=(time.monotonic() - start) * 1000,
            )
        
        latency = (time.monotonic() - start) * 1000
        
        # Build impact objects
        impacts = []
        for item in parsed.get("affected_markets", []):
            impact = MarketImpact(
                market_slug=item.get("market_slug", ""),
                market_question=item.get("market_question", ""),
                direction=item.get("direction", "NEUTRAL"),
                probability_shift=min(float(item.get("probability_shift", 0)), 0.30),
                reasoning=item.get("reasoning", ""),
                confidence=max(0, min(1, float(item.get("confidence", 0.5)))),
                is_new_information=item.get("is_new_information", False),
                causal_chain=item.get("causal_chain", ""),
                news_article_id=article.id,
                news_title=article.title,
                analysis_latency_ms=latency,
            )
            impacts.append(impact)
        
        self._analysis_count += 1
        self._total_latency += latency
        
        result = AnalysisResult(
            article_id=article.id,
            article_title=article.title,
            impacts=impacts,
            signal_strength=parsed.get("overall_signal_strength", "NOISE"),
            total_latency_ms=latency,
        )
        
        logger.info(
            f"Analyzed '{article.title[:50]}...' → "
            f"{len(impacts)} impacts, signal={result.signal_strength}, "
            f"latency={latency:.0f}ms"
        )
        
        return result
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM API."""
        if not self.api_key:
            # Fallback: return a mock response for demo/paper trading
            return self._mock_analysis(prompt)
        
        async with aiohttp.ClientSession() as session:
            if self.provider == "anthropic":
                return await self._call_anthropic(session, prompt)
            else:
                return await self._call_openai(session, prompt)
    
    async def _call_anthropic(self, session: aiohttp.ClientSession, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": self.model,
            "max_tokens": 2000,
            "system": ANALYSIS_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        async with session.post(self.api_url, json=body, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"Anthropic API error: {data}")
            return data["content"][0]["text"]
    
    async def _call_openai(self, session: aiohttp.ClientSession, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        
        async with session.post(self.api_url, json=body, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"OpenAI API error: {data}")
            return data["choices"][0]["message"]["content"]
    
    def _mock_analysis(self, prompt: str) -> str:
        """
        Deterministic mock for demo/backtesting without API keys.
        Uses keyword matching to simulate reasonable analysis.
        """
        import hashlib
        import random
        
        # Use prompt hash for deterministic but varied results
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        # Extract market slugs from prompt
        lines = prompt.split("\n")
        market_slugs = []
        for line in lines:
            if line.strip().startswith("- ") and "(" in line:
                slug = line.split("(")[-1].rstrip(")")
                market_slugs.append(slug)
        
        # Keywords that suggest signal
        strong_keywords = ["war", "election", "resign", "crash", "emergency", "dead", "dies"]
        moderate_keywords = ["announce", "plan", "report", "data", "poll", "vote", "policy"]
        
        prompt_lower = prompt.lower()
        has_strong = any(kw in prompt_lower for kw in strong_keywords)
        has_moderate = any(kw in prompt_lower for kw in moderate_keywords)
        
        if has_strong and market_slugs:
            slug = rng.choice(market_slugs) if market_slugs else "unknown"
            return json.dumps({
                "affected_markets": [{
                    "market_slug": slug,
                    "market_question": "Simulated market",
                    "direction": rng.choice(["UP", "DOWN"]),
                    "probability_shift": round(rng.uniform(0.05, 0.15), 3),
                    "reasoning": "Strong signal detected in news (simulated analysis)",
                    "confidence": round(rng.uniform(0.6, 0.85), 2),
                    "is_new_information": True,
                    "causal_chain": "News → Direct impact on market outcome"
                }],
                "overall_signal_strength": "STRONG"
            })
        elif has_moderate and market_slugs:
            slug = rng.choice(market_slugs) if market_slugs else "unknown"
            return json.dumps({
                "affected_markets": [{
                    "market_slug": slug,
                    "market_question": "Simulated market",
                    "direction": rng.choice(["UP", "DOWN"]),
                    "probability_shift": round(rng.uniform(0.02, 0.08), 3),
                    "reasoning": "Moderate signal from policy/data news (simulated)",
                    "confidence": round(rng.uniform(0.4, 0.65), 2),
                    "is_new_information": rng.choice([True, False]),
                    "causal_chain": "News → Indirect influence on market outcome"
                }],
                "overall_signal_strength": "MODERATE"
            })
        else:
            return json.dumps({
                "affected_markets": [],
                "overall_signal_strength": "NOISE"
            })
    
    @staticmethod
    def _format_markets(markets: list) -> str:
        """Format markets list for the prompt."""
        lines = []
        for m in markets[:30]:  # Limit to 30 markets in prompt
            lines.append(
                f"- {m.question} "
                f"(slug: {m.slug}, "
                f"volume_24h: ${m.volume_24hr:,.0f}, "
                f"liquidity: ${m.liquidity:,.0f})"
            )
        return "\n".join(lines) if lines else "No markets available."
    
    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse LLM JSON response, handling common formatting issues."""
        # Strip markdown code fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, raw text: {text[:200]}")
            return {"affected_markets": [], "overall_signal_strength": "NOISE"}
    
    def get_stats(self) -> dict:
        return {
            "total_analyses": self._analysis_count,
            "avg_latency_ms": (
                round(self._total_latency / self._analysis_count, 1)
                if self._analysis_count > 0 else 0
            ),
        }
