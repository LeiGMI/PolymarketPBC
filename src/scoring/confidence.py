"""
Confidence Scoring Engine — Bayesian edge estimation + Kelly criterion sizing.

Mathematical Framework:
━━━━━━━━━━━━━━━━━━━━━

1. BAYESIAN PROBABILITY UPDATE
   Given market price p_market and AI-estimated shift Δp:
   
   p_agent = p_market + Δp × w_evidence × w_confidence × w_freshness
   
   Where:
   - w_evidence: Evidence weight (how much to trust AI analysis)
   - w_confidence: LLM self-reported confidence (0-1)
   - w_freshness: Time decay factor = exp(-t / τ), τ = half-life

2. MULTI-SOURCE AGGREGATION
   When multiple news sources point in the same direction:
   
   p_combined = p_market + Σ(Δp_i × w_i) × (1 + bonus_agreement)
   
   Agreement bonus only applies when ≥2 independent sources agree on direction.

3. EDGE CALCULATION
   edge = p_agent - p_market
   
   We only trade when |edge| > min_edge_threshold (default 5%).

4. KELLY CRITERION POSITION SIZING
   Full Kelly: f* = (p × b - q) / b
   Where:
   - p = estimated probability of winning
   - q = 1 - p
   - b = payout odds = (1 / p_market) - 1
   
   We use fractional Kelly (f = fraction × f*) for safety.
   Default fraction = 0.25 (quarter-Kelly).

5. ORDERBOOK-ADJUSTED SIZING
   Final size is reduced if:
   - Spread is too wide (cost of crossing)
   - Depth is too thin (slippage risk)
   - Imbalance strongly opposes our direction
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """A fully scored trade recommendation."""
    market_slug: str
    market_question: str
    token_id: str
    
    # Direction
    side: str  # "BUY" or "SELL"
    outcome: str  # "Yes" or "No"
    
    # Probabilities
    market_price: float  # Current market-implied probability
    agent_price: float  # Our estimated probability
    edge: float  # agent_price - market_price (signed)
    
    # Sizing
    kelly_fraction: float  # Raw Kelly fraction of bankroll
    position_size_usdc: float  # Dollar amount to trade
    limit_price: float  # Price to submit the order at
    
    # Confidence metrics
    confidence_score: float  # Overall confidence (0-1)
    evidence_count: int  # Number of supporting signals
    source_agreement: bool  # Multiple sources agree?
    freshness_weight: float  # Time decay factor
    
    # Orderbook quality
    spread_cost: float  # Cost of crossing the spread
    depth_score: float  # How deep the book is (0-1)
    
    # Reasoning
    reasoning: str
    causal_chain: str
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        return {
            "market_slug": self.market_slug,
            "market_question": self.market_question,
            "token_id": self.token_id,
            "side": self.side,
            "outcome": self.outcome,
            "market_price": round(self.market_price, 4),
            "agent_price": round(self.agent_price, 4),
            "edge": round(self.edge, 4),
            "edge_pct": f"{self.edge * 100:+.1f}%",
            "kelly_fraction": round(self.kelly_fraction, 4),
            "position_size_usdc": round(self.position_size_usdc, 2),
            "limit_price": round(self.limit_price, 4),
            "confidence_score": round(self.confidence_score, 3),
            "evidence_count": self.evidence_count,
            "source_agreement": self.source_agreement,
            "freshness_weight": round(self.freshness_weight, 3),
            "spread_cost": round(self.spread_cost, 4),
            "depth_score": round(self.depth_score, 3),
            "reasoning": self.reasoning,
            "causal_chain": self.causal_chain,
            "created_at": self.created_at.isoformat(),
        }


class ConfidenceScorer:
    """
    Combines AI analysis, orderbook data, and Kelly criterion
    to produce fully scored trade signals.
    """
    
    def __init__(
        self,
        evidence_weight: float = 0.15,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.05,
        max_position_pct: float = 0.10,
        decay_half_life_hours: float = 4.0,
        source_agreement_bonus: float = 0.10,
    ):
        self.evidence_weight = evidence_weight
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_position_pct = max_position_pct
        self.decay_half_life = decay_half_life_hours * 3600  # Convert to seconds
        self.agreement_bonus = source_agreement_bonus
    
    def score(
        self,
        market,            # Market object
        impacts: list,     # list[MarketImpact] for this market
        orderbook,         # OrderBook object
        bankroll: float,   # Current bankroll in USDC
    ) -> Optional[TradeSignal]:
        """
        Score a market given AI impacts and orderbook state.
        Returns a TradeSignal if edge exceeds threshold, else None.
        """
        if not impacts or not orderbook.mid_price:
            return None
        
        market_price = orderbook.mid_price
        
        # Clamp market price to valid range
        market_price = max(0.01, min(0.99, market_price))
        
        # ─── Step 1: Aggregate probability shifts ───
        now = datetime.now(timezone.utc)
        weighted_shifts = []
        sources = set()
        directions = []
        
        for impact in impacts:
            # Time decay
            age_seconds = (now - impact.analyzed_at).total_seconds()
            freshness = math.exp(-age_seconds * math.log(2) / self.decay_half_life)
            
            # Weighted shift
            w = (
                impact.signed_shift
                * self.evidence_weight
                * impact.confidence
                * freshness
                * (1.5 if impact.is_new_information else 1.0)  # New info bonus
            )
            weighted_shifts.append(w)
            sources.add(impact.news_article_id)
            directions.append(impact.direction)
        
        # Check for source agreement
        up_count = sum(1 for d in directions if d == "UP")
        down_count = sum(1 for d in directions if d == "DOWN")
        source_agreement = (
            len(sources) >= 2
            and (up_count >= 2 or down_count >= 2)
            and (up_count == 0 or down_count == 0)  # No conflicting signals
        )
        
        total_shift = sum(weighted_shifts)
        if source_agreement:
            total_shift *= (1 + self.agreement_bonus)
        
        # ─── Step 2: Compute agent probability ───
        agent_price = market_price + total_shift
        agent_price = max(0.01, min(0.99, agent_price))
        
        edge = agent_price - market_price
        
        # ─── Step 3: Check minimum edge ───
        if abs(edge) < self.min_edge:
            logger.debug(
                f"Edge too small for {market.slug}: {edge:.3f} < {self.min_edge}"
            )
            return None
        
        # ─── Step 4: Determine trade direction ───
        if edge > 0:
            # We think YES is underpriced → BUY YES
            side = "BUY"
            outcome = "Yes"
            p_win = agent_price
            entry_price = orderbook.best_ask if orderbook.best_ask else market_price
            token_id = market.tokens[0]["token_id"] if market.tokens else ""
        else:
            # We think YES is overpriced → BUY NO (equivalent to selling YES)
            side = "BUY"
            outcome = "No"
            p_win = 1 - agent_price
            entry_price = 1 - (orderbook.best_bid if orderbook.best_bid else market_price)
            token_id = market.tokens[1]["token_id"] if len(market.tokens) > 1 else ""
        
        entry_price = max(0.01, min(0.99, entry_price))
        
        # ─── Step 5: Kelly criterion sizing ───
        b = (1.0 / entry_price) - 1  # Payout odds
        if b <= 0:
            return None
        
        q = 1 - p_win
        kelly_full = (p_win * b - q) / b
        kelly_full = max(0, kelly_full)
        
        kelly_adjusted = kelly_full * self.kelly_fraction
        
        # ─── Step 6: Orderbook quality adjustment ───
        spread_cost = orderbook.spread if orderbook.spread else 0.05
        
        # Depth score: how much USDC is available near best price
        relevant_depth = (
            orderbook.ask_depth if side == "BUY" else orderbook.bid_depth
        )
        # Normalize: $10k+ depth = score 1.0, $100 depth = score ~0.3
        depth_score = min(1.0, math.log1p(relevant_depth) / math.log1p(10000))
        
        # Reduce sizing if orderbook is thin or spread is wide
        book_adjustment = depth_score * max(0.3, 1 - spread_cost * 5)
        
        # ─── Step 7: Final position size ───
        position_pct = kelly_adjusted * book_adjustment
        position_pct = min(position_pct, self.max_position_pct)
        position_size = bankroll * position_pct
        
        if position_size < 5.0:  # Below minimum order size
            return None
        
        # ─── Step 8: Compute overall confidence ───
        avg_confidence = sum(i.confidence for i in impacts) / len(impacts)
        freshness_weight = math.exp(
            -min(i.analysis_latency_ms for i in impacts) / 1000 * math.log(2)
            / self.decay_half_life
        )
        
        overall_confidence = (
            avg_confidence * 0.4
            + depth_score * 0.2
            + (1 - spread_cost * 10) * 0.2
            + (0.2 if source_agreement else 0.1)
        )
        overall_confidence = max(0, min(1, overall_confidence))
        
        # ─── Build signal ───
        best_impact = max(impacts, key=lambda i: abs(i.probability_shift))
        
        signal = TradeSignal(
            market_slug=market.slug,
            market_question=market.question,
            token_id=token_id,
            side=side,
            outcome=outcome,
            market_price=market_price,
            agent_price=agent_price,
            edge=edge,
            kelly_fraction=kelly_adjusted,
            position_size_usdc=round(position_size, 2),
            limit_price=entry_price,
            confidence_score=overall_confidence,
            evidence_count=len(impacts),
            source_agreement=source_agreement,
            freshness_weight=freshness_weight,
            spread_cost=spread_cost,
            depth_score=depth_score,
            reasoning=best_impact.reasoning,
            causal_chain=best_impact.causal_chain,
        )
        
        logger.info(
            f"SIGNAL: {signal.side} {signal.outcome} on '{market.slug}' "
            f"| edge={signal.edge:+.3f} | size=${signal.position_size_usdc:.2f} "
            f"| confidence={signal.confidence_score:.2f}"
        )
        
        return signal
    
    @staticmethod
    def expected_value(signal: TradeSignal) -> float:
        """
        Calculate expected value of the trade.
        EV = (p_win × profit_if_win) - (p_lose × loss_if_lose)
        
        In a binary market:
        - Win payout = (1 / entry_price - 1) × position_size
        - Loss = position_size
        """
        p_win = signal.agent_price if signal.outcome == "Yes" else (1 - signal.agent_price)
        p_lose = 1 - p_win
        
        profit_if_win = signal.position_size_usdc * (1 / signal.limit_price - 1)
        loss_if_lose = signal.position_size_usdc
        
        ev = p_win * profit_if_win - p_lose * loss_if_lose
        return round(ev, 2)
