"""
Polymarket Market Data Module — Fetches markets, orderbooks, and prices.

Uses the Gamma API for market discovery and the CLOB API for orderbook data.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


@dataclass
class OrderBookLevel:
    """Single price level in the orderbook."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Full orderbook snapshot for a token."""
    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_pct(self) -> Optional[float]:
        if self.spread is not None and self.mid_price and self.mid_price > 0:
            return self.spread / self.mid_price
        return None
    
    @property
    def bid_depth(self) -> float:
        """Total USDC on bid side."""
        return sum(level.price * level.size for level in self.bids)
    
    @property
    def ask_depth(self) -> float:
        """Total USDC on ask side."""
        return sum(level.price * level.size for level in self.asks)
    
    @property
    def imbalance_ratio(self) -> Optional[float]:
        """
        Orderbook imbalance: positive = more buying pressure, negative = more selling.
        Range: [-1, 1]
        """
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return None
        return (self.bid_depth - self.ask_depth) / total
    
    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_pct": round(self.spread_pct, 4) if self.spread_pct else None,
            "bid_depth_usdc": round(self.bid_depth, 2),
            "ask_depth_usdc": round(self.ask_depth, 2),
            "imbalance_ratio": round(self.imbalance_ratio, 4) if self.imbalance_ratio else None,
            "num_bid_levels": len(self.bids),
            "num_ask_levels": len(self.asks),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Market:
    """Represents a Polymarket market."""
    condition_id: str
    question: str
    description: str
    slug: str
    tokens: list[dict]  # [{token_id, outcome}]
    active: bool
    closed: bool
    volume: float
    volume_24hr: float
    liquidity: float
    end_date: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    neg_risk: bool = False
    
    # Current prices
    best_bid_yes: Optional[float] = None
    best_ask_yes: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "slug": self.slug,
            "active": self.active,
            "volume": self.volume,
            "volume_24hr": self.volume_24hr,
            "liquidity": self.liquidity,
            "tokens": self.tokens,
            "tags": self.tags,
            "neg_risk": self.neg_risk,
            "best_bid_yes": self.best_bid_yes,
            "best_ask_yes": self.best_ask_yes,
        }


class MarketDataClient:
    """
    Client for fetching Polymarket market data.
    
    Uses Gamma API for market metadata and CLOB API for live orderbooks.
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_active_markets(
        self,
        limit: int = 100,
        order: str = "volume_24hr",
        min_liquidity: float = 1000,
    ) -> list[Market]:
        """
        Fetch active markets sorted by 24h volume.
        Filters out low-liquidity markets that aren't worth trading.
        """
        session = await self._get_session()
        markets = []
        offset = 0
        
        while len(markets) < limit:
            url = (
                f"{GAMMA_API}/events?"
                f"active=true&closed=false"
                f"&order={order}&ascending=false"
                f"&limit=100&offset={offset}"
            )
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(f"Gamma API returned {resp.status}")
                        break
                    events = await resp.json()
            except Exception as e:
                logger.error(f"Failed to fetch markets: {e}")
                break
            
            if not events:
                break
            
            for event in events:
                event_markets = event.get("markets", [])
                for m in event_markets:
                    liquidity = float(m.get("liquidity", 0) or 0)
                    if liquidity < min_liquidity:
                        continue
                    
                    tokens = []
                    for token in m.get("clobTokenIds", "").split(","):
                        token = token.strip()
                        if token:
                            tokens.append({"token_id": token})
                    
                    # Map outcomes to tokens
                    outcomes_str = m.get("outcomes", "")
                    if isinstance(outcomes_str, str):
                        outcomes = [o.strip().strip('"') for o in outcomes_str.split(",")]
                    elif isinstance(outcomes_str, list):
                        outcomes = outcomes_str
                    else:
                        outcomes = []
                    
                    for i, tok in enumerate(tokens):
                        tok["outcome"] = outcomes[i] if i < len(outcomes) else f"outcome_{i}"
                    
                    # Parse prices
                    prices_str = m.get("outcomePrices", "")
                    prices = []
                    if isinstance(prices_str, str) and prices_str:
                        prices = [float(p.strip().strip('"')) for p in prices_str.split(",") if p.strip()]
                    elif isinstance(prices_str, list):
                        prices = [float(p) for p in prices_str]
                    
                    market = Market(
                        condition_id=m.get("conditionId", ""),
                        question=m.get("question", event.get("title", "")),
                        description=m.get("description", event.get("description", "")),
                        slug=event.get("slug", m.get("slug", "")),
                        tokens=tokens,
                        active=m.get("active", True),
                        closed=m.get("closed", False),
                        volume=float(m.get("volume", 0) or 0),
                        volume_24hr=float(m.get("volume24hr", 0) or 0),
                        liquidity=liquidity,
                        end_date=m.get("endDate"),
                        tags=[t.get("label", "") for t in event.get("tags", []) if isinstance(t, dict)],
                        neg_risk=m.get("negRisk", False),
                        best_bid_yes=prices[0] if len(prices) > 0 else None,
                        best_ask_yes=None,
                    )
                    markets.append(market)
                    
                    if len(markets) >= limit:
                        break
                if len(markets) >= limit:
                    break
            
            offset += 100
        
        logger.info(f"Fetched {len(markets)} active markets with liquidity >= ${min_liquidity}")
        return markets
    
    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch live orderbook for a specific token."""
        session = await self._get_session()
        url = f"{CLOB_API}/book?token_id={token_id}"
        
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"CLOB orderbook returned {resp.status} for {token_id}")
                    return OrderBook(token_id=token_id)
                data = await resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch orderbook: {e}")
            return OrderBook(token_id=token_id)
        
        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]
        
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        return OrderBook(token_id=token_id, bids=bids, asks=asks)
    
    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get current midpoint price for a token."""
        session = await self._get_session()
        url = f"{CLOB_API}/midpoint?token_id={token_id}"
        
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return float(data.get("mid", 0))
        except Exception:
            return None
    
    async def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get current best price for a side."""
        session = await self._get_session()
        url = f"{CLOB_API}/price?token_id={token_id}&side={side}"
        
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return float(data.get("price", 0))
        except Exception:
            return None
