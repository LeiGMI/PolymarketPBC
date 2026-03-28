"""
Trade Execution Module — Paper trading simulator + live trading interface.

Paper trading simulates fills against real orderbook data.
Live trading uses the Polymarket py-clob-client SDK.
"""
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of an executed (or simulated) trade."""
    trade_id: str
    timestamp: datetime
    market_slug: str
    market_question: str
    token_id: str
    side: str  # BUY or SELL
    outcome: str  # Yes or No
    
    # Pricing
    limit_price: float
    fill_price: float  # Actual fill (may differ due to slippage)
    slippage: float  # fill_price - limit_price
    
    # Sizing
    size_usdc: float
    shares_acquired: float  # size_usdc / fill_price
    
    # Signal metadata
    edge: float
    confidence: float
    kelly_fraction: float
    
    # Execution
    mode: str  # "paper" or "live"
    order_id: Optional[str] = None
    status: str = "filled"  # filled, partial, rejected
    
    # P&L tracking
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "market_slug": self.market_slug,
            "market_question": self.market_question,
            "token_id": self.token_id,
            "side": self.side,
            "outcome": self.outcome,
            "limit_price": round(self.limit_price, 4),
            "fill_price": round(self.fill_price, 4),
            "slippage": round(self.slippage, 4),
            "size_usdc": round(self.size_usdc, 2),
            "shares_acquired": round(self.shares_acquired, 2),
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 3),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "mode": self.mode,
            "order_id": self.order_id,
            "status": self.status,
            "current_price": round(self.current_price, 4) if self.current_price else None,
            "unrealized_pnl": round(self.unrealized_pnl, 2) if self.unrealized_pnl else None,
        }


@dataclass
class Portfolio:
    """Paper trading portfolio state."""
    initial_bankroll: float
    cash: float
    positions: dict = field(default_factory=dict)  # token_id -> {shares, avg_price, ...}
    trade_history: list = field(default_factory=list)
    
    @property
    def total_value(self) -> float:
        """Total portfolio value = cash + sum of position values."""
        position_value = sum(
            pos.get("shares", 0) * pos.get("current_price", pos.get("avg_price", 0))
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_bankroll
    
    @property
    def total_pnl_pct(self) -> float:
        if self.initial_bankroll == 0:
            return 0
        return (self.total_pnl / self.initial_bankroll) * 100
    
    @property
    def num_trades(self) -> int:
        return len(self.trade_history)
    
    @property
    def win_rate(self) -> Optional[float]:
        """Win rate based on resolved trades."""
        resolved = [t for t in self.trade_history if t.get("realized_pnl") is not None]
        if not resolved:
            return None
        wins = sum(1 for t in resolved if t["realized_pnl"] > 0)
        return wins / len(resolved)
    
    def to_dict(self) -> dict:
        return {
            "initial_bankroll": self.initial_bankroll,
            "cash": round(self.cash, 2),
            "total_value": round(self.total_value, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "num_trades": self.num_trades,
            "num_open_positions": len(self.positions),
            "win_rate": round(self.win_rate, 3) if self.win_rate is not None else None,
            "positions": {
                k: {
                    "shares": round(v["shares"], 2),
                    "avg_price": round(v["avg_price"], 4),
                    "cost_basis": round(v["cost_basis"], 2),
                    "market_question": v.get("market_question", ""),
                    "outcome": v.get("outcome", ""),
                }
                for k, v in self.positions.items()
            },
        }


class ExecutionEngine:
    """
    Handles trade execution in paper mode.
    
    Paper mode: Simulates fills against real orderbook data with
    realistic slippage modeling.
    
    Live mode: Not yet implemented — architected for future integration
    with the Polymarket py-clob-client SDK.
    """
    
    def __init__(
        self,
        mode: str = "paper",
        initial_bankroll: float = 10000.0,
        max_slippage_pct: float = 0.02,
        log_dir: str = "logs",
    ):
        self.mode = mode
        self.max_slippage = max_slippage_pct
        self.log_dir = log_dir
        
        self.portfolio = Portfolio(
            initial_bankroll=initial_bankroll,
            cash=initial_bankroll,
        )
        
        # Live trading client (initialized if needed)
        self._clob_client = None
        
        os.makedirs(log_dir, exist_ok=True)
    
    async def execute(self, signal, orderbook=None) -> Optional[TradeRecord]:
        """
        Execute a trade signal.
        
        Args:
            signal: TradeSignal from the scoring engine
            orderbook: Current OrderBook (used for slippage simulation)
        
        Returns:
            TradeRecord if executed, None if rejected
        """
        if self.mode == "paper":
            return await self._execute_paper(signal, orderbook)
        else:
            return await self._execute_live(signal)
    
    async def _execute_paper(self, signal, orderbook=None) -> Optional[TradeRecord]:
        """Simulate a paper trade with realistic slippage."""
        
        # Check available cash
        if signal.position_size_usdc > self.portfolio.cash:
            logger.warning(
                f"Insufficient cash: need ${signal.position_size_usdc:.2f}, "
                f"have ${self.portfolio.cash:.2f}"
            )
            return None
        
        # Simulate slippage based on orderbook
        fill_price = self._simulate_fill(signal, orderbook)
        slippage = fill_price - signal.limit_price
        
        # Check slippage tolerance
        if abs(slippage) / signal.limit_price > self.max_slippage:
            logger.warning(
                f"Slippage too high: {slippage:.4f} "
                f"({abs(slippage)/signal.limit_price*100:.1f}%)"
            )
            return None
        
        # Calculate shares
        shares = signal.position_size_usdc / fill_price
        
        # Update portfolio
        self.portfolio.cash -= signal.position_size_usdc
        
        token_id = signal.token_id
        if token_id in self.portfolio.positions:
            pos = self.portfolio.positions[token_id]
            total_cost = pos["cost_basis"] + signal.position_size_usdc
            total_shares = pos["shares"] + shares
            pos["avg_price"] = total_cost / total_shares if total_shares > 0 else 0
            pos["shares"] = total_shares
            pos["cost_basis"] = total_cost
        else:
            self.portfolio.positions[token_id] = {
                "shares": shares,
                "avg_price": fill_price,
                "cost_basis": signal.position_size_usdc,
                "market_slug": signal.market_slug,
                "market_question": signal.market_question,
                "outcome": signal.outcome,
                "current_price": fill_price,
            }
        
        # Create trade record
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc),
            market_slug=signal.market_slug,
            market_question=signal.market_question,
            token_id=signal.token_id,
            side=signal.side,
            outcome=signal.outcome,
            limit_price=signal.limit_price,
            fill_price=fill_price,
            slippage=slippage,
            size_usdc=signal.position_size_usdc,
            shares_acquired=shares,
            edge=signal.edge,
            confidence=signal.confidence_score,
            kelly_fraction=signal.kelly_fraction,
            mode="paper",
            status="filled",
        )
        
        self.portfolio.trade_history.append(trade.to_dict())
        
        # Log the trade
        self._log_trade(trade)
        
        logger.info(
            f"PAPER TRADE: {trade.side} {trade.outcome} on '{trade.market_slug}' "
            f"| {shares:.2f} shares @ ${fill_price:.4f} "
            f"| total ${trade.size_usdc:.2f} | slippage={slippage:+.4f}"
        )
        
        return trade
    
    async def _execute_live(self, signal) -> Optional[TradeRecord]:
        """
        Live trading via Polymarket SDK — not yet implemented.
        
        This method is a placeholder for future integration with
        py-clob-client. For now, it falls back to paper execution
        so the full pipeline can still be demonstrated end-to-end.
        """
        logger.warning(
            "Live trading is not yet implemented. "
            "Falling back to paper execution. "
            "See roadmap in README for py-clob-client integration plan."
        )
        return await self._execute_paper(signal)
    
    def _simulate_fill(self, signal, orderbook=None) -> float:
        """
        Simulate realistic fill price based on orderbook.
        
        Walks the orderbook to estimate actual fill price for our order size.
        """
        if orderbook is None:
            # Without orderbook data, assume small slippage
            return signal.limit_price * 1.002  # 0.2% slippage
        
        levels = orderbook.asks if signal.side == "BUY" else orderbook.bids
        
        if not levels:
            return signal.limit_price * 1.005
        
        remaining_usdc = signal.position_size_usdc
        total_cost = 0
        total_shares = 0
        
        for level in levels:
            level_usdc = level.price * level.size
            if remaining_usdc <= level_usdc:
                shares_at_level = remaining_usdc / level.price
                total_cost += remaining_usdc
                total_shares += shares_at_level
                remaining_usdc = 0
                break
            else:
                total_cost += level_usdc
                total_shares += level.size
                remaining_usdc -= level_usdc
        
        if total_shares > 0:
            return total_cost / total_shares
        return signal.limit_price * 1.005
    
    def _log_trade(self, trade: TradeRecord):
        """Append trade to the log file."""
        log_file = os.path.join(self.log_dir, "trade_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(trade.to_dict()) + "\n")
    
    def get_portfolio_summary(self) -> dict:
        return self.portfolio.to_dict()
    
    def save_state(self, filepath: str = None):
        """Save portfolio state to disk."""
        filepath = filepath or os.path.join(self.log_dir, "portfolio_state.json")
        with open(filepath, "w") as f:
            json.dump(self.portfolio.to_dict(), f, indent=2)
        logger.info(f"Portfolio state saved to {filepath}")
