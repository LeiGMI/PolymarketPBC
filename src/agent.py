"""
Autonomous Trading Agent — Main loop.

Orchestrates: News Ingestion → AI Analysis → Confidence Scoring → Trade Execution
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AgentConfig
from src.ingestion.news import NewsIngestionEngine
from src.ingestion.markets import MarketDataClient
from src.analysis.engine import AnalysisEngine
from src.scoring.confidence import ConfidenceScorer, TradeSignal
from src.execution.executor import ExecutionEngine

logger = logging.getLogger("agent")


class PolymarketAgent:
    """
    Fully autonomous Polymarket trading agent.
    
    Lifecycle per cycle:
    1. Fetch breaking news from RSS feeds
    2. Fetch active Polymarket markets
    3. Use AI to match news → affected markets
    4. Score each opportunity (Bayesian + Kelly)
    5. Execute trades that exceed edge threshold
    6. Log everything for backtesting and auditing
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig.from_env()
        
        # Initialize components
        self.news_engine = NewsIngestionEngine(
            rss_feeds=self.config.news.rss_feeds,
            max_articles=self.config.news.max_articles_per_cycle,
        )
        self.market_client = MarketDataClient()
        self.analysis_engine = AnalysisEngine()
        self.scorer = ConfidenceScorer(
            evidence_weight=self.config.scoring.evidence_weight,
            kelly_fraction=self.config.scoring.kelly_fraction,
            min_edge=self.config.scoring.min_edge_threshold,
            max_position_pct=self.config.scoring.max_position_pct,
            decay_half_life_hours=self.config.scoring.decay_half_life_hours,
            source_agreement_bonus=self.config.scoring.source_agreement_bonus,
        )
        self.executor = ExecutionEngine(
            mode=self.config.execution.mode,
            initial_bankroll=self.config.execution.initial_bankroll,
            max_slippage_pct=self.config.execution.max_slippage_pct,
            log_dir=self.config.log_dir,
        )
        
        self._cycle_count = 0
        self._total_signals = 0
        self._total_trades = 0
        
        os.makedirs(self.config.log_dir, exist_ok=True)
    
    async def run_cycle(self) -> dict:
        """
        Run one complete agent cycle.
        Returns a summary dict of what happened.
        """
        cycle_start = time.monotonic()
        self._cycle_count += 1
        cycle_id = self._cycle_count
        
        logger.info(f"═══ CYCLE {cycle_id} START ═══")
        
        summary = {
            "cycle_id": cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "articles_ingested": 0,
            "markets_scanned": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "signals": [],
            "trades": [],
            "errors": [],
        }
        
        try:
            # ─── Step 1: Ingest News ───
            logger.info("Step 1: Ingesting news...")
            articles = await self.news_engine.fetch_all()
            summary["articles_ingested"] = len(articles)
            
            if not articles:
                logger.info("No new articles this cycle.")
                return summary
            
            # ─── Step 2: Fetch Active Markets ───
            logger.info("Step 2: Fetching active markets...")
            markets = await self.market_client.get_active_markets(
                limit=50,
                min_liquidity=5000,
            )
            summary["markets_scanned"] = len(markets)
            
            if not markets:
                logger.warning("No markets available.")
                return summary
            
            # ─── Step 3: AI Analysis ───
            logger.info(f"Step 3: Analyzing {len(articles)} articles against {len(markets)} markets...")
            all_impacts = {}  # market_slug -> [impacts]
            
            for article in articles[:10]:  # Process top 10 newest articles
                result = await self.analysis_engine.analyze_news(article, markets)
                
                for impact in result.impacts:
                    slug = impact.market_slug
                    if slug not in all_impacts:
                        all_impacts[slug] = []
                    all_impacts[slug].append(impact)
            
            # ─── Step 4: Score & Generate Signals ───
            logger.info(f"Step 4: Scoring {len(all_impacts)} potential opportunities...")
            signals = []
            
            for slug, impacts in all_impacts.items():
                # Find corresponding market
                market = next((m for m in markets if m.slug == slug), None)
                if not market or not market.tokens:
                    continue
                
                # Fetch orderbook for primary token
                token_id = market.tokens[0]["token_id"]
                orderbook = await self.market_client.get_orderbook(token_id)
                
                signal = self.scorer.score(
                    market=market,
                    impacts=impacts,
                    orderbook=orderbook,
                    bankroll=self.executor.portfolio.cash,
                )
                
                if signal:
                    ev = self.scorer.expected_value(signal)
                    signals.append((signal, ev, orderbook))
                    summary["signals"].append({
                        **signal.to_dict(),
                        "expected_value": ev,
                    })
            
            # Sort signals by expected value
            signals.sort(key=lambda x: x[1], reverse=True)
            summary["signals_generated"] = len(signals)
            self._total_signals += len(signals)
            
            # ─── Step 5: Execute Best Signals ───
            logger.info(f"Step 5: Executing top signals ({len(signals)} candidates)...")
            
            for signal, ev, orderbook in signals[:3]:  # Max 3 trades per cycle
                if ev <= 0:
                    logger.info(f"Skipping negative EV signal: {signal.market_slug}")
                    continue
                
                trade = await self.executor.execute(signal, orderbook)
                if trade:
                    summary["trades"].append(trade.to_dict())
                    summary["trades_executed"] += 1
                    self._total_trades += 1
        
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            summary["errors"].append(str(e))
        
        cycle_time = (time.monotonic() - cycle_start) * 1000
        summary["cycle_time_ms"] = round(cycle_time, 1)
        summary["portfolio"] = self.executor.get_portfolio_summary()
        
        logger.info(
            f"═══ CYCLE {cycle_id} COMPLETE ═══ "
            f"| {summary['articles_ingested']} articles "
            f"| {summary['signals_generated']} signals "
            f"| {summary['trades_executed']} trades "
            f"| {cycle_time:.0f}ms "
            f"| Portfolio: ${self.executor.portfolio.total_value:,.2f}"
        )
        
        # Save cycle log
        self._save_cycle_log(summary)
        
        return summary
    
    async def run(self, max_cycles: int = None):
        """Run the agent in a continuous loop."""
        logger.info(
            f"Agent starting in {self.config.execution.mode.upper()} mode "
            f"with ${self.config.execution.initial_bankroll:,.2f} bankroll"
        )
        
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                await self.run_cycle()
                cycle += 1
                
                if max_cycles is None or cycle < max_cycles:
                    logger.info(
                        f"Sleeping {self.config.cycle_interval_seconds}s "
                        f"until next cycle..."
                    )
                    await asyncio.sleep(self.config.cycle_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Agent stopped by user.")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Clean shutdown."""
        await self.market_client.close()
        self.executor.save_state()
        logger.info(
            f"Agent shutdown. Total: {self._cycle_count} cycles, "
            f"{self._total_signals} signals, {self._total_trades} trades"
        )
    
    def _save_cycle_log(self, summary: dict):
        """Save cycle summary to log file."""
        log_file = os.path.join(self.config.log_dir, "cycle_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(summary, default=str) + "\n")
    
    def get_status(self) -> dict:
        """Get current agent status."""
        return {
            "cycles_completed": self._cycle_count,
            "total_signals": self._total_signals,
            "total_trades": self._total_trades,
            "mode": self.config.execution.mode,
            "portfolio": self.executor.get_portfolio_summary(),
            "news_latency": self.news_engine.get_latency_report(),
            "analysis_stats": self.analysis_engine.get_stats(),
        }


def setup_logging(level=logging.INFO):
    """Configure logging for the agent."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def main():
    """Entry point."""
    setup_logging()
    config = AgentConfig.from_env()
    agent = PolymarketAgent(config)
    
    # Run for specified cycles or indefinitely
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Autonomous Trading Agent")
    parser.add_argument("--cycles", type=int, default=None, help="Number of cycles (None=infinite)")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--bankroll", type=float, default=10000.0)
    args = parser.parse_args()
    
    config.execution.mode = args.mode
    config.execution.initial_bankroll = args.bankroll
    
    await agent.run(max_cycles=args.cycles)


if __name__ == "__main__":
    asyncio.run(main())
