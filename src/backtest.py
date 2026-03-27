"""
Backtest Simulation — Generates a realistic simulated trading log.

This module creates a compelling demo by:
1. Fetching real active markets from Polymarket
2. Simulating news events that would affect those markets
3. Running the full scoring pipeline
4. Generating a detailed trade log with P&L tracking

The output is a JSON log file that demonstrates the agent's
decision-making process end-to-end.
"""
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AgentConfig
from src.ingestion.markets import MarketDataClient, OrderBook, OrderBookLevel, Market
from src.scoring.confidence import ConfidenceScorer, TradeSignal
from src.execution.executor import ExecutionEngine, TradeRecord

logger = logging.getLogger("backtest")


# Simulated news events with clear market connections
SIMULATED_NEWS = [
    {
        "title": "Federal Reserve signals potential rate pause at upcoming meeting",
        "description": "Fed Chair indicates economic data may warrant holding rates steady, citing cooling inflation and stable employment figures.",
        "source": "Reuters",
        "category": "economics",
        "keywords": ["fed", "rate", "interest", "inflation", "economy", "cut"],
    },
    {
        "title": "Major tech company announces breakthrough in AI model efficiency",
        "description": "New research paper shows 10x reduction in compute costs for large language model training, potentially accelerating AI deployment.",
        "source": "Bloomberg",
        "category": "technology",
        "keywords": ["ai", "tech", "model", "compute", "breakthrough", "efficiency"],
    },
    {
        "title": "Senate committee advances bipartisan cryptocurrency regulation bill",
        "description": "The bill would establish clear regulatory framework for digital assets, with provisions for stablecoin oversight and exchange licensing.",
        "source": "Politico",
        "category": "politics",
        "keywords": ["crypto", "regulation", "senate", "bill", "digital", "exchange"],
    },
    {
        "title": "European Central Bank surprises with larger-than-expected rate cut",
        "description": "ECB cuts rates by 50 basis points, exceeding market expectations of 25bp, citing deteriorating economic outlook in the eurozone.",
        "source": "Financial Times",
        "category": "economics",
        "keywords": ["ecb", "rate", "cut", "europe", "economy", "monetary"],
    },
    {
        "title": "Poll shows significant shift in key swing state ahead of election",
        "description": "New high-quality poll shows a 5-point swing in voter preferences, potentially reshaping the electoral map.",
        "source": "NYT/Siena",
        "category": "politics",
        "keywords": ["poll", "election", "swing", "vote", "candidate", "state"],
    },
    {
        "title": "Major cryptocurrency exchange reports security breach affecting user funds",
        "description": "Exchange confirms unauthorized access to hot wallets, estimated losses in the tens of millions. User funds partially affected.",
        "source": "CoinDesk",
        "category": "crypto",
        "keywords": ["exchange", "hack", "security", "crypto", "breach", "digital"],
    },
    {
        "title": "Climate summit reaches unexpected agreement on emissions targets",
        "description": "Major emitters agree to accelerated timeline for carbon reduction, with binding commitments and financial mechanisms.",
        "source": "BBC News",
        "category": "environment",
        "keywords": ["climate", "emissions", "agreement", "carbon", "summit", "binding"],
    },
    {
        "title": "Pharmaceutical company releases positive Phase 3 trial results for breakthrough drug",
        "description": "Drug candidate shows statistically significant improvement over standard of care, clearing path for FDA regulatory submission.",
        "source": "Reuters Health",
        "category": "health",
        "keywords": ["pharma", "drug", "trial", "fda", "approval", "phase"],
    },
]


def generate_synthetic_orderbook(mid_price: float, depth_usdc: float = 5000) -> OrderBook:
    """Generate a realistic-looking orderbook around a mid price."""
    bids = []
    asks = []
    
    for i in range(10):
        offset = (i + 1) * 0.005 + random.uniform(0, 0.003)
        size = random.uniform(50, 500) * (1 / (i + 1))  # More size near top
        
        bid_price = max(0.01, mid_price - offset)
        ask_price = min(0.99, mid_price + offset)
        
        bids.append(OrderBookLevel(price=round(bid_price, 4), size=round(size, 2)))
        asks.append(OrderBookLevel(price=round(ask_price, 4), size=round(size, 2)))
    
    bids.sort(key=lambda x: x.price, reverse=True)
    asks.sort(key=lambda x: x.price)
    
    return OrderBook(token_id="sim", bids=bids, asks=asks)


async def run_backtest(output_dir: str = "logs"):
    """
    Run a simulated backtest demonstrating the agent's full pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("═══ BACKTEST SIMULATION START ═══")
    logger.info("Fetching real markets from Polymarket...")
    
    # Fetch real markets
    market_client = MarketDataClient()
    try:
        real_markets = await market_client.get_active_markets(limit=30, min_liquidity=5000)
    except Exception as e:
        logger.warning(f"Could not fetch live markets: {e}. Using synthetic data.")
        real_markets = _generate_synthetic_markets()
    finally:
        await market_client.close()
    
    if not real_markets:
        real_markets = _generate_synthetic_markets()
    
    logger.info(f"Using {len(real_markets)} markets for backtest")
    
    # Initialize components — slightly more aggressive params for demo
    scorer = ConfidenceScorer(
        evidence_weight=0.40,       # Higher weight in backtest (more responsive)
        kelly_fraction=0.25,
        min_edge=0.03,              # Lower threshold for demo (3%)
        max_position_pct=0.10,
    )
    executor = ExecutionEngine(
        mode="paper",
        initial_bankroll=10000.0,
        max_slippage_pct=0.03,
        log_dir=output_dir,
    )
    
    # Simulation state
    backtest_log = {
        "metadata": {
            "simulation_id": f"bt_{int(time.time())}",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "initial_bankroll": 10000.0,
            "num_markets": len(real_markets),
            "num_news_events": len(SIMULATED_NEWS),
            "scoring_params": {
                "evidence_weight": 0.40,
                "kelly_fraction": 0.25,
                "min_edge_threshold": 0.03,
                "max_position_pct": 0.10,
                "decay_half_life_hours": 4.0,
                "note": "Backtest uses more aggressive evidence_weight (0.40 vs 0.15 production) to demonstrate trading pipeline. Production config is more conservative."
            },
        },
        "cycles": [],
    }
    
    # Run simulated cycles
    for cycle_num, news_event in enumerate(SIMULATED_NEWS, 1):
        logger.info(f"\n─── Cycle {cycle_num}: {news_event['title'][:60]}... ───")
        
        cycle_result = {
            "cycle_id": cycle_num,
            "timestamp": (datetime.now(timezone.utc) - timedelta(hours=len(SIMULATED_NEWS) - cycle_num)).isoformat(),
            "news_event": news_event,
            "analysis": [],
            "signals": [],
            "trades": [],
            "portfolio_snapshot": None,
        }
        
        # Simulate AI analysis: match news to markets
        matched_markets = _match_news_to_markets(news_event, real_markets)
        
        for market, impact_data in matched_markets:
            cycle_result["analysis"].append({
                "market_slug": market.slug,
                "market_question": market.question,
                "direction": impact_data["direction"],
                "probability_shift": impact_data["shift"],
                "confidence": impact_data["confidence"],
                "reasoning": impact_data["reasoning"],
            })
            
            # Generate orderbook
            current_price = market.best_bid_yes or random.uniform(0.3, 0.7)
            orderbook = generate_synthetic_orderbook(current_price)
            
            # Create mock impact for scoring
            from src.analysis.engine import MarketImpact
            impact = MarketImpact(
                market_slug=market.slug,
                market_question=market.question,
                direction=impact_data["direction"],
                probability_shift=impact_data["shift"],
                reasoning=impact_data["reasoning"],
                confidence=impact_data["confidence"],
                is_new_information=True,
                causal_chain=impact_data["causal_chain"],
                news_article_id=f"sim_{cycle_num}",
                news_title=news_event["title"],
            )
            
            # Score the opportunity
            signal = scorer.score(
                market=market,
                impacts=[impact],
                orderbook=orderbook,
                bankroll=executor.portfolio.cash,
            )
            
            if signal:
                ev = scorer.expected_value(signal)
                cycle_result["signals"].append({
                    **signal.to_dict(),
                    "expected_value": ev,
                })
                
                # Execute if positive EV
                if ev > 0:
                    trade = await executor.execute(signal, orderbook)
                    if trade:
                        cycle_result["trades"].append(trade.to_dict())
                        logger.info(
                            f"  TRADE: {trade.side} {trade.outcome} on '{trade.market_slug}' "
                            f"| ${trade.size_usdc:.2f} @ {trade.fill_price:.4f} "
                            f"| edge={trade.edge:+.3f} | EV=${ev:.2f}"
                        )
        
        # Simulate price movements for existing positions
        _simulate_position_updates(executor.portfolio, cycle_num)
        
        cycle_result["portfolio_snapshot"] = executor.get_portfolio_summary()
        backtest_log["cycles"].append(cycle_result)
        
        logger.info(
            f"  Portfolio: ${executor.portfolio.total_value:,.2f} "
            f"(PnL: ${executor.portfolio.total_pnl:+,.2f} / "
            f"{executor.portfolio.total_pnl_pct:+.1f}%)"
        )
    
    # Final summary
    backtest_log["summary"] = {
        "end_time": datetime.now(timezone.utc).isoformat(),
        "total_cycles": len(SIMULATED_NEWS),
        "total_trades": executor.portfolio.num_trades,
        "final_portfolio_value": round(executor.portfolio.total_value, 2),
        "total_pnl": round(executor.portfolio.total_pnl, 2),
        "total_pnl_pct": round(executor.portfolio.total_pnl_pct, 2),
        "portfolio": executor.get_portfolio_summary(),
    }
    
    # Save backtest log
    log_path = os.path.join(output_dir, "backtest_log.json")
    with open(log_path, "w") as f:
        json.dump(backtest_log, f, indent=2, default=str)
    
    logger.info(f"\n═══ BACKTEST COMPLETE ═══")
    logger.info(f"Final Portfolio: ${executor.portfolio.total_value:,.2f}")
    logger.info(f"Total PnL: ${executor.portfolio.total_pnl:+,.2f} ({executor.portfolio.total_pnl_pct:+.1f}%)")
    logger.info(f"Total Trades: {executor.portfolio.num_trades}")
    logger.info(f"Backtest log saved to: {log_path}")
    
    return backtest_log


def _match_news_to_markets(news: dict, markets: list) -> list:
    """
    Simple keyword matching to connect news to markets.
    In production, this is done by the AI analysis engine.
    """
    matches = []
    keywords = set(news.get("keywords", []))
    
    for market in markets:
        question_lower = market.question.lower()
        desc_lower = (market.description or "").lower()
        slug_lower = market.slug.lower()
        text = question_lower + " " + desc_lower + " " + slug_lower
        
        overlap = sum(1 for kw in keywords if kw in text)
        
        if overlap >= 2:
            # Generate plausible impact with deterministic seed
            import hashlib
            seed = int(hashlib.md5(f"{news['title']}|{market.slug}".encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            
            direction = rng.choice(["UP", "DOWN"])
            base_shift = 0.04 + (overlap / max(len(keywords), 1)) * 0.08
            shift = round(rng.uniform(base_shift * 0.8, base_shift * 1.3), 3)
            confidence = round(min(0.90, 0.45 + overlap * 0.12 + rng.uniform(0, 0.1)), 2)
            
            matches.append((market, {
                "direction": direction,
                "shift": shift,
                "confidence": confidence,
                "reasoning": f"News about {news['category']} directly affects market: '{market.question[:60]}'",
                "causal_chain": f"{news['title'][:50]}... → {direction} pressure on '{market.question[:50]}...'"
            }))
    
    # Sort by confidence, return top matches
    matches.sort(key=lambda x: x[1]["confidence"], reverse=True)
    return matches[:3]


def _simulate_position_updates(portfolio, cycle_num: int):
    """Simulate realistic price movements on existing positions."""
    for token_id, pos in portfolio.positions.items():
        # Random walk with slight mean reversion
        change = random.gauss(0, 0.02)
        current = pos.get("current_price", pos["avg_price"])
        new_price = max(0.01, min(0.99, current + change))
        pos["current_price"] = round(new_price, 4)


def _generate_synthetic_markets() -> list:
    """Fallback: generate synthetic markets for offline testing."""
    synthetic = [
        (
            "Will the Fed cut interest rates at the next meeting?",
            "fed-rate-cut-next-meeting",
            "Will the Federal Reserve cut the fed funds rate at the next FOMC meeting? Covers inflation data, economy signals, and rate policy decisions.",
            0.45, 85000, 32000,
        ),
        (
            "Will Bitcoin exceed $150K by July 2026?",
            "btc-150k-july",
            "Will the price of Bitcoin (BTC) on major crypto exchanges surpass $150,000 before July 1, 2026? Covers crypto regulation, exchange security, and digital asset markets.",
            0.32, 120000, 45000,
        ),
        (
            "Will the US pass comprehensive crypto regulation in 2026?",
            "us-crypto-regulation-2026",
            "Will the United States Senate pass a comprehensive digital asset and cryptocurrency regulation bill and have it signed into law in 2026?",
            0.55, 80000, 30000,
        ),
        (
            "Will GPT-5 or a comparable AI model be released before September 2026?",
            "gpt5-release-sep",
            "Will OpenAI release GPT-5, or will any major tech company release a breakthrough AI model with significantly improved compute efficiency, before September 2026?",
            0.40, 60000, 20000,
        ),
        (
            "Will there be a US government shutdown in 2026?",
            "govt-shutdown-2026",
            "Will the US federal government experience a shutdown lasting at least 24 hours in 2026? Covers budget votes and Senate policy deadlocks.",
            0.25, 40000, 15000,
        ),
        (
            "Will any major election poll shift by 5+ points in a swing state?",
            "election-poll-swing-state",
            "Will a major high-quality poll (NYT/Siena, Fox News, Quinnipiac) show a 5+ point swing in any key swing state in the next election cycle? Covers candidate favorability and vote intention.",
            0.30, 55000, 22000,
        ),
        (
            "Will a major global climate agreement be ratified in 2026?",
            "climate-agreement-2026",
            "Will nations at a major climate summit reach a binding agreement on carbon emissions reduction targets with enforcement mechanisms in 2026?",
            0.20, 25000, 12000,
        ),
        (
            "Will the FDA approve a major new drug in Q2 2026?",
            "fda-drug-approval-q2",
            "Will the FDA grant approval to a new drug candidate based on positive Phase 3 clinical trial results in Q2 2026? Covers pharma regulatory submissions.",
            0.60, 35000, 18000,
        ),
        (
            "Will the ECB cut rates more than expected in 2026?",
            "ecb-rate-cut-surprise",
            "Will the European Central Bank deliver a rate cut larger than market consensus expectations at any meeting in 2026? Covers ECB monetary policy and the European economy.",
            0.35, 42000, 19000,
        ),
        (
            "Will a major crypto exchange suffer a hack exceeding $50M in 2026?",
            "crypto-exchange-hack-2026",
            "Will any major cryptocurrency exchange experience a security breach resulting in losses exceeding $50 million in 2026? Covers exchange security and digital asset custody.",
            0.28, 30000, 14000,
        ),
    ]
    
    markets = []
    for question, slug, description, price, vol, liq in synthetic:
        m = Market(
            condition_id=f"synth_{slug}",
            question=question,
            description=description,
            slug=slug,
            tokens=[
                {"token_id": f"tok_yes_{slug}", "outcome": "Yes"},
                {"token_id": f"tok_no_{slug}", "outcome": "No"},
            ],
            active=True,
            closed=False,
            volume=vol,
            volume_24hr=vol * 0.05,
            liquidity=liq,
            best_bid_yes=price,
        )
        markets.append(m)
    
    return markets


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run_backtest())
