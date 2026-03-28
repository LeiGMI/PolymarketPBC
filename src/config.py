"""
Configuration for the Polymarket Autonomous Trading Agent.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PolymarketConfig:
    """Polymarket API configuration."""
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137  # Polygon mainnet
    private_key: Optional[str] = None
    funder_address: Optional[str] = None
    signature_type: int = 0  # 0=EOA, 1=Magic/Email, 2=Gnosis Safe


@dataclass
class NewsConfig:
    """News ingestion configuration."""
    # RSS feeds for breaking news (low latency)
    rss_feeds: list = field(default_factory=lambda: [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.reuters.com/reuters/topNews",
        "https://www.politico.com/rss/politicopicks.xml",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ])
    # Polling interval in seconds
    poll_interval: int = 30
    # Max articles to process per cycle
    max_articles_per_cycle: int = 20
    # Relevance score threshold (0-1) for filtering
    relevance_threshold: float = 0.3


@dataclass
class ScoringConfig:
    """Confidence scoring configuration."""
    # Bayesian prior parameters
    prior_confidence: float = 0.5  # Start at maximum uncertainty
    evidence_weight: float = 0.15  # How much each evidence shifts posterior
    
    # Kelly criterion parameters
    kelly_fraction: float = 0.25  # Fractional Kelly (quarter-Kelly for safety)
    min_edge_threshold: float = 0.05  # Minimum 5% edge to trade
    max_position_pct: float = 0.10  # Max 10% of bankroll per trade
    
    # Confidence decay
    decay_half_life_hours: float = 4.0  # Confidence decays over time
    
    # Multi-source agreement bonus
    source_agreement_bonus: float = 0.10  # Bonus when sources agree


@dataclass
class ExecutionConfig:
    """Trade execution configuration."""
    mode: str = "paper"  # "paper" only (live not yet implemented)
    initial_bankroll: float = 10000.0  # Starting paper bankroll in USDC
    min_order_size: float = 5.0  # Minimum order size in USDC
    max_slippage_pct: float = 0.02  # Max 2% slippage tolerance
    order_type: str = "GTC"  # Good-til-cancelled


@dataclass
class LLMConfig:
    """LLM analysis engine configuration."""
    provider: str = "anthropic"  # "anthropic" or "openai"
    api_key: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"
    api_url: str = "https://api.anthropic.com/v1/messages"


@dataclass
class AgentConfig:
    """Top-level agent configuration."""
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Agent loop
    cycle_interval_seconds: int = 60
    log_dir: str = "logs"
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load config from environment variables."""
        config = cls()
        
        # Polymarket
        config.polymarket.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        config.polymarket.funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")
        if os.getenv("POLYMARKET_SIG_TYPE"):
            config.polymarket.signature_type = int(os.getenv("POLYMARKET_SIG_TYPE"))
        
        # Execution
        if os.getenv("EXECUTION_MODE"):
            config.execution.mode = os.getenv("EXECUTION_MODE")
        
        # LLM provider — auto-detect from available API keys
        if os.getenv("ANTHROPIC_API_KEY"):
            config.llm.provider = "anthropic"
            config.llm.api_key = os.getenv("ANTHROPIC_API_KEY")
            config.llm.model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
            config.llm.api_url = "https://api.anthropic.com/v1/messages"
        elif os.getenv("OPENAI_API_KEY"):
            config.llm.provider = "openai"
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
            config.llm.model = os.getenv("LLM_MODEL", "gpt-4o")
            config.llm.api_url = "https://api.openai.com/v1/chat/completions"
        # If neither key is set, analysis engine uses mock mode (no API calls)
        
        return config
