"""
News Ingestion Module — Multi-source, latency-optimized news pipeline.

Supports RSS feeds, with architecture ready for webhooks and streaming APIs.
Each article is timestamped at discovery for latency measurement.
"""
import asyncio
import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a single news article with metadata."""
    id: str                          # SHA-256 hash of title+source
    title: str
    description: str
    source: str                      # Feed URL or source name
    published_at: Optional[datetime] = None
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    url: Optional[str] = None
    categories: list = field(default_factory=list)
    
    # Latency tracking
    ingestion_latency_ms: float = 0.0  # Time from published_at to discovered_at
    
    def __post_init__(self):
        if self.published_at and self.discovered_at:
            delta = (self.discovered_at - self.published_at).total_seconds() * 1000
            self.ingestion_latency_ms = max(0, delta)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "discovered_at": self.discovered_at.isoformat(),
            "url": self.url,
            "categories": self.categories,
            "ingestion_latency_ms": round(self.ingestion_latency_ms, 1),
        }


def _generate_article_id(title: str, source: str) -> str:
    """Generate a deterministic ID for deduplication."""
    raw = f"{title.strip().lower()}|{source.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _parse_rss_date(date_str: str) -> Optional[datetime]:
    """Parse various RSS date formats."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            continue
    return None


class NewsIngestionEngine:
    """
    Multi-source news ingestion engine.
    
    Design goals:
    - Minimize latency from publication to discovery
    - Deduplicate across sources
    - Track ingestion latency metrics per source
    """
    
    def __init__(self, rss_feeds: list[str], max_articles: int = 20):
        self.rss_feeds = rss_feeds
        self.max_articles = max_articles
        self._seen_ids: set[str] = set()
        self._latency_stats: dict[str, list[float]] = {}
    
    async def fetch_all(self) -> list[NewsArticle]:
        """
        Fetch news from all configured sources concurrently.
        Returns deduplicated, sorted-by-recency articles.
        """
        fetch_start = time.monotonic()
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            tasks = [self._fetch_rss(session, url) for url in self.rss_feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_articles = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Feed fetch error: {result}")
                continue
            all_articles.extend(result)
        
        # Deduplicate
        new_articles = []
        for article in all_articles:
            if article.id not in self._seen_ids:
                self._seen_ids.add(article.id)
                new_articles.append(article)
                
                # Track latency
                source = article.source
                if source not in self._latency_stats:
                    self._latency_stats[source] = []
                self._latency_stats[source].append(article.ingestion_latency_ms)
        
        # Sort by recency (newest first) and limit
        new_articles.sort(
            key=lambda a: a.discovered_at if a.published_at is None else a.published_at,
            reverse=True
        )
        new_articles = new_articles[:self.max_articles]
        
        fetch_time = (time.monotonic() - fetch_start) * 1000
        logger.info(
            f"Ingested {len(new_articles)} new articles from "
            f"{len(self.rss_feeds)} feeds in {fetch_time:.0f}ms"
        )
        
        return new_articles
    
    async def _fetch_rss(
        self, session: aiohttp.ClientSession, url: str
    ) -> list[NewsArticle]:
        """Fetch and parse a single RSS feed."""
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"RSS feed {url} returned {resp.status}")
                    return []
                text = await resp.text()
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return []
        
        articles = []
        try:
            root = ET.fromstring(text)
            
            # Handle both RSS 2.0 and Atom feeds
            items = root.findall(".//item")
            if not items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                items = root.findall(".//atom:entry", ns)
            
            for item in items:
                title = self._get_text(item, "title")
                if not title:
                    continue
                
                description = (
                    self._get_text(item, "description")
                    or self._get_text(item, "summary")
                    or self._get_text(item, "{http://www.w3.org/2005/Atom}summary")
                    or ""
                )
                # Strip HTML tags from description
                description = self._strip_html(description)
                
                link = (
                    self._get_text(item, "link")
                    or self._get_attr(item, "link", "href")
                    or ""
                )
                
                pub_date_str = (
                    self._get_text(item, "pubDate")
                    or self._get_text(item, "published")
                    or self._get_text(item, "{http://www.w3.org/2005/Atom}published")
                )
                published_at = _parse_rss_date(pub_date_str) if pub_date_str else None
                
                categories = [
                    cat.text for cat in item.findall("category") if cat.text
                ]
                
                article_id = _generate_article_id(title, url)
                articles.append(NewsArticle(
                    id=article_id,
                    title=title.strip(),
                    description=description[:500],
                    source=url,
                    published_at=published_at,
                    url=link,
                    categories=categories,
                ))
        except ET.ParseError as e:
            logger.warning(f"XML parse error for {url}: {e}")
        
        return articles
    
    @staticmethod
    def _get_text(element, tag: str) -> Optional[str]:
        el = element.find(tag)
        return el.text if el is not None and el.text else None
    
    @staticmethod
    def _get_attr(element, tag: str, attr: str) -> Optional[str]:
        el = element.find(tag)
        return el.get(attr) if el is not None else None
    
    @staticmethod
    def _strip_html(text: str) -> str:
        """Simple HTML tag removal."""
        import re
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
    
    def get_latency_report(self) -> dict:
        """Return average ingestion latency per source."""
        report = {}
        for source, latencies in self._latency_stats.items():
            if latencies:
                report[source] = {
                    "avg_ms": round(sum(latencies) / len(latencies), 1),
                    "min_ms": round(min(latencies), 1),
                    "max_ms": round(max(latencies), 1),
                    "count": len(latencies),
                }
        return report
    
    def reset_seen(self):
        """Clear the dedup cache (useful for long-running agents)."""
        self._seen_ids.clear()
