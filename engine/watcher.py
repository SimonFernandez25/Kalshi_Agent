"""
Market watcher loop.

Polls Kalshi for current market price at regular intervals.
Triggers when price >= threshold or timeout is reached.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List

from api.kalshi_client import KalshiClient
from config import WATCHER_POLL_INTERVAL_SEC, WATCHER_TIMEOUT_SEC
from schemas import WatcherTick

logger = logging.getLogger(__name__)


def watch_market(
    client: KalshiClient,
    market_id: str,
    threshold: float,
    poll_interval: int = WATCHER_POLL_INTERVAL_SEC,
    timeout: int = WATCHER_TIMEOUT_SEC,
    max_ticks: int = 5,
) -> List[WatcherTick]:
    """
    Poll market price until threshold is hit or timeout.

    For V1, we cap at max_ticks to keep runs fast.
    In production, remove the cap and let timeout govern.

    Args:
        client: KalshiClient instance.
        market_id: Market to poll.
        threshold: Price threshold that triggers bet.
        poll_interval: Seconds between polls.
        timeout: Max seconds before giving up.
        max_ticks: Max number of polls (V1 safety cap).

    Returns:
        List of WatcherTick records.
    """
    ticks: List[WatcherTick] = []
    start = time.monotonic()
    tick_count = 0

    logger.info(
        "Watcher started: market=%s threshold=%.4f interval=%ds timeout=%ds",
        market_id,
        threshold,
        poll_interval,
        timeout,
    )

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            logger.info("Watcher timeout after %.1fs", elapsed)
            break

        if tick_count >= max_ticks:
            logger.info("Watcher hit max_ticks=%d — stopping early.", max_ticks)
            break

        price = client.get_market_price(market_id)
        triggered = price >= threshold

        tick = WatcherTick(
            market_id=market_id,
            polled_price=price,
            threshold=threshold,
            triggered=triggered,
        )
        ticks.append(tick)
        tick_count += 1

        logger.info(
            "  Tick %d: price=%.4f threshold=%.4f triggered=%s",
            tick_count,
            price,
            threshold,
            triggered,
        )

        if triggered:
            logger.info("Threshold hit! Watcher stopping.")
            break

        if tick_count < max_ticks and (time.monotonic() - start + poll_interval) < timeout:
            time.sleep(poll_interval)

    return ticks
