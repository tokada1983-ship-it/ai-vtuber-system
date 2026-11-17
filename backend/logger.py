"""構造化ロガー設定"""
from __future__ import annotations

import logging

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


class _FallbackLogger:
    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def info(self, msg: str, **kwargs):
        self._log.info("%s %s", msg, kwargs if kwargs else "")

    def warning(self, msg: str, **kwargs):
        self._log.warning("%s %s", msg, kwargs if kwargs else "")

    def error(self, msg: str, **kwargs):
        self._log.error("%s %s", msg, kwargs if kwargs else "")

    def debug(self, msg: str, **kwargs):
        self._log.debug("%s %s", msg, kwargs if kwargs else "")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )


def get_logger(name: str):
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _FallbackLogger(name)

