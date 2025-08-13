# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import structlog
import logging

if not structlog.is_configured():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(sort_keys=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False
    )

def get_logger(rel: str | None = None) -> structlog.stdlib.BoundLogger:
    name = "krnel" if not rel else f"krnel.{rel}"
    return structlog.get_logger(name)