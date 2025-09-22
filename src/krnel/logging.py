# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import logging
import os

import structlog


def configure_logging(log_level: str | None = None, force_reconfigure: bool = False):
    if not structlog.is_configured() or force_reconfigure:
        log_level = log_level or os.getenv("KRNEL_LOG_LEVEL", "INFO").upper()
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.CallsiteParameterAdder(
                    {
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    }
                ),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
                structlog.dev.ConsoleRenderer(sort_keys=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                min_level=getattr(logging, log_level)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )


configure_logging()


def get_logger(rel: str | None = None) -> structlog.stdlib.BoundLogger:
    name = "krnel" if not rel else f"krnel.{rel}"
    return structlog.get_logger(name)
