# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import structlog

#structlog.configure(
#    processors=[
#        structlog.processors.KeyValueRenderer(key_order=["event", "level", "logger"]),
#    ]
#)
log = structlog.get_logger("krnel")
