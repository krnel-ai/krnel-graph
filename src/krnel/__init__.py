# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from krnel import graph
from krnel.graph import OpSpec
from krnel.runners import LocalArrowRunner, ModelProvider, MaterializedResult

from krnel.logging import get_logger


def main():
    from krnel import cli
    cli.app()