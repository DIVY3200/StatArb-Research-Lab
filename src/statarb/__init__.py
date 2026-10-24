"""StatArb Research Lab: reproducible research tools for cointegration-based pairs trading.

This package is for research and education. It does not execute live trades and its
outputs are not financial advice.
"""

from .config import ResearchConfig, load_config

__version__ = "0.1.0"

__all__ = ["ResearchConfig", "__version__", "load_config"]
