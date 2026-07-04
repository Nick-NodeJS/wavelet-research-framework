"""Service entry point."""

from __future__ import annotations

import logging
import sys

from wavelet_research.service.app import create_app
from wavelet_research.service.config import ServiceConfig

logger = logging.getLogger(__name__)


def main() -> int:
    """Start the Wavelet Service.

    Returns
    -------
    int
        Exit code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = ServiceConfig.from_env()
    app = create_app(config)

    logger.info(
        "Starting Wavelet Service on %s:%d (debug=%s)",
        config.host, config.port, config.debug,
    )

    app.run(host=config.host, port=config.port, debug=config.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
