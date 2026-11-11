"""Load IBM Quantum credentials from the environment or local ignored .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def get_ibm_quantum_token(token: Optional[str] = None) -> str:
    """Resolve an explicit token or IBM_QUANTUM_TOKEN without exposing its value."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(_ENV_PATH, override=False)

    return (token or os.getenv("IBM_QUANTUM_TOKEN", "")).strip()
