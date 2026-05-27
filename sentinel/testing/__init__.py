"""Sentinel simulation testing infrastructure.

Provides stack lifecycle management, auto-pilot helpers, metrics collection,
and report generation for autonomous simulation test suites.
"""

from .auto_pilot import AutoPilot
from .metrics import SimMetrics
from .reporter import SimReporter
from .stack_launcher import SimStackLauncher

__all__ = [
    "AutoPilot",
    "SimMetrics",
    "SimReporter",
    "SimStackLauncher",
]
