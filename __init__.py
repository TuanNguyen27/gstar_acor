"""
ACOR (Agentic, Self-Correcting Modular Reasoner) System

A novel architecture for enhancing reasoning reliability in language models
through dynamic, iterative refinement loops with process supervision.
"""

__version__ = "0.1.0"
__author__ = "ACOR Development Team"

from .modules import PlannerModule, ExecutorModule, CriticModule
from .orchestrator import ACORSystem

__all__ = [
    'PlannerModule',
    'ExecutorModule',
    'CriticModule',
    'ACORSystem'
]