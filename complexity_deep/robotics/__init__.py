"""
Complexity Robotics Module
==========================

Adapts Complexity model for real-time robotics control.

Key features:
- Multi-modal tokenization (vision, proprio, actions)
- Real-time inference optimized for edge devices
- Action decoding to motor commands
- Control loop integration

Token-Routed MLP is particularly suited for robotics:
- Expert 0: Vision features (high frequency tokens)
- Expert 1: Proprioception (joint positions, velocities)
- Expert 2: Action history (previous commands)
- Expert 3: Language/goals (instructions)

Author: Pacific Prime
"""

from .tokenizer import (
    RoboticsTokenizer,
    VisionEncoder,
    ProprioEncoder,
    ActionEncoder,
)

from .model import (
    RoboticsComplexity,
    RoboticsConfig,
)

from .inference import (
    RealtimeInference,
    ActionDecoder,
    ControlLoop,
)

from .simulation import (
    SimulationInterface,
    DummyRobot,
)

__all__ = [
    # Tokenization
    "RoboticsTokenizer",
    "VisionEncoder",
    "ProprioEncoder",
    "ActionEncoder",

    # Model
    "RoboticsComplexity",
    "RoboticsConfig",

    # Inference
    "RealtimeInference",
    "ActionDecoder",
    "ControlLoop",

    # Simulation
    "SimulationInterface",
    "DummyRobot",
]
