"""
Real-Time Inference for Robotics
================================

Optimized inference pipeline for robotics control:
- Async preprocessing
- Batched inference
- Action smoothing
- Latency monitoring

Target: <20ms end-to-end latency on edge GPU

Author: Pacific Prime
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Callable, List
from dataclasses import dataclass
import time
import threading
import queue
from collections import deque

try:
    from ..cuda import HAS_TRITON
    from ..cuda.quantization import quantize_model, Int8TokenRoutedMLP
    QUANTIZATION_AVAILABLE = HAS_TRITON
except ImportError:
    QUANTIZATION_AVAILABLE = False


@dataclass
class InferenceConfig:
    """Configuration for real-time inference."""
    max_batch_size: int = 1
    use_fp16: bool = True
    use_int8: bool = False  # INT8 quantization for faster inference
    use_cuda_graphs: bool = True  # CUDA graphs for reduced overhead
    warmup_iterations: int = 10
    action_smoothing: float = 0.3  # EMA smoothing factor
    max_latency_ms: float = 20.0  # Target latency


class LatencyMonitor:
    """Monitor inference latency statistics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.start_time = None

    def start(self):
        """Start timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timing and return latency in ms."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - self.start_time) * 1000
        self.latencies.append(latency)
        return latency

    @property
    def mean_latency(self) -> float:
        """Mean latency in ms."""
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0

    @property
    def max_latency(self) -> float:
        """Max latency in ms."""
        return max(self.latencies) if self.latencies else 0

    @property
    def p99_latency(self) -> float:
        """99th percentile latency."""
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[idx]


class ActionDecoder:
    """
    Decode model outputs to robot actions.

    Features:
    - Discrete to continuous conversion
    - Action smoothing (EMA)
    - Safety bounds
    - Gripper hysteresis
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        smoothing_factor: float = 0.3,
        gripper_threshold: float = 0.5,
    ):
        self.action_dim = action_dim
        self.smoothing_factor = smoothing_factor
        self.gripper_threshold = gripper_threshold

        # Action bounds [action_dim] for min and max
        if action_bounds is not None:
            self.action_min, self.action_max = action_bounds
        else:
            self.action_min = torch.full((action_dim,), -1.0)
            self.action_max = torch.full((action_dim,), 1.0)

        # Previous action for smoothing
        self.prev_action = None

        # Gripper state for hysteresis
        self.gripper_open = True

    def decode(
        self,
        actions: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        apply_smoothing: bool = True,
    ) -> torch.Tensor:
        """
        Decode and post-process actions.

        Args:
            actions: [batch, horizon, action_dim] raw actions in [-1, 1]
            uncertainty: Optional uncertainty estimates
            apply_smoothing: Whether to apply EMA smoothing

        Returns:
            processed_actions: [batch, horizon, action_dim] processed actions
        """
        # Take first action from horizon (or use uncertainty-weighted average)
        if uncertainty is not None and actions.shape[1] > 1:
            # Weight actions by inverse uncertainty
            weights = 1.0 / (uncertainty + 1e-6)
            weights = weights / weights.sum(dim=1, keepdim=True)
            action = (actions * weights).sum(dim=1)
        else:
            action = actions[:, 0, :]  # [batch, action_dim]

        # Apply smoothing
        if apply_smoothing and self.prev_action is not None:
            action = self.smoothing_factor * action + (1 - self.smoothing_factor) * self.prev_action

        # Apply bounds
        action = torch.clamp(action, self.action_min.to(action.device), self.action_max.to(action.device))

        # Gripper hysteresis (prevent oscillation)
        if self.action_dim > 6:  # Assuming last dim is gripper
            gripper = action[:, -1]
            if self.gripper_open and gripper < -self.gripper_threshold:
                self.gripper_open = False
            elif not self.gripper_open and gripper > self.gripper_threshold:
                self.gripper_open = True
            action[:, -1] = 1.0 if self.gripper_open else -1.0

        self.prev_action = action.detach()
        return action

    def reset(self):
        """Reset decoder state."""
        self.prev_action = None
        self.gripper_open = True


class RealtimeInference:
    """
    Real-time inference engine for robotics.

    Features:
    - Model optimization (FP16, INT8, CUDA graphs)
    - Latency monitoring
    - Warmup for consistent performance
    - Thread-safe inference
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[InferenceConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or InferenceConfig()
        self.device = torch.device(device)
        self.tokenizer = tokenizer

        # Prepare model
        self.model = self._prepare_model(model)

        # Monitoring
        self.latency_monitor = LatencyMonitor()

        # Action decoder
        self.action_decoder = ActionDecoder(
            smoothing_factor=self.config.action_smoothing,
        )

        # CUDA graph for static input shapes
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None

        # Thread safety
        self.lock = threading.Lock()

        # Warmup
        self._warmup()

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for inference."""
        model = model.to(self.device)
        model.eval()

        # FP16
        if self.config.use_fp16:
            model = model.half()

        # INT8 quantization
        if self.config.use_int8 and QUANTIZATION_AVAILABLE:
            print("Applying INT8 quantization...")
            model = quantize_model(model)

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _warmup(self):
        """Warmup model for consistent performance."""
        print(f"Warming up model ({self.config.warmup_iterations} iterations)...")

        # Create dummy input
        dummy_tokens = torch.randint(
            0, 100000,
            (1, 256),
            device=self.device,
        )

        # Warmup iterations
        for i in range(self.config.warmup_iterations):
            with torch.no_grad():
                if self.config.use_fp16:
                    with torch.cuda.amp.autocast():
                        _ = self.model(dummy_tokens)
                else:
                    _ = self.model(dummy_tokens)

        # Setup CUDA graph if enabled
        if self.config.use_cuda_graphs and torch.cuda.is_available():
            self._setup_cuda_graph(dummy_tokens.shape)

        print(f"Warmup complete. Ready for inference.")

    def _setup_cuda_graph(self, input_shape: tuple):
        """Setup CUDA graph for reduced kernel launch overhead."""
        print("Setting up CUDA graph...")

        # Static input buffer
        self.static_input = torch.zeros(
            input_shape,
            dtype=torch.long,
            device=self.device,
        )

        # Warmup for graph capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    if self.config.use_fp16:
                        with torch.cuda.amp.autocast():
                            self.static_output = self.model(self.static_input)
                    else:
                        self.static_output = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            with torch.no_grad():
                if self.config.use_fp16:
                    with torch.cuda.amp.autocast():
                        self.static_output = self.model(self.static_input)
                else:
                    self.static_output = self.model(self.static_input)

        print("CUDA graph ready.")

    @torch.no_grad()
    def infer(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Run inference on observation.

        Args:
            observation: Dict with 'image', 'proprio', 'prev_actions'

        Returns:
            action: [action_dim] decoded action
            stats: Dict with latency stats
        """
        with self.lock:
            self.latency_monitor.start()

            # Tokenize
            tokens = self.tokenizer.tokenize_observation(
                image=observation.get("image"),
                proprio=observation.get("proprio"),
                prev_actions=observation.get("prev_actions"),
            )

            input_ids = tokens["token_ids"].to(self.device)

            # Use CUDA graph if available and input matches
            if (self.cuda_graph is not None and
                input_ids.shape == self.static_input.shape):
                self.static_input.copy_(input_ids)
                self.cuda_graph.replay()
                outputs = self.static_output
            else:
                # Standard inference
                if self.config.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, return_continuous_actions=True)
                else:
                    outputs = self.model(input_ids, return_continuous_actions=True)

            # Decode action
            action = self.action_decoder.decode(
                outputs["actions"],
                outputs.get("uncertainty"),
            )

            latency = self.latency_monitor.stop()

            stats = {
                "latency_ms": latency,
                "mean_latency_ms": self.latency_monitor.mean_latency,
                "max_latency_ms": self.latency_monitor.max_latency,
                "p99_latency_ms": self.latency_monitor.p99_latency,
            }

            # Warn if exceeding target latency
            if latency > self.config.max_latency_ms:
                print(f"WARNING: Latency {latency:.1f}ms exceeds target {self.config.max_latency_ms}ms")

            return action.squeeze(0), stats

    def reset(self):
        """Reset inference state."""
        self.action_decoder.reset()


class ControlLoop:
    """
    Main control loop for robotics.

    Runs at fixed frequency, handles:
    - Observation collection
    - Model inference
    - Action execution
    - Timing synchronization
    """

    def __init__(
        self,
        inference_engine: RealtimeInference,
        robot_interface,  # Robot-specific interface
        control_freq: int = 50,  # Hz
    ):
        self.inference = inference_engine
        self.robot = robot_interface
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        self.running = False
        self.step_count = 0

    def step(self) -> Dict:
        """
        Single control step.

        Returns:
            info: Dict with action, observation, timing stats
        """
        step_start = time.perf_counter()

        # Get observation from robot
        observation = self.robot.get_observation()

        # Run inference
        action, stats = self.inference.infer(observation)

        # Execute action
        self.robot.execute_action(action.cpu().numpy())

        # Timing
        step_time = time.perf_counter() - step_start
        sleep_time = max(0, self.dt - step_time)

        if sleep_time > 0:
            time.sleep(sleep_time)

        self.step_count += 1

        return {
            "action": action,
            "observation": observation,
            "step_time_ms": step_time * 1000,
            "inference_stats": stats,
        }

    def run(self, max_steps: Optional[int] = None):
        """
        Run control loop.

        Args:
            max_steps: Maximum steps (None for infinite)
        """
        self.running = True
        self.step_count = 0

        print(f"Starting control loop at {self.control_freq}Hz...")

        try:
            while self.running:
                info = self.step()

                if self.step_count % 100 == 0:
                    print(f"Step {self.step_count}: "
                          f"inference={info['inference_stats']['latency_ms']:.1f}ms, "
                          f"total={info['step_time_ms']:.1f}ms")

                if max_steps is not None and self.step_count >= max_steps:
                    break

        except KeyboardInterrupt:
            print("\nControl loop interrupted.")

        finally:
            self.running = False
            print(f"Control loop stopped after {self.step_count} steps.")

    def stop(self):
        """Stop the control loop."""
        self.running = False
