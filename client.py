# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Route Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import RouteAction, RouteObservation


class RouteEnv(
    EnvClient[RouteAction, RouteObservation, State]
):
    """
    Client for the Route Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with RouteEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(RouteAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = RouteEnv.from_docker_image("route_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(RouteAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: RouteAction) -> Dict:
        """
        Convert RouteAction to JSON payload for step message.

        Args:
            action: RouteAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "ride_id": action.ride_id,
            "target_node": action.target_node,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RouteObservation]:
        """
        Parse server response into StepResult[RouteObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with RouteObservation
        """
        obs_data = payload.get("observation", {})
        observation = RouteObservation(
            task_name=obs_data.get("task_name", "easy"),
            current_node=obs_data.get("current_node", 0),
            time_of_day_sin=obs_data.get("time_of_day_sin", 0.0),
            time_of_day_cos=obs_data.get("time_of_day_cos", 1.0),
            driver_status=obs_data.get("driver_status", "idle"),
            shift_hours_remaining=obs_data.get("shift_hours_remaining", 0.0),
            live_demand_matrix=obs_data.get("live_demand_matrix", []),
            available_rides=obs_data.get("available_rides", []),
            current_zone=obs_data.get("current_zone", "Unknown"),
            supply_pressure=obs_data.get("supply_pressure", []),
            adjacent_nodes=obs_data.get("adjacent_nodes", []),
            last_action_error=obs_data.get("last_action_error"),
            normalized_progress_score=obs_data.get("normalized_progress_score", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
