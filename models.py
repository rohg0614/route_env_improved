# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the route dispatch optimization environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class RouteAction(Action):
    """One action in the driver dispatch environment."""

    action_type: Literal["wait", "accept_ride", "reposition"] = Field(
        ...,
        description="Type of action to execute.",
    )
    ride_id: int | None = Field(
        default=None,
        description="Ride identifier when action_type is accept_ride.",
    )
    target_node: int | None = Field(
        default=None,
        description="Adjacent node identifier when action_type is reposition.",
    )


class RouteObservation(Observation):
    """Observation returned after each simulation step."""

    task_name: Literal["easy", "medium", "hard"] = Field(
        ..., description="Current task difficulty."
    )
    current_node: int = Field(..., description="Current node ID for the driver.")
    time_of_day_sin: float = Field(..., description="Sine encoding of hour-of-day.")
    time_of_day_cos: float = Field(..., description="Cosine encoding of hour-of-day.")
    driver_status: Literal["idle", "busy", "en_route"] = Field(
        ...,
        description="Driver state in the simulator.",
    )
    shift_hours_remaining: float = Field(..., description="Remaining legal shift hours.")
    live_demand_matrix: list[float] = Field(
        default_factory=list,
        description="Current demand intensity by node.",
    )
    available_rides: list[dict] = Field(
        default_factory=list,
        description="Rides available at current node.",
    )
    current_zone: str = Field(
        default="Unknown",
        description=(
            "Human-readable name of the driver's current zone "
            "(e.g. 'Airport', 'Downtown', 'Hospital'). "
            "Use this for spatial reasoning about ride demand patterns."
        ),
    )
    supply_pressure: list[float] = Field(
        default_factory=list,
        description=(
            "Fraction of competitor drivers at each zone (sums to 1.0). "
            "High value at a zone means many competitors are there — "
            "fewer rides will be available to you. "
            "Avoid zones where supply_pressure is high relative to live_demand_matrix."
        ),
    )
    adjacent_nodes: list[int] = Field(
        default_factory=list,
        description=(
            "Node IDs directly reachable from current_node. "
            "Only these are valid targets for a reposition action. "
            "Includes both ring neighbours and chord shortcuts."
        ),
    )
    last_action_error: str | None = Field(
        default=None,
        description="Raw error for invalid or impossible action.",
    )
    normalized_progress_score: float = Field(
        default=0.0,
        description="Task grader score in [0,1] based on current trajectory.",
    )
