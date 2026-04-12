"""Task presets for the route dispatch benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    horizon_steps: int
    node_count: int
    max_shift_hours: float
    base_lambda: float
    lateness_budget: float
    distance_scale: float
    zone_names: tuple[str, ...] = ()
    zone_edges: tuple[tuple[int, int], ...] = ()
    zone_demand_weights: tuple[float, ...] = ()


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        horizon_steps=60,
        node_count=6,
        max_shift_hours=8.0,
        base_lambda=1.2,
        lateness_budget=0.45,
        distance_scale=8.0,
        zone_names=("Airport", "Downtown", "University", "Hospital", "Stadium", "Suburbs"),
        zone_edges=((0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,2),(1,4)),
        zone_demand_weights=(2.0, 1.8, 1.2, 1.0, 1.5, 0.8),
    ),
    "medium": TaskConfig(
        name="medium",
        horizon_steps=84,
        node_count=8,
        max_shift_hours=9.0,
        base_lambda=1.8,
        lateness_budget=0.32,
        distance_scale=10.0,
        zone_names=("Airport", "Downtown", "Financial", "University", "Hospital", "Stadium", "Suburbs", "Mall"),
        zone_edges=((0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,3),(1,5),(2,6),(4,7)),
        zone_demand_weights=(2.0, 1.8, 1.6, 1.2, 1.0, 1.5, 0.7, 1.1),
    ),
    "hard": TaskConfig(
        name="hard",
        horizon_steps=120,
        node_count=12,
        max_shift_hours=10.0,
        base_lambda=3.2,
        lateness_budget=0.15,
        distance_scale=14.0,
        zone_names=(
            "Airport", "Terminal2", "Downtown", "Financial", "Midtown",
            "University", "Hospital", "Stadium", "Convention",
            "Suburbs_N", "Suburbs_S", "Mall"
        ),
        zone_edges=(
            (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,0),
            (0,2),(2,5),(3,7),(4,8),(6,10),(1,9),(5,11)
        ),
        zone_demand_weights=(2.2, 1.9, 2.0, 1.7, 1.4, 1.2, 1.0, 1.8, 1.3, 0.7, 0.6, 1.1),
    ),
}

TASK_ORDER = ["easy", "medium", "hard"]