"""ROS-style in-process autonomy stack for state-only Flightmare evaluation."""

__all__ = [
    "BaseControllerNode",
    "ControlCommand",
    "CourseConfig",
    "EpisodeResult",
    "FlightmareAutonomyGraph",
    "FlightmareStateNode",
    "MissionWorldModelNode",
    "PlannerOutput",
    "PolicyPlannerNode",
    "VehicleState",
]


_GRAPH_EXPORTS = {"EpisodeResult", "FlightmareAutonomyGraph"}
_MESSAGE_EXPORTS = {"ControlCommand", "PlannerOutput", "VehicleState"}
_NODE_EXPORTS = {
    "BaseControllerNode",
    "CourseConfig",
    "FlightmareStateNode",
    "MissionWorldModelNode",
    "PolicyPlannerNode",
}


def __getattr__(name: str):
    if name in _GRAPH_EXPORTS:
        from src.robotics.flightmare_autonomy_fsw import graph

        value = getattr(graph, name)
    elif name in _MESSAGE_EXPORTS:
        from src.robotics.flightmare_autonomy_fsw import messages

        value = getattr(messages, name)
    elif name in _NODE_EXPORTS:
        from src.robotics.flightmare_autonomy_fsw import nodes

        value = getattr(nodes, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
