"""Geolocalizer metrics."""

from __future__ import annotations

from sentinel.common.types import GeoObservation


def compute_geolocalizer_metrics(observations: list[GeoObservation]) -> dict:
    n = len(observations)
    if n == 0:
        return {
            "observation_count": 0,
            "classes": {},
            "average_confidence": 0.0,
            "average_accuracy_estimate_m": 0.0,
            "methods": {},
        }

    classes: dict[str, int] = {}
    methods: dict[str, int] = {}
    total_conf = 0.0
    total_acc = 0.0

    for obs in observations:
        classes[obs.class_name] = classes.get(obs.class_name, 0) + 1
        methods[obs.method] = methods.get(obs.method, 0) + 1
        total_conf += obs.confidence
        total_acc += obs.accuracy_estimate_m

    return {
        "observation_count": n,
        "classes": classes,
        "average_confidence": round(total_conf / n, 4),
        "average_accuracy_estimate_m": round(total_acc / n, 2),
        "methods": methods,
    }
