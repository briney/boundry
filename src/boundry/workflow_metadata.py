"""Helpers for workflow metadata merging and metric resolution."""

from __future__ import annotations

from typing import Any, Dict, Optional

WORKFLOW_NAMESPACE = "_workflow"


def merge_metadata(
    previous: Dict[str, Any],
    new_values: Dict[str, Any],
    operation: Optional[str] = None,
) -> Dict[str, Any]:
    """Shallow-merge metadata and maintain workflow namespace helpers."""
    merged = dict(previous)
    merged.update(new_values)

    workflow_ns = dict(merged.get(WORKFLOW_NAMESPACE) or {})
    metrics = dict(workflow_ns.get("metrics") or {})
    state = dict(workflow_ns.get("state") or {})
    provenance = dict(workflow_ns.get("provenance") or {})

    _collect_numeric_values(new_values, prefix="", target=metrics)

    if operation is not None:
        state["last_operation"] = operation
        history = list(provenance.get("operations") or [])
        history.append(operation)
        provenance["operations"] = history

    workflow_ns["metrics"] = metrics
    workflow_ns["state"] = state
    workflow_ns["provenance"] = provenance
    merged[WORKFLOW_NAMESPACE] = workflow_ns
    return merged


def extract_numeric_metric(
    metadata: Dict[str, Any], path: str
) -> Optional[float]:
    """Extract a numeric metric from metadata by dotted path."""
    value = resolve_path(metadata, path)
    if _is_numeric(value):
        return float(value)

    workflow_ns = metadata.get(WORKFLOW_NAMESPACE)
    if isinstance(workflow_ns, dict):
        metrics = workflow_ns.get("metrics")
        if isinstance(metrics, dict):
            alt = metrics.get(path)
            if _is_numeric(alt):
                return float(alt)
    return None


def resolve_path(root: Any, path: str) -> Any:
    """Resolve a dotted path through nested dicts and public attributes."""
    current = root
    for segment in path.split("."):
        if not segment or segment.startswith("_"):
            return None
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
            continue
        if not hasattr(current, segment):
            return None
        current = getattr(current, segment)
    return current


def _collect_numeric_values(
    data: Any, prefix: str, target: Dict[str, float]
) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith("_"):
                continue
            path = f"{prefix}.{key}" if prefix else key
            _collect_numeric_values(value, path, target)
        return
    if _is_numeric(data) and prefix:
        target[prefix] = float(data)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
