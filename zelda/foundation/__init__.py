"""Shared capability, peer-delegation and enrolled-device primitives.

Transport authentication (TLS plus device proof-of-possession) must happen before
calling DeviceRegistry.connect. This module deliberately does not trust a device
because it merely presents a device ID or public key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

PEER_IDS = frozenset({"zelda", "kosi", "candida"})
RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: str = "low"
    scopes: frozenset[str] = frozenset()
    idempotent: bool = False
    requires_confirmation: bool = False
    timeout_ms: int = 30000


@dataclass(frozen=True)
class ToolOutcome:
    request_id: str
    ok: bool
    data: Any = None
    error: dict[str, Any] | None = None


@dataclass
class _RegisteredTool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], Any]


class CapabilityRegistry:
    """Per-agent allowlist. No arbitrary function or shell execution is exposed."""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: Callable[[dict[str, Any]], Any]) -> None:
        if not spec.name or "." not in spec.name:
            raise ValueError("tool name must be namespaced")
        if spec.risk not in RISK_LEVELS:
            raise ValueError("unsupported risk level")
        if not isinstance(spec.input_schema, dict) or spec.input_schema.get("type") != "object":
            raise ValueError("input_schema must describe an object")
        if spec.timeout_ms < 100 or spec.timeout_ms > 300000:
            raise ValueError("timeout_ms must be between 100 and 300000")
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = _RegisteredTool(spec, handler)

    def manifest(self, agent_id: str, runtime_version: str = "dev") -> dict[str, Any]:
        if agent_id not in PEER_IDS:
            raise ValueError("unknown peer agent")
        return {
            "protocolVersion": "1.0",
            "agentId": agent_id,
            "runtimeVersion": runtime_version,
            "tools": [{
                "name": t.spec.name, "description": t.spec.description,
                "inputSchema": t.spec.input_schema, "risk": t.spec.risk,
                "scopes": sorted(t.spec.scopes), "idempotent": t.spec.idempotent,
                "requiresConfirmation": t.spec.requires_confirmation,
                "timeoutMs": t.spec.timeout_ms,
            } for t in self._tools.values()],
        }

    def invoke(
        self, name: str, arguments: dict[str, Any], granted_scopes: set[str],
        *, confirmed: bool = False, request_id: str | None = None,
    ) -> ToolOutcome:
        request_id = request_id or str(uuid4())
        entry = self._tools.get(name)
        if entry is None:
            return ToolOutcome(request_id, False, error={"code": "tool_not_found", "message": name})
        spec = entry.spec
        if not spec.scopes.issubset(granted_scopes):
            return ToolOutcome(request_id, False, error={"code": "scope_denied", "message": "Required scopes not granted"})
        if (spec.requires_confirmation or spec.risk in {"high", "critical"}) and not confirmed:
            return ToolOutcome(request_id, False, error={"code": "confirmation_required", "message": "Explicit confirmation required"})
        try:
            return ToolOutcome(request_id, True, data=entry.handler(arguments))
        except Exception as exc:
            return ToolOutcome(request_id, False, error={"code": "tool_execution_failed", "message": str(exc)[:1000]})


@dataclass(frozen=True)
class Delegation:
    delegation_id: str
    from_agent: str
    to_agent: str
    objective: str
    allowed_tools: frozenset[str]
    allowed_data: frozenset[str]
    scopes: frozenset[str]
    expires_at: datetime
    depth: int = 0
    parent_id: str | None = None
    max_tool_calls: int = 20
    max_duration_ms: int = 300000


class DelegationPolicy:
    """Validates peer-to-peer work grants; a peer never inherits broader authority."""

    def validate(self, item: Delegation, parent: Delegation | None = None) -> None:
        if item.from_agent not in PEER_IDS or item.to_agent not in PEER_IDS:
            raise PermissionError("unknown_agent")
        if item.from_agent == item.to_agent:
            raise PermissionError("self_delegation_not_allowed")
        if not item.objective.strip() or len(item.objective) > 12000:
            raise ValueError("invalid_objective")
        if item.depth > 3 or item.max_tool_calls < 1 or item.max_tool_calls > 100:
            raise PermissionError("delegation_budget_exceeded")
        if item.max_duration_ms < 1000 or item.max_duration_ms > 3600000:
            raise PermissionError("delegation_duration_exceeded")
        if item.expires_at.tzinfo is None or item.expires_at <= datetime.now(timezone.utc):
            raise PermissionError("delegation_expired_or_timezone_missing")
        if parent is not None:
            if item.parent_id != parent.delegation_id or item.depth != parent.depth + 1:
                raise PermissionError("delegation_chain_invalid")
            if not item.allowed_tools.issubset(parent.allowed_tools):
                raise PermissionError("delegated_tool_scope_escalation")
            if not item.allowed_data.issubset(parent.allowed_data):
                raise PermissionError("delegated_data_scope_escalation")
            if not item.scopes.issubset(parent.scopes):
                raise PermissionError("delegated_permission_escalation")


@dataclass(frozen=True)
class DeviceSession:
    device_id: str
    session_id: str
    device_type: str
    capabilities: frozenset[str]
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DeviceRegistry:
    """Tracks enrolled devices only after the transport authenticates them."""

    def __init__(self, enrolled_device_ids: set[str] | None = None) -> None:
        self._enrolled = set(enrolled_device_ids or ())
        self._sessions: dict[str, DeviceSession] = {}

    def enroll(self, device_id: str) -> None:
        if not device_id:
            raise ValueError("device_id required")
        self._enrolled.add(device_id)

    def connect(
        self, device_id: str, device_type: str, capabilities: set[str],
        *, transport_authenticated: bool,
    ) -> DeviceSession:
        if not transport_authenticated:
            raise PermissionError("device_transport_not_authenticated")
        if device_id not in self._enrolled:
            raise PermissionError("device_not_enrolled")
        session = DeviceSession(device_id, str(uuid4()), device_type, frozenset(capabilities))
        self._sessions[device_id] = session
        return session

    def heartbeat(self, device_id: str, session_id: str) -> DeviceSession:
        current = self._sessions.get(device_id)
        if current is None or current.session_id != session_id:
            raise PermissionError("invalid_device_session")
        updated = DeviceSession(
            current.device_id, current.session_id, current.device_type,
            current.capabilities, current.connected_at, datetime.now(timezone.utc),
        )
        self._sessions[device_id] = updated
        return updated

    def disconnect(self, device_id: str, session_id: str) -> None:
        current = self._sessions.get(device_id)
        if current and current.session_id == session_id:
            del self._sessions[device_id]
