from .base import Policy, combine
from .scripted import ScriptedPolicy
from .laya import LayaPolicy
from .remote import JevPolicy, LunaPolicy, make_policy

__all__ = ["Policy", "combine", "ScriptedPolicy", "LayaPolicy", "JevPolicy", "LunaPolicy",
           "make_policy", "build"]


def build(spec: str, **kw) -> Policy:
    """Build any policy from a CLI/web spec string."""
    if spec in ("scripted", "heuristic", "baseline"):
        return ScriptedPolicy(**{k: v for k, v in kw.items() if k in ("dead_zone",)})
    if spec in ("laya", "local"):
        return LayaPolicy(**{k: v for k, v in kw.items()
                             if k in ("exe", "model", "threshold", "state_format", "timeout_ms",
                                      "device", "questions_design", "fire_mode")})
    return make_policy(spec, **kw)