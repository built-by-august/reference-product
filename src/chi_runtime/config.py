"""Structured configuration for the Chimeric runtime.

Config is loaded from (in increasing precedence):

  1. sensible built-in defaults (see ``Config`` / the section dataclasses),
  2. an optional YAML file (``CHI_CONFIG_PATH`` or ``config.yaml`` next to cwd),
  3. environment variables (``CHI_*``, nested via ``__`` e.g. ``CHI_RETRY__MAX_ATTEMPTS``).

Everything is validated into a single frozen ``Config`` object so the rest of the
runtime can read it without guessing where a value came from. The retry/backoff,
streaming, and model-provider layers all read their knobs from here.

This module has zero hard dependencies: ``pyyaml`` is only imported lazily if a
YAML file is actually present, so the runtime stays importable and testable on a
bare install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

ENV_PREFIX = "CHI_"  # e.g. CHI_MODEL__MODEL, CHI_RETRY__MAX_ATTEMPTS

_SECTION_ALIASES = {
    "model": "model",
    "retry": "retry",
    "stream": "streaming",
    "streaming": "streaming",
    "obs": "observability",
    "observability": "observability",
}


@dataclass(frozen=True)
class RetryConfig:
    """Retry/backoff policy for model calls."""

    max_attempts: int = 3
    base_delay: float = 0.5  # seconds, doubled each attempt (exponential backoff)
    max_delay: float = 8.0  # cap on a single backoff wait
    backoff_factor: float = 2.0
    # Only these errors are retried; everything else fails fast.
    retry_on: tuple[str, ...] = ("timeout", "rate_limit", "server_error", "connection")

    def is_retryable(self, kind: str) -> bool:
        return kind in self.retry_on


@dataclass(frozen=True)
class ModelConfig:
    """Model provider selection + connection knobs."""

    provider: str = "stub"  # stub | openai | anthropic | openrouter
    model: str = "stub"
    temperature: float = 0.0
    api_base: str | None = None
    api_key_env: str | None = "CHI_API_KEY"  # name of the env var holding the key
    request_timeout: float = 60.0


@dataclass(frozen=True)
class StreamingConfig:
    """Streaming token-output behaviour for the harness and web layer."""

    enabled: bool = True  # stream tokens as they arrive

    def as_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


@dataclass(frozen=True)
class ObservabilityConfig:
    """What the dashboard / metrics layer records."""

    emit_metrics: bool = True  # record latency + token totals per run
    max_stored_runs: int = 100  # ring-buffer cap for the metrics store
    log_level: str = "INFO"

    def as_dict(self) -> dict[str, Any]:
        return {
            "emit_metrics": self.emit_metrics,
            "max_stored_runs": self.max_stored_runs,
            "log_level": self.log_level,
        }


@dataclass(frozen=True)
class Config:
    """The fully-resolved runtime configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    config_path: str | None = None  # which file (if any) the config was loaded from

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "provider": self.model.provider,
                "model": self.model.model,
                "temperature": self.model.temperature,
                "api_base": self.model.api_base,
                "request_timeout": self.model.request_timeout,
            },
            "retry": {
                "max_attempts": self.retry.max_attempts,
                "base_delay": self.retry.base_delay,
                "max_delay": self.retry.max_delay,
                "backoff_factor": self.retry.backoff_factor,
                "retry_on": list(self.retry.retry_on),
            },
            "streaming": self.streaming.as_dict(),
            "observability": self.observability.as_dict(),
            "config_path": self.config_path,
        }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _load_yaml(path: str) -> dict[str, Any]:
    """Read a YAML file. Raises a clear error if pyyaml is missing or the file is bad."""
    try:
        import yaml  # lazy import — kept optional so the runtime stays lean
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "CHI_CONFIG path points at a YAML file but PyYAML is not installed. "
            "Run `uv add pyyaml` or use env vars / JSON instead."
        ) from exc
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    return data


def _read_env() -> dict[str, Any]:
    """Collect CHI_-prefixed env vars into a nested config dict."""
    out: dict[str, Any] = {}
    for raw_key, raw_val in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        # CHI_MODEL__MODEL -> ["model","model"]; CHI_LOG_LEVEL -> ["log_level"]
        parts = raw_key[len(ENV_PREFIX) :].lower().split("__")
        if len(parts) == 1:
            out[parts[0]] = raw_val  # top-level scalar (rare; used as a hint)
            continue
        section = _SECTION_ALIASES.get(parts[0], parts[0])
        out.setdefault(section, {})[parts[1]] = raw_val
    return out


def _apply_env_strings(cfg: Config, env: dict[str, Any]) -> Config:
    """Apply raw env strings (correctly typed) on top of a constructed Config."""
    model = _env_section(env, "model")
    retry = _env_section(env, "retry")
    streaming = _env_section(env, "streaming")
    streaming.update(_env_section(env, "stream"))
    obs = _env_section(env, "observability")
    obs.update(_env_section(env, "obs"))

    new_model = ModelConfig(
        provider=model.get("provider", cfg.model.provider),
        model=model.get("model", cfg.model.model),
        temperature=_env_float(model, "temperature", cfg.model.temperature),
        api_base=model.get("api_base", cfg.model.api_base),
        api_key_env=model.get("api_key_env", cfg.model.api_key_env),
        request_timeout=_env_float(model, "request_timeout", cfg.model.request_timeout),
    )
    new_retry = RetryConfig(
        max_attempts=_env_int(retry, "max_attempts", cfg.retry.max_attempts),
        base_delay=_env_float(retry, "base_delay", cfg.retry.base_delay),
        max_delay=_env_float(retry, "max_delay", cfg.retry.max_delay),
        backoff_factor=_env_float(retry, "backoff_factor", cfg.retry.backoff_factor),
        retry_on=tuple(retry["retry_on"]) if "retry_on" in retry else cfg.retry.retry_on,
    )
    new_stream = StreamingConfig(
        enabled=_env_bool(streaming, "enabled", cfg.streaming.enabled),
    )
    new_obs = ObservabilityConfig(
        emit_metrics=_env_bool(obs, "emit_metrics", cfg.observability.emit_metrics),
        max_stored_runs=_env_int(obs, "max_stored_runs", cfg.observability.max_stored_runs),
        log_level=obs.get("log_level", cfg.observability.log_level),
    )
    return Config(
        model=new_model, retry=new_retry, streaming=new_stream,
        observability=new_obs, config_path=cfg.config_path,
    )


def _env_section(env: dict[str, Any], name: str) -> dict[str, Any]:
    value = env.get(name)
    return value if isinstance(value, dict) else {}


def _env_int(section: dict[str, Any], key: str, default: int) -> int:
    return int(section[key]) if key in section else default


def _env_float(section: dict[str, Any], key: str, default: float) -> float:
    return float(section[key]) if key in section else default


def _env_bool(section: dict[str, Any], key: str, default: bool) -> bool:
    return _coerce_bool(section[key]) if key in section else default


def _yaml_to_config(data: dict[str, Any], config_path: str) -> Config:
    """Build a Config from a parsed YAML mapping (with sane defaults fallback)."""

    def section(name: str) -> dict[str, Any]:
        value = data.get(name)
        return value if isinstance(value, dict) else {}

    m = section("model")
    r = section("retry")
    s = section("streaming")
    o = section("observability")

    return Config(
        model=ModelConfig(
            provider=m.get("provider", "stub"),
            model=m.get("model", m.get("model_name", "stub")),
            temperature=float(m.get("temperature", 0.0)),
            api_base=m.get("api_base"),
            api_key_env=m.get("api_key_env", "CHI_API_KEY"),
            request_timeout=float(m.get("request_timeout", 60.0)),
        ),
        retry=RetryConfig(
            max_attempts=int(r.get("max_attempts", 3)),
            base_delay=float(r.get("base_delay", 0.5)),
            max_delay=float(r.get("max_delay", 8.0)),
            backoff_factor=float(r.get("backoff_factor", 2.0)),
            retry_on=tuple(r.get("retry_on", RetryConfig().retry_on)),
        ),
        streaming=StreamingConfig(enabled=_coerce_bool(s.get("enabled", True))),
        observability=ObservabilityConfig(
            emit_metrics=_coerce_bool(o.get("emit_metrics", True)),
            max_stored_runs=int(o.get("max_stored_runs", 100)),
            log_level=o.get("log_level", "INFO"),
        ),
        config_path=config_path,
    )


def load_config(path: str | None = None) -> Config:
    """Build a ``Config`` from defaults, an optional YAML file, and env overrides.

    Resolution order: defaults -> YAML file -> environment variables.
    """
    if path is None:
        path = os.environ.get(f"{ENV_PREFIX}CONFIG_PATH")
    if path is None:
        for cand in (os.path.join(os.getcwd(), "config.yaml"),
                     os.path.join(os.getcwd(), "config.yml")):
            if os.path.exists(cand):
                path = cand
                break

    if path and os.path.exists(path):
        yaml_cfg = _yaml_to_config(_load_yaml(path), path)
    else:
        yaml_cfg = Config()

    return _apply_env_strings(yaml_cfg, _read_env())


# A process-wide config, lazily built on first access. Callers that want to force a
# reload (e.g. after changing env) can call ``reload_config()``.
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Config:
    global _config
    _config = load_config()
    return _config
