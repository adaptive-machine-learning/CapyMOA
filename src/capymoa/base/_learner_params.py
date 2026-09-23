"""Capture, serialize, register, and reconstruct learner parameters."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import (
    Any,
    Concatenate,
    Generic,
    NotRequired,
    ParamSpec,
    TypedDict,
    TypeVar,
    cast,
)

P = ParamSpec("P")
TClass = TypeVar("TClass", bound=type)


class LearnerSpec(TypedDict):
    """Serialized learner specification used for capturing and reconstructing learner
    hyper-parameters."""

    learner: str
    """Fully qualified name of the learner class. `module.ClassName` format."""
    params: NotRequired[dict[str, Any | LearnerSpec]]
    """Serialized parameters for the learner. May contain nested learner specifications."""


class LearnerRegistry(Generic[TClass]):
    """Registry for learner classes keyed by fully qualified name."""

    def __init__(self, expected_type: type | None = None) -> None:
        self._registered: dict[str, TClass] = {}
        self._expected_type = expected_type

    def register(self, cls: TClass) -> TClass:
        if self._expected_type is not None and not issubclass(cls, self._expected_type):
            raise TypeError(f"Expected subclass of {self._expected_type}, got {cls}")
        self._registered[_get_learner_name(cls)] = cls
        return cls

    def get(self, name: str) -> TClass | None:
        return self._registered.get(name)

    def require(self, name: str) -> TClass:
        cls = self.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown learner {name!r}. Known learners: {self.list_names()}."
            )
        return cls

    def list_names(self) -> list[str]:
        return list(self._registered)

    def build(self, spec: LearnerSpec, schema: Any, random_seed: int = 1) -> Any:
        """Construct a registered learner from a serialized specification."""
        cls = self.require(spec["learner"])
        return cls.from_params(schema, spec.get("params", {}), random_seed)


_LEARNER_REGISTRY = LearnerRegistry()


def _get_learner_name(cls: type) -> str:
    module = cls.__module__.split("._", 1)[0]
    return f"{module}.{cls.__qualname__}"


def _capture_learner_params(
    init: Callable[Concatenate[Any, P], Any],
) -> Callable[Concatenate[Any, P], Any]:
    @functools.wraps(init)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> Any:
        if getattr(self, "_capturing_params", False):
            return init(self, *args, **kwargs)
        bound = inspect.signature(init).bind(self, *args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        params.pop("self", None)
        self._params = params
        self._capturing_params = True
        try:
            return init(self, *args, **kwargs)
        finally:
            self._capturing_params = False

    return wrapper


def _get_construction_kwargs(
    cls: type, schema: Any, random_seed: int, params: dict[str, Any]
) -> dict[str, Any]:
    parameters = inspect.signature(cls.__init__).parameters
    has_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    kwargs = dict(params)
    if has_var_kwargs or "schema" in parameters:
        kwargs["schema"] = schema
    if has_var_kwargs or "random_seed" in parameters:
        kwargs["random_seed"] = random_seed
    return kwargs


def _serialize_param(value: Any) -> Any:
    """Recursively convert captured values into a YAML/JSON-friendly form."""
    if isinstance(value, LearnerParamsMixin):
        return {
            "learner": _get_learner_name(type(value)),
            "params": value.get_params(),
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_serialize_param(item) for item in value)
    if isinstance(value, dict):
        return {key: _serialize_param(item) for key, item in value.items()}
    return value


def _deserialize_param(value: Any, schema: Any, random_seed: int) -> Any:
    """Recursively resolve nested specs in ``value`` back into live objects."""
    if isinstance(value, dict):
        if "learner" in value and "params" in value:
            cls = _LEARNER_REGISTRY.require(value["learner"])
            return cls.from_params(schema, value.get("params", {}), random_seed)
        return {
            key: _deserialize_param(item, schema, random_seed)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            _deserialize_param(item, schema, random_seed) for item in value
        )
    return value


def learner_from_params(spec: LearnerSpec, schema: Any, random_seed: int = 1) -> Any:
    """Construct a learner from a serialized fully qualified learner spec.

    See :class:`~capymoa.base.LearnerParamsMixin` for the parameter capture
    and serialization support used by learners.
    """
    return _LEARNER_REGISTRY.build(spec, schema, random_seed)


class LearnerParamsMixin:
    """Mixin enabling capturing and reconstructing learner hyper-parameters.

    Use :func:`~capymoa.base.learner_from_params` to reconstruct a learner
    from a serialized specification.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            cls.__init__ = cast(Any, _capture_learner_params(cls.__dict__["__init__"]))
        _LEARNER_REGISTRY.register(cls)

    def get_params(self) -> dict[str, Any]:
        """Return the hyper-parameters captured from the constructor."""
        params = dict(getattr(self, "_params", {}))
        params.pop("schema", None)
        params.pop("random_seed", None)
        return {key: _serialize_param(value) for key, value in params.items()}

    @classmethod
    def from_params(
        cls,
        schema: Any = None,
        params: dict[str, Any] | None = None,
        random_seed: int = 1,
    ) -> Any:
        """Construct an instance from parameters produced by ``get_params``."""
        resolved = {
            key: _deserialize_param(value, schema, random_seed)
            for key, value in (params or {}).items()
        }
        kwargs = _get_construction_kwargs(cls, schema, random_seed, resolved)
        return cls(**kwargs)
