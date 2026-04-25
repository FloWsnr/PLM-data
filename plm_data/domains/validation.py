"""Domain configuration validation and dimension inference."""

import math
from typing import Any, Protocol

from plm_data.sampling.values import is_param_ref, is_sampler_spec


class DomainConfigLike(Protocol):
    """Structural domain config interface used by domain builders."""

    type: str
    params: dict[str, Any]
    periodic_maps: dict[str, Any]


def _require(raw: dict[str, Any], key: str, context: str = "config") -> Any:
    if key not in raw:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return raw[key]


def _as_mapping(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping. Got: {raw!r}")
    return raw


def _is_param_ref(value: Any) -> bool:
    return is_param_ref(value)


def _validate_numeric_literal_or_param_ref(value: Any, context: str) -> None:
    if isinstance(value, (int, float)) or _is_param_ref(value):
        return
    raise ValueError(
        f"{context} must be a number or 'param:<name>' reference. Got {value!r}."
    )


def _validate_integer_literal_or_param_ref(value: Any, context: str) -> None:
    if isinstance(value, int):
        return
    if isinstance(value, float) and value.is_integer():
        return
    if _is_param_ref(value):
        return
    raise ValueError(
        f"{context} must be an integer or 'param:<name>' reference. Got {value!r}."
    )


def _validate_sampleable_numeric(value: Any, context: str) -> None:
    if not isinstance(value, dict) or "sample" not in value:
        _validate_numeric_literal_or_param_ref(value, context)
        return

    sample_type = _require(value, "sample", context)
    if sample_type == "uniform":
        if set(value) != {"sample", "min", "max"}:
            raise ValueError(
                f"{context} uniform sampler must contain exactly ['max', 'min', "
                f"'sample']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["min"], f"{context}.min")
        _validate_numeric_literal_or_param_ref(value["max"], f"{context}.max")
        return

    if sample_type == "normal":
        if set(value) != {"sample", "mean", "std"}:
            raise ValueError(
                f"{context} normal sampler must contain exactly ['mean', 'sample', "
                f"'std']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["mean"], f"{context}.mean")
        _validate_numeric_literal_or_param_ref(value["std"], f"{context}.std")
        return

    if sample_type == "randint":
        if set(value) != {"sample", "min", "max"}:
            raise ValueError(
                f"{context} randint sampler must contain exactly ['max', 'min', "
                f"'sample']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["min"], f"{context}.min")
        _validate_numeric_literal_or_param_ref(value["max"], f"{context}.max")
        return

    raise ValueError(f"{context} uses unknown sampler '{sample_type}'.")


def _validate_sampleable_integer(value: Any, context: str) -> None:
    if not isinstance(value, dict) or "sample" not in value:
        _validate_integer_literal_or_param_ref(value, context)
        return
    _validate_sampleable_numeric(value, context)


class DomainValidationContext:
    """Shared validation helpers passed to domain-specific spec validators."""

    def __init__(
        self,
        domain_type: str,
        params: dict[str, Any],
        *,
        allow_sampling: bool,
    ) -> None:
        self.domain_type = domain_type
        self.params = params
        self.allow_sampling = allow_sampling

    def require_keys(self, *required: str) -> None:
        missing = [name for name in required if name not in self.params]
        if missing:
            raise ValueError(
                f"{self.domain_type.capitalize()} domain requires parameters "
                f"{sorted(required)}. Missing {sorted(missing)}."
            )

    def as_mapping(self, raw: Any, context: str) -> dict[str, Any]:
        return _as_mapping(raw, context)

    def _validate_domain_sampler(
        self, value: Any, context: str, *, integer: bool
    ) -> None:
        if not is_sampler_spec(value):
            return
        sample_type = _require(value, "sample", context)
        if sample_type not in {"uniform", "randint"}:
            raise ValueError(
                f"{context} domain sampling supports only 'uniform' and 'randint'."
            )
        if integer and sample_type != "randint":
            raise ValueError(f"{context} must use 'randint' for integer sampling.")

    def float_value(
        self, raw: Any, context: str, *, positive: bool = False
    ) -> float | None:
        if is_sampler_spec(raw):
            if not self.allow_sampling:
                raise ValueError(
                    f"{context} uses sampled values, but domain sampling is disabled. "
                    "Set 'domain.allow_sampling: true' to enable it."
                )
            self._validate_domain_sampler(raw, context, integer=False)
            _validate_sampleable_numeric(raw, context)
            return None
        if _is_param_ref(raw):
            return None
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{context} must be finite. Got {value}.")
        if positive and value <= 0.0:
            raise ValueError(f"{context} must be > 0. Got {value}.")
        return value

    def int_param(self, name: str, *, minimum: int) -> int | None:
        raw = self.params[name]
        context = f"{self.domain_type.capitalize()} domain parameter '{name}'"
        if is_sampler_spec(raw):
            if not self.allow_sampling:
                raise ValueError(
                    f"{context} uses sampled values, but domain sampling is disabled. "
                    "Set 'domain.allow_sampling: true' to enable it."
                )
            self._validate_domain_sampler(raw, context, integer=True)
            _validate_sampleable_integer(raw, context)
            return None
        if _is_param_ref(raw):
            return None
        value = int(raw)
        if float(value) != float(raw) or value < minimum:
            raise ValueError(f"{context} must be an integer >= {minimum}. Got {raw!r}.")
        return value

    def float_param(self, name: str, *, positive: bool = False) -> float | None:
        return self.float_value(
            self.params[name],
            f"{self.domain_type.capitalize()} domain parameter '{name}'",
            positive=positive,
        )

    def vector_value(
        self, raw: Any, context: str, *, length: int
    ) -> list[float] | None:
        if not isinstance(raw, list) or len(raw) != length:
            raise ValueError(
                f"{context} must be a list with {length} entries. Got {raw!r}."
            )
        values: list[float] = []
        saw_nonconcrete = False
        for index, value in enumerate(raw):
            entry_context = f"{context}[{index}]"
            if is_sampler_spec(value):
                if not self.allow_sampling:
                    raise ValueError(
                        f"{entry_context} uses sampled values, but domain sampling is "
                        "disabled. Set 'domain.allow_sampling: true' to enable it."
                    )
                self._validate_domain_sampler(value, entry_context, integer=False)
                _validate_sampleable_numeric(value, entry_context)
                saw_nonconcrete = True
                continue
            if _is_param_ref(value):
                saw_nonconcrete = True
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{entry_context} must be finite. Got {numeric}.")
            values.append(numeric)
        if saw_nonconcrete:
            return None
        return values

    def vector_param(self, name: str, *, length: int) -> list[float] | None:
        return self.vector_value(
            self.params[name],
            f"{self.domain_type.capitalize()} domain parameter '{name}'",
            length=length,
        )

    def positive_int_vector(self, name: str, *, length: int) -> list[int] | None:
        raw = self.params[name]
        if not isinstance(raw, list) or len(raw) != length:
            raise ValueError(
                f"{self.domain_type.capitalize()} domain parameter '{name}' must be a "
                f"list with {length} entries. Got {raw!r}."
            )
        values: list[int] = []
        saw_nonconcrete = False
        for index, value in enumerate(raw):
            context = (
                f"{self.domain_type.capitalize()} domain parameter '{name}[{index}]'"
            )
            if is_sampler_spec(value):
                if not self.allow_sampling:
                    raise ValueError(
                        f"{context} uses sampled values, but domain sampling is "
                        "disabled. Set 'domain.allow_sampling: true' to enable it."
                    )
                self._validate_domain_sampler(value, context, integer=True)
                _validate_sampleable_integer(value, context)
                saw_nonconcrete = True
                continue
            if _is_param_ref(value):
                saw_nonconcrete = True
                continue
            int_value = int(value)
            if float(int_value) != float(value) or int_value <= 0:
                raise ValueError(
                    f"{context} must be a positive integer. Got {value!r}."
                )
            values.append(int_value)
        if saw_nonconcrete:
            return None
        return values


def infer_domain_dimension(domain_type: str, params: dict[str, Any]) -> int:
    """Infer the spatial dimension from the configured domain."""
    try:
        from plm_data.domains import get_domain_spec

        return get_domain_spec(domain_type).dimension
    except ValueError:
        pass

    size = params.get("size")
    if isinstance(size, list):
        return len(size)

    raise ValueError(
        f"Cannot infer spatial dimension for domain type '{domain_type}'. "
        "Provide a supported built-in domain."
    )


def _validate_against_parameter_specs(context: DomainValidationContext) -> None:
    from plm_data.domains import get_domain_spec

    spec = get_domain_spec(context.domain_type)
    context.require_keys(*spec.parameters)
    for parameter in spec.parameters.values():
        if parameter.kind == "float":
            context.float_param(parameter.name)
            continue
        if parameter.kind == "int":
            minimum = int(parameter.hard_min) if parameter.hard_min is not None else 0
            context.int_param(parameter.name, minimum=minimum)
            continue
        if parameter.kind == "float_vector":
            if parameter.length is None:
                raise ValueError(
                    f"Domain spec '{spec.name}' parameter '{parameter.name}' needs "
                    "a vector length."
                )
            context.vector_param(parameter.name, length=parameter.length)
            continue
        if parameter.kind == "int_vector":
            if parameter.length is None:
                raise ValueError(
                    f"Domain spec '{spec.name}' parameter '{parameter.name}' needs "
                    "a vector length."
                )
            context.positive_int_vector(parameter.name, length=parameter.length)
            continue
        if parameter.kind in {"hole_list"}:
            continue
        raise ValueError(
            f"Domain spec '{spec.name}' parameter '{parameter.name}' has unsupported "
            f"kind '{parameter.kind}'."
        )


def validate_domain_params(
    domain_type: str,
    params: dict[str, Any],
    *,
    allow_sampling: bool = False,
) -> None:
    """Validate domain parameters through the domain's registered spec."""
    from plm_data.domains import get_domain_spec

    spec = get_domain_spec(domain_type)
    context = DomainValidationContext(
        domain_type,
        params,
        allow_sampling=allow_sampling,
    )
    _validate_against_parameter_specs(context)
    if spec.validate_params is not None:
        spec.validate_params(context)
