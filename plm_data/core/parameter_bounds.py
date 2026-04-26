"""Shared validation for sampling-facing numeric bounds."""


def validate_parameter_bounds(
    context: str,
    *,
    hard_min: float | int | None,
    hard_max: float | int | None,
    sampling_min: float | int | None,
    sampling_max: float | int | None,
) -> None:
    """Validate declared hard and sampling bounds for one parameter spec."""
    if (sampling_min is None) != (sampling_max is None):
        raise ValueError(
            f"{context} must define both sampling_min and sampling_max, or neither."
        )
    if (
        sampling_min is not None
        and sampling_max is not None
        and sampling_min > sampling_max
    ):
        raise ValueError(f"{context} has sampling_min greater than sampling_max.")
    if hard_min is not None and hard_max is not None and hard_min > hard_max:
        raise ValueError(f"{context} has hard_min greater than hard_max.")
    if sampling_min is not None and hard_min is not None and sampling_min < hard_min:
        raise ValueError(
            f"{context} sampling_min must be >= hard_min {hard_min}. "
            f"Got {sampling_min!r}."
        )
    if sampling_max is not None and hard_min is not None and sampling_max < hard_min:
        raise ValueError(
            f"{context} sampling_max must be >= hard_min {hard_min}. "
            f"Got {sampling_max!r}."
        )
    if sampling_min is not None and hard_max is not None and sampling_min > hard_max:
        raise ValueError(
            f"{context} sampling_min must be <= hard_max {hard_max}. "
            f"Got {sampling_min!r}."
        )
    if sampling_max is not None and hard_max is not None and sampling_max > hard_max:
        raise ValueError(
            f"{context} sampling_max must be <= hard_max {hard_max}. "
            f"Got {sampling_max!r}."
        )
