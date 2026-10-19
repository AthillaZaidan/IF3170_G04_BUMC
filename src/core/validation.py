import math


def require_integer(value, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        requirement = "positive integer" if minimum == 1 else f"integer >= {minimum}"
        raise ValueError(f"{name} must be a {requirement}, got {value!r}")


def require_number(value, name: str, minimum: float = 0, maximum: float | None = None) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        upper = maximum if maximum is not None else "infinity"
        raise ValueError(f"{name} must be finite and in [{minimum}, {upper}]")
