from collections.abc import Callable

__all__ = ['validate_scaling_range']


_SCALING_RANGE_LENGTH = 2


def validate_scaling_range[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Validate `scaling_range` kwarg before delegating to the wrapped callable."""

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Validate `scaling_range` before delegating to the wrapped callable."""
        try:
            raw_scaling_range = kwargs['scaling_range']
        except KeyError as error:
            message = f'Missing required keyword argument: {error!s}'
            raise KeyError(message) from None

        if (
            not isinstance(raw_scaling_range, tuple)
            or len(raw_scaling_range) != _SCALING_RANGE_LENGTH
        ):
            message = (
                'Scaling range should be a tuple of two numbers where '
                f'the first is less than the second, but got {raw_scaling_range}'
            )
            raise ValueError(message)

        lower_bound, upper_bound = raw_scaling_range
        if not isinstance(lower_bound, int | float) or not isinstance(upper_bound, int | float):
            message = (
                'Scaling range should be a tuple of two numbers where '
                f'the first is less than the second, but got {raw_scaling_range}'
            )
            raise TypeError(message)

        if float(lower_bound) >= float(upper_bound):
            message = (
                'Scaling range should be a tuple of two numbers where '
                f'the first is less than the second, but got {raw_scaling_range}'
            )
            raise ValueError(message)
        return func(*args, **kwargs)

    return wrapper
