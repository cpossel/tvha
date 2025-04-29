"""Collection of auxiliary functions."""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def convert_dict_keys_to_numeric(input_dict: dict) -> dict:
    """Converts string keys from dict (e.g. from JSON data) to integer/float.

    The function recursively attempts to convert every string key to float type,
    and further to integer type if the float is a whole number.
    If convertion fails, the original key is used.

    Raises:
        KeyError if existing key would be overwritten.

    Returns:
        Dictionary with converted keys (new dict, no in-place conversion)."""
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            value = convert_dict_keys_to_numeric(value)
        try:
            key = float(key)
            if key.is_integer():
                key = int(key)
        except ValueError:
            pass
        if key in output_dict:
            raise KeyError(
                f"Attempting to overwrite existing key `{key}`. "
                "Ensure that the input dictionary does not contain any duplicate entries "
                "(here e.g. string `1` and integer 1 are considered equal)."
            )
        output_dict[key] = value
    return output_dict


def convert_numpy_dtypes_to_native_python_types(obj: Any) -> Any:  # noqa: ANN401
    """Recursively converts all numpy dtypes in 'obj' into native python data types.

    Supported structures are dict, list, str, complex, np.ndarray, np.generic, and
    any non-iterable python types as well as any nested structure of beforementioned types.
    Dict keys, str and non-iterable python types are left unchanged.
    Tuples and similar iterables do not throw errors but are converted to lists;
    so, side effects might occur.

    This function is useful to save data to file;
    in case of saving to json/json5 files (e.g. with pyjson5)
    errors might occur if numpy dtypes are not converted."""
    if isinstance(obj, dict):
        return {
            key: convert_numpy_dtypes_to_native_python_types(value) for key, value in obj.items()
        }
    if isinstance(obj, complex):
        obj = np.real_if_close(obj).tolist()  # convert complex numbers to flaots
        if isinstance(obj, complex):  # still complex i.e. nonzero imaginary part
            return repr(obj)  # complex number not jsonable so return string representation of it
        return obj  # real part (float)
    if isinstance(obj, np.ndarray | np.generic):
        return obj.tolist()  # equivalent to obj.item() for np.generic types (i.e. numpy scalars)
    if isinstance(obj, str):  # catch edge case: str are iterable in Python
        return obj
    try:
        return [convert_numpy_dtypes_to_native_python_types(o) for o in obj]
    except TypeError:
        return obj


def convert_to_logging_level(verbosity: int) -> int:
    """Converts verbosity level from argparse input to logging level."""
    if verbosity <= 0:
        return logging.ERROR
    if verbosity == 1:
        return logging.WARNING
    if verbosity == 2:
        return logging.INFO
    # if args.verbose >= 3:
    return logging.DEBUG


def merge_dicts_safely(input_dicts: Iterable[dict]) -> dict:
    """Merges dicts only if there are no duplicate keys.

    Raises:
        KeyError if there are common keys."""
    output_dict = {}
    for d in input_dicts:
        for key, value in d.items():
            if key in output_dict:
                raise KeyError(
                    f"Key {key} already exists in dictionary. "
                    "Refusing to overwrite existing key."
                )
            output_dict[key] = value
    return output_dict


def merge_dicts_renaming_keys(input_dicts: Iterable[dict]) -> dict[int, Any]:
    """Merges dicts conserving all entries while assigning new keys.

    This method completely re-assigns every key,
    not caring about preserving any key to its original value.
    """
    return dict(enumerate(v for d in input_dicts for v in d.values()))


def literal_eval_extended_bool(string: str) -> str | bool | int | float:
    """Wrapper around ast.literal_eval to convert string to Python type.

    Additionally to ast.literal_eval the following common string representations of bools
    are evaluated as bools:
    'true', 'yes', 't', 'y', 'false', 'no', 'f', 'n' (case insensitive)."""
    try:
        return ast.literal_eval(string)
    except ValueError:
        if string.lower() in ("true", "yes", "t", "y"):
            return True
        if string.lower() in ("false", "no", "f", "n"):
            return False
    return string


class TerminationChecker:
    """Callback to terminate optimization."""

    def __init__(
        self, num_last_data_points_for_slope: int = 100, num_last_data_points_for_minimum: int = 10
    ) -> None:
        """Callback to terminate optimization.

        Callback to terminate optimization when the average slope over
        the last num_last_data_points_for_slope data points does not decrease anymore.
        Up to num_last_data_points_for_minimum iterations are performed afterwards,
        taking the first one that is smaller than the last
        num_last_data_points_for_minimum data points as final value.

        The maximum number of iterations passed to the optimizer usually takes precedence over the
        termination condition of the TerminationChecker.
        Read the optimizer's documentation for clarification.

        Args:
            num_last_data_points_for_slope:
                Number of considered data points to calculate the slope.
                Convergence is reached when the slope of these data points gets positive
                (i.e. increasing value of cost function).
            num_last_data_points_for_minimum:
                Number of last data points to determine the minimum one.
                After the slope got positive (see num_last_data_points_for_slope),
                some more iterations are performed to avoid getting the worst value of the
                last num_last_data_points_for_slope ones (i.e. the one that caused
                the slope to become positive) as final value.
                If no data point is smaller than the final one from the slope,
                the last value value after the num_last_data_points_for_minimum iterations
                is chosen as final value.

        """
        self.num_last_data_points_for_slope = num_last_data_points_for_slope
        self.num_last_data_points_for_minimum = num_last_data_points_for_minimum
        self.values: list[float] = []
        self._termination_counter: int | None = None

    def __call__(
        self,
        n_function_evaluations: int,  # noqa: ARG002
        parameters: np.ndarray,  # noqa: ARG002
        value: float,
        stepsize: float,  # noqa: ARG002
        accepted: bool,  # noqa: ARG002
    ) -> bool:
        """Calls the function.

        Returns:
            True if the optimization loop should be terminated.
        """
        self.values.append(value)

        if len(self.values) > self.num_last_data_points_for_slope:
            last_values = self.values[-self.num_last_data_points_for_slope :]
            polyfit = np.polynomial.Polynomial.fit(
                range(self.num_last_data_points_for_slope), last_values, 1
            )
            slope = polyfit.coef[1] / self.num_last_data_points_for_slope

            if self._termination_counter is not None:
                if self._termination_counter == 0:
                    # Termination condition reached: termination counter expired
                    return True
                last_values_for_minimum = self.values[-self.num_last_data_points_for_minimum : -1]
                if value < min(last_values_for_minimum):
                    # Pick the first value that is smaller than the
                    # last num_last_data_points_for_minimum values as final value
                    return True
                self._termination_counter -= 1
                return False

            if slope > 0:
                # Do num_last_data_points_for_minimum more iterations after the slope got to zero
                # to avoid using the worst function evaluation
                # (i.e. the one that caused that the slope got up to zero) as final value.
                # Keep in mind that maxiter might prevent the algorithm from
                # reaching this function's termination condition.
                self._termination_counter = self.num_last_data_points_for_minimum
                return False
        return False
