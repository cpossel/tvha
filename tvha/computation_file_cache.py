"""Collection of code to support saving calculations to a file."""

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ComputationFileCache(ABC):
    """Base class for computations cached into a CSV file.

    Public API:
        get_datapoints(datapoints=None, **kwargs)

    Internal representation:
        params: dict[str, list]  (always lists, floats rounded)

    Stages:
        1. Parse/normalize parameters
        2. Determine missing datapoints
        3. Compute missing datapoints
        4. Update local cache + write to CSV
        5. Filter and return resulting DataFrame
    """

    def __init__(self, output_file: Path | str = "output/data.csv") -> None:
        """Initialize the computation cache.

        Args:
            output_file: The cache file for calculation results.
        """
        self.output_file = Path(output_file).resolve()
        self.output_file.parent.mkdir(exist_ok=True)

        self._default_param_dict = self._get_default_param_values()
        self._output_param_names = set()  # determined on the fly

        self.data = self._load_existing_cache_from_csv_file()

    def _load_existing_cache_from_csv_file(self) -> pd.DataFrame:
        """Load cached data from CSV or return empty DataFrame upon failure."""
        try:
            df = pd.read_csv(self.output_file, index_col=0)
        except FileNotFoundError:
            logger.info("Cached data file not found.", exc_info=True)
            return pd.DataFrame(columns=[*self._default_param_dict, *self._output_param_names])

        if df.empty:
            if df.columns.empty:
                return pd.DataFrame(columns=[*self._default_param_dict, *self._output_param_names])
            return df

        if not set(self._default_param_dict) <= set(df.columns):
            raise ValueError(
                f"Columns {set(self._default_param_dict) - set(df.columns)} missing in cache file. "
                f"Please add them manually into file {self.output_file} before resuming."
            )

        # round all floats consistently
        df.update(df.select_dtypes(include=[float]).round(12))

        return df.drop_duplicates(subset=list(self._default_param_dict)).replace({None: np.nan})

    def _flush_row_to_csv_file(self, df_row: pd.DataFrame) -> None:
        """Append a single row DataFrame to the CSV file.

        Args:
            df_row: A one-row DataFrame representing a computed datapoint.
        """
        if set(df_row.columns) != set(self.data.columns):
            raise ValueError(
                "Different columns for calculated data and cache file detected. "
                f"Fix file {self.output_file} or signature of 'calculate_datapoint' manually "
                "before restarting the calculation."
                "Also merge the corrupted datapoint manually "
                "unless you consider its computation time negligible:\n"
                f"{df_row.to_csv()}"
            )

        write_header = True
        if self.output_file.exists():
            try:
                pd.read_csv(self.output_file, nrows=1)
                write_header = False
            except Exception:  # noqa: S110
                pass

        df_row.to_csv(
            self.output_file,
            mode="a",
            header=write_header,
            columns=self.data.columns,
        )

    def _parse_param_values(self, raw_params: dict[str, Any]) -> dict[str, list]:
        """Convert raw kwargs into list-like parameters, rounding floats.

        Rounding floats to 12 digits ensures that subsequent equality comparisons succeed.

        Args:
            raw_params: Arbitrary user input parameters.

        Returns:
            A dict of lists.
        """
        params: dict[str, list] = {}

        for key, value in raw_params.items():
            # Wrap into list as needed
            if isinstance(value, str):  # noqa: SIM114
                values = [value]
            elif not isinstance(value, Sequence):
                values = [value]
            else:
                values = list(value)

            # Round floats
            rounded = [
                round(v, 12) if isinstance(v, float) else (np.nan if v is None else v)
                for v in values
            ]

            params[key] = rounded

        return params

    def _get_default_param_values(self) -> dict[str, Any]:
        """Determine default parameter values from self.calculate_datapoint's signature."""
        sig = inspect.signature(self.calculate_datapoint)
        defaults = {}
        for name, param in sig.parameters.items():
            if param.default is not param.empty:  # has default
                defaults[name] = param.default
            else:
                defaults[name] = np.nan
        return defaults

    def _filter_cached(self, params: dict[str, list]) -> pd.DataFrame:
        """Filter cached DataFrame using params dict."""
        if self.data.empty:
            return pd.DataFrame()

        mask = pd.Series(True, index=self.data.index)
        for column, values in params.items():
            mask &= self.data[column].isin(values)

        return self.data[mask]

    def _find_missing_datapoints(
        self, datapoints: Sequence[dict[str, Any]] | None, params: dict[str, list]
    ) -> list[dict[str, Any]]:
        """Return parameter dicts that are not yet cached."""
        if datapoints:  # explicit datatpoints from user
            df = pd.DataFrame(datapoints)
            param_cols = list(df.columns)
        else:  # cartesian product from parameter list
            df = pd.MultiIndex.from_product(params.values(), names=params.keys()).to_frame(
                index=False
            )
            param_cols = list(params.keys())

        if self.data.empty:
            return df.to_dict(orient="records")

        merged = df.replace({None: np.nan}).merge(
            self.data[param_cols], on=param_cols, how="left", indicator=True
        )

        missing_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        return missing_df.to_dict(orient="records")

    def _compute_missing_datapoints(
        self,
        missing_datapoints: Sequence[dict[str, Any]],
    ) -> list[pd.DataFrame]:
        """Compute missing datapoints and flust results to CSV.

        Args:
            missing_datapoints: List of parameter sets (as dicts).

        Returns:
            list[pd.DataFrame]: List of newly computed rows.
        """
        if not missing_datapoints:
            return []

        new_rows: list[pd.DataFrame] = []

        for i, datapoint in enumerate(
            tqdm(
                missing_datapoints,
                desc="Data Calculation",
                disable=len(missing_datapoints) <= 1,
            )
        ):
            result_dict = self.calculate_datapoint(**datapoint)

            if not self._output_param_names:
                self._output_param_names = set(result_dict)
                self.data = pd.DataFrame(
                    self.data,
                    columns=list(self.data.columns)
                    + [name for name in self._output_param_names if name not in self.data.columns],
                )

            df_row = pd.DataFrame(
                [{**datapoint, **result_dict}],
                columns=self.data.columns,
                index=[i + len(self.data)],
            )
            df_row.update(df_row.select_dtypes(include=[float]).round(12))

            with FileLock(str(self.output_file) + ".lock"):
                self._flush_row_to_csv_file(df_row)

            new_rows.append(df_row)

        return new_rows

    def _update_cache(self, new_rows: list[pd.DataFrame]) -> None:
        """Merge newly computed rows into cache, performing sorting and re-indexing."""
        self.data = pd.concat([self.data, *new_rows], ignore_index=True, verify_integrity=True)

        self.data.sort_values(by=list(self._default_param_dict), inplace=True, ignore_index=True)

    def _clean_csv_file(self) -> None:
        """Clean the CSV file removing duplicates, performing resorting and re-indexing.

        Another process might have dropped some rows into the file in the meantime, so data
        is read again from file.
        """
        df = self._load_existing_cache_from_csv_file()
        if set(df.columns) != set(self.data.columns):
            raise ValueError(
                "Column names from file and from cached data differ. "
                f"Please fix file {self.output_file} or internal 'data_header' manually."
            )
        df.sort_values(by=list(self._default_param_dict), ignore_index=True).to_csv(
            self.output_file, mode="w"
        )

    def get_datapoints(
        self,
        datapoints: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> pd.DataFrame:
        """Return datapoints computed or retrieved from the cache.

        Args:
            datapoints: Optional explicit datapoints.
            **kwargs: Parameter names + lists/scalars of parameter values.

        Returns:
            Filtered DataFrame containing the requested datapoints.
        """
        if datapoints and kwargs:
            raise ValueError("Use either 'datapoints' OR kwargs, not both.")

        params = self._parse_param_values(kwargs)

        if datapoints:
            datapoints_with_defaults = []
            for datapoint in datapoints:
                for param_name in self._default_param_dict:
                    value = datapoint.get(param_name, self._default_param_dict[param_name])
                    datapoint[param_name] = np.nan if value is None else value
                datapoints_with_defaults.append(datapoint)
        else:
            datapoints_with_defaults = None

        missing_datapoints = self._find_missing_datapoints(datapoints_with_defaults, params)
        new_rows = self._compute_missing_datapoints(missing_datapoints=missing_datapoints)
        self._update_cache(new_rows)
        self._clean_csv_file()

        if datapoints_with_defaults:
            return pd.concat(
                [
                    self._filter_cached({k: [v] for k, v in datapoint.items()})
                    for datapoint in datapoints_with_defaults
                ],
                ignore_index=True,
            )

        return self._filter_cached(params)

    @abstractmethod
    def calculate_datapoint(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Subclass must implement this to compute a single datapoint."""
        raise NotImplementedError
