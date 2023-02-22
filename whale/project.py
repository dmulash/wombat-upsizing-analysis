"""Provides the Project class that ties to together ORBIT (CapEx), WOMBAT (OpEx), and
FLORIS (AEP) simulation libraries for simplified modeling workflow.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from itertools import product

import yaml
import attrs
import numpy as np
import pandas as pd
import pyarrow as pa
import networkx as nx
import pyarrow.csv  # pylint: disable=W0611
import matplotlib.pyplot as plt
from tqdm import tqdm
from attrs import field, define
from ORBIT import ProjectManager, load_config
from floris.tools import FlorisInterface
from floris.tools.wind_rose import WindRose

from whale.core import load_yaml
from wombat.core import Simulation
from wombat.core.data_classes import FromDictMixin


def resolve_path(value: str | Path) -> Path:
    """Converts a user-input string to a ``Path`` object and resolves it.

    Args:
        value (str | Path): A string or Path to a configuration library.

    Raises
    ------
        TypeError: Raised if the input to :py:attr:`value` is not either a ``str`` or
            ``pathlib.Path``.

    Returns
    -------
        Path: The resolved Path object versio of the input library path.
    """
    if isinstance(value, str):
        value = Path(value)
    if isinstance(value, Path):
        return value.resolve()

    raise TypeError(f"The input path: {value}, must be of type `str` or `pathlib.Path`.")


def load_weather(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Loads in the weather file using PyArrow, but returing a ``pandas.DataFrame``
    object. Must have the column "datetime", which can be converted to a
    ``pandas.DatetimeIndex``.

    Args:
        value : str | Path | pd.DataFrame
            The input file name and path, or a ``pandas.DataFrame`` (gets passed back
            without modification).

    Returns
    -------
        pd.DataFrame
            The full weather profile with the column "datetime" as a ``pandas.DatetimeIndex``.
    """
    if isinstance(value, pd.DataFrame):
        return value

    value = resolve_path(value)
    convert_options = pa.csv.ConvertOptions(
        timestamp_parsers=[
            "%m/%d/%y %H:%M",
            "%m/%d/%y %I:%M",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y %I:%M",
            "%m-%d-%y %H:%M",
            "%m-%d-%y %I:%M",
            "%m-%d-%Y %H:%M",
            "%m-%d-%Y %I:%M",
        ]
    )
    weather = (
        pa.csv.read_csv(value, convert_options=convert_options)
        .to_pandas()
        .set_index("datetime")
        .fillna(0.0)
        .resample("H")
        .interpolate(limit_direction="both", limit=5)
    )
    return weather


def run_chunked_floris(
    args: tuple,
) -> tuple[tuple[int, int], FlorisInterface, pd.DataFrame]:
    """Runs ``fi.calculate_wake()`` over a chunk of a larger time series analysis and
    returns the individual turbine powers for each corresponding time.

    Args:
        fi : FlorisInterface
            A copy of the base ``FlorisInterface`` object.
        weather : pd.DataFrame
            A subset of the full weather profile, with only the datetime index and
            columns: "windspeed" and "wind_direction".
        chunk_id : tuple[int, int]
            A tuple of the year and month for the data being processed.
        reinit_kwargs : dict, optional
            Any additional reinitialization keyword arguments. Defaults to {}.
        run_kwargs : dict, optional
            Any additional calculate_wake keyword arguments. Defaults to {}.

    Returns
    -------
        tuple[tuple[int, int], FlorisInterface, pd.DataFrame]
            The ``chunk_id``, a reinitialized ``fi`` using the appropriate wind
            parameters that can be used for further post-processing, and the
            resulting turbine powers.
    """
    fi: FlorisInterface = args[0]
    weather: pd.DataFrame = args[1]
    chunk_id: tuple[int, int] = args[2]
    reinit_kwargs: dict = args[3]
    run_kwargs: dict = args[4]

    reinit_kwargs["wind_directions"] = weather.wind_direction.values
    reinit_kwargs["wind_speeds"] = weather.windspeed.values
    fi.reinitialize(time_series=True, **reinit_kwargs)
    fi.calculate_wake(**run_kwargs)
    power_df = pd.DataFrame(fi.get_turbine_powers()[:, 0, :], index=weather.index)
    return chunk_id, fi, power_df


def run_parallel_floris(
    args_list: list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]],
    nodes: int = -1,
) -> tuple[dict[tuple[int, int], FlorisInterface], pd.DataFrame]:
    """Runs the time series floris calculations in parallel.

    Args:
        args_list : list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]])
            A list of the chunked by month arguments that get passed to
            ``run_chunked_floris``.
        nodes : int, optional
            The number of nodes to parallelize over. If -1, then it will use the floor
            of 80% of the available CPUs on the computer. Defaults to -1.

    Returns
    -------
        tuple[dict[tuple[int, int], FlorisInterface], pd.DataFrame]
            A dictionary of the ``chunk_id`` and ``FlorisInterface`` object, and the
            full turbine power dataframe (without renamed columns).
    """
    nodes = int(mp.cpu_count() * 0.7) if nodes == -1 else nodes
    with mp.Pool(nodes) as pool:
        with tqdm(total=len(args_list), desc="Time series energy calculation") as pbar:
            df_list = []
            fi_dict = {}
            for chunk_id, fi, df in pool.imap_unordered(run_chunked_floris, args_list):
                df_list.append(df)
                fi_dict[chunk_id] = fi
                pbar.update()

    fi_dict = dict(sorted(fi_dict.items()))
    turbine_power_df = pd.concat(df_list).sort_index()
    return fi_dict, turbine_power_df


@define(auto_attribs=True)
class Project(FromDictMixin):
    """The unified interface for creating, running, and assessing analyses that combine
    ORBIT, WOMBAT, and FLORIS.

    Args:
        library_path : str | pathlib.Path
            The file path where the configuration data for ORBIT, WOMBAT, and FLORIS can
            be found.
        weather_profile : str | pathlib.Path
            The file path where the weather profile data is located, with the following
            column requirements:
             - "datetime": The timestamp column
             - orbit_weather_cols: see ``orbit_weather_cols``
             - floris_windspeed: see ``floris_windspeed``
             - floris_wind_direction: see ``floris_wind_direction``
        orbit_weather_cols : list[str]
            The windspeed and wave height column names in ``weather`` to use for
            running ORBIT. Defaults to ``["windspeed", "wave_height"]``.
        floris_windspeed : str
            The windspeed column in ``weather`` that will be used for the FLORIS
            wake analysis. Defaults to "windspeed_100m".
        floris_wind_direction : str
            The wind direction column in ``weather`` that will be used for the FLORIS
            wake analysis. Defaults to "wind_direction_100m".
        floris_x_col : str
            The column of x-coordinates in the WOMBAT layout file that corresponds to
            the ``Floris.farm.layout_x`` Defaults to "floris_x".
        floris_y_col : str
            The column of x-coordinates in the WOMBAT layout file that corresponds to
            the ``Floris.farm.layout_y`` Defaults to "floris_y".
        orbit_config : str | pathlib.Path | None
            The ORBIT configuration file name or dictionary. If None, will not set up
            the ORBIT simulation component.
        wombat_config : str | pathlib.Path | None
            The WOMBAT configuration file name or dictionary. If None, will not set up
            the WOMBAT simulation component.
        floris_config : str | pathlib.Path | None
            The FLORIS configuration file name or dictionary. If None, will not set up
            the FLORIS simulation component.
    """

    library_path: Path = field(converter=resolve_path)
    weather_profile: str = field(converter=str)
    orbit_config: str | Path | dict | None = field(
        default=None, validator=attrs.validators.instance_of((str, Path, dict, None))
    )
    wombat_config: str | Path | dict | None = field(
        default=None, validator=attrs.validators.instance_of((str, Path, dict, None))
    )
    floris_config: str | Path | dict | None = field(
        default=None, validator=attrs.validators.instance_of((str, Path, dict, None))
    )
    orbit_weather_cols: list[str] = field(
        default=["windspeed", "wave_height"],
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(str),
            iterable_validator=attrs.validators.instance_of(list),
        ),
    )
    floris_windspeed: str = field(default="windspeed", converter=str)
    floris_wind_direction: str = field(default="wind_direction", converter=str)
    floris_x_col: str = field(default="floris_x", converter=str)
    floris_y_col: str = field(default="floris_y", converter=str)

    # Internally created attributes, aka, no user inputs to these
    weather: pd.DataFrame = field(init=False)
    orbit_config_dict: dict = field(factory=dict, init=False)
    wombat_config_dict: dict = field(factory=dict, init=False)
    floris_config_dict: dict = field(factory=dict, init=False)
    wombat: Simulation = field(init=False)
    orbit: ProjectManager = field(init=False)
    floris: FlorisInterface = field(init=False)
    wind_rose: WindRose = field(init=False)
    floris_turbine_order: list[str] = field(init=False, factory=list)
    floris_results_type: str = field(init=False)
    aep_mwh: float = field(init=False)
    turbine_aep_mwh: pd.DataFrame = field(init=False)
    _fi_dict: dict[tuple[int, int], FlorisInterface] = field(init=False, factory=dict)
    operations_start: pd.Timestamp = field(init=False)
    operations_end: pd.Timestamp = field(init=False)
    operations_years: int = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Post-initialization hook to complete the setup."""
        if isinstance(self.weather_profile, str | Path):
            weather_path = self.library_path / "weather" / self.weather_profile
            self.weather = load_weather(weather_path)
        self.setup_orbit()
        self.setup_wombat()
        self.setup_floris()
        if self.wombat_config is not None:
            self.connect_floris_to_turbines()

    @library_path.validator  # type: ignore
    def library_exists(self, attribute: attrs.Attribute, value: Path) -> None:
        """Validates that the user input to :py:attr:`library_path` is a valid directory.

        Args:
            attribute (attrs.Attribute): The attrs Attribute information/metadata/configuration.
            value (str | Path): The user input.

        Raises
        ------
            FileNotFoundError: Raised if :py:attr:`value` does not exist.
            ValueError: Raised if the :py:attr:`value` exists, but is not a directory.
        """
        if not value.exists():
            raise FileNotFoundError(f"The input path to {attribute.name} cannot be found: {value}")
        if not value.is_dir():
            raise ValueError(f"The input path to {attribute.name}: {value} is not a directory.")

    @classmethod
    def from_file(cls, library_path: str | Path, config_file: str | Path) -> Project:
        """Creates a ``Project`` object from either a JSON or YAML file. See
        :py:class:`Project` for configuration requirements.

        Args:
            library_path (`str | Path`): The library path to be used in the simulation.
            config_file (str | Path): The configuration file to create a :py:class:`Project`
                object from, which should be located at:
                ``library_path`` / project / config / ``config_file``.

        Raises
        ------
            FileExistsError: Raised if :py:attr:`library_path` is not a valid directory.
            ValueError: Raised if :py:attr:`config_file` is not a JSON or YAML file.

        Returns
        -------
            Project: An initialized Project object.
        """
        library_path = Path(library_path).resolve()
        if not library_path.is_dir():
            raise FileExistsError(f"{library_path} cannot be found.")
        config_file = Path(config_file)
        if config_file.suffix == ".json":
            with open(library_path / "project/config" / config_file) as f:
                config_dict = dict(json.load(f))
        if config_file.suffix in (".yml", ".yaml"):
            config_dict = load_yaml(library_path / "project/config", config_file)
        else:
            raise ValueError(
                "The configuration file must be a JSON (.json) or YAML (.yaml or .yml) file."
            )
        return Project.from_dict(config_dict)

    @property
    def config_dict(self) -> dict:
        """Generates a configuration dictionary that can be saved to a new file for later
        re/use.

        Returns
        -------
            dict: YAML-safe dictionary of a Project-loadable configuration.
        """
        wombat_config_dict = deepcopy(self.wombat_config_dict)
        wombat_config_dict["library"] = str(wombat_config_dict["library"])
        config_dict = {
            "library_path": str(self.library_path),
            "orbit_config": self.orbit_config_dict,
            "wombat_config": wombat_config_dict,
            "floris_config": self.floris_config_dict,
            "weather_profile": self.weather_profile,
            "orbit_weather_cols": self.orbit_weather_cols,
            "floris_windspeed": self.floris_windspeed,
            "floris_wind_direction": self.floris_wind_direction,
            "floris_x_col": self.floris_x_col,
            "floris_y_col": self.floris_y_col,
        }
        return config_dict

    def save_config(self, config_file: str | Path) -> None:
        """Saves a copy of the Project configuration settings to recreate the results of
        the current settings.

        Args:
            config_file (str | Path): The name to use for saving to a YAML configuration
                file.
        """
        config_dict = self.config_dict
        with open(self.library_path / "project/config" / config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def setup_orbit(self) -> None:
        """Creates the ORBIT Project Manager object and readies it for running an analysis."""
        if self.orbit_config is None:
            print("No ORBIT configuration provided, skipping model setup.")
            return

        if isinstance(self.orbit_config, (str, Path)):
            orbit_config = self.library_path / "project/config" / self.orbit_config
            self.orbit_config_dict = load_config(orbit_config)
        else:
            self.orbit_config_dict = self.orbit_config

        assert isinstance(self.weather, pd.DataFrame)  # mypy helper
        self.orbit = ProjectManager(
            self.orbit_config_dict,
            library_path=str(self.library_path),
            weather=self.weather.loc[:, self.orbit_weather_cols],
        )

    def setup_wombat(self) -> None:
        """Creates the WOMBAT Simulation object and readies it for running an analysis."""
        if self.wombat_config is None:
            print("No WOMBAT configuration provided, skipping model setup.")
            return

        if isinstance(self.wombat_config, (str, Path)):
            wombat_config = (
                self.library_path / "project/config" / self.wombat_config  # type: ignore
            )
        else:
            wombat_config = self.wombat_config  # type: ignore
        self.wombat = Simulation.from_config(wombat_config)
        self.wombat_config_dict = attrs.asdict(self.wombat.config)
        self.operations_start = self.wombat.env.weather.index.min()
        self.operations_end = self.wombat.env.weather.index.max()
        self.operations_years = self.operations_end.year - self.operations_start.year

    def setup_floris(self) -> None:
        """Creates the FLORIS FlorisInterface object and readies it for running an
        analysis.
        """
        if self.floris_config is None:
            print("No FLORIS configuration provided, skipping model setup.")
            return

        if isinstance(self.floris_config, (str, Path)):
            self.floris_config_dict = load_yaml(
                self.library_path / "project/config", self.floris_config
            )
        else:
            self.floris_config_dict = self.floris_config
        self.floris = FlorisInterface(configuration=self.floris_config_dict)

    def connect_floris_to_turbines(self, x_col: str = "floris_x", y_col: str = "floris_y"):
        """Generates ``floris_turbine_order`` from the WOMBAT ``Windfarm.layout_df``."""
        layout = self.wombat.windfarm.layout_df
        self.floris_turbine_order = [
            layout.loc[(layout[x_col] == x) & (layout[y_col] == y), "id"].values[0]
            for x, y in zip(self.floris.layout_x, self.floris.layout_y)
        ]

    def preprocess_monthly_floris(
        self,
        reinitialize_kwargs: dict = {},
        run_kwargs: dict = {},
        cut_in_wind_speed: float | None = None,
        cut_out_wind_speed: float | None = None,
    ) -> tuple[
        list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]],
        np.ndarray,
    ]:
        """Creates the monthly chunked inputs to run a parallelized FLORIS time series
        analysis.

        Args:
            reinitialize_kwargs : dict, optional
                Any keyword arguments to be assed to ``FlorisInterface.reinitialize()``.
                Defaults to {}.
            run_kwargs : dict, optional
                Any keyword arguments to be assed to ``FlorisInterface.calculate_wake()``.
                Defaults to {}.
            cut_in_wind_speed : float, optional
                The wind speed, in m/s, at which a turbine will start producing power.
            cut_out_wind_speed : float, optional
                The wind speed, in m/s, at which a turbine will stop producing power.

        Returns
        -------
            tuple[
                list[tuple[
                    FlorisInterface,
                    pd.DataFrame,
                    tuple[int, int],
                    dict,
                    dict
                ]],
                np.ndarray
            ]
                A list of tuples of:
                 - a copy of the ``FlorisInterface`` object
                 - tuple of year and month
                 - a copy of ``reinitialize_kwargs``
                 - c copy of ``run_kwargs``
        """
        month_list = range(1, 13)
        year_list = range(self.operations_start.year, self.operations_end.year + 1)

        assert isinstance(self.weather, pd.DataFrame)  # mypy helper
        weather = self.weather.loc[
            self.operations_start : self.operations_end,
            [self.floris_windspeed, self.floris_wind_direction],
        ].rename(
            columns={
                self.floris_windspeed: "windspeed",
                self.floris_wind_direction: "wind_direction",
            }
        )
        zero_power_filter = np.full((weather.shape[0]), True)
        if cut_out_wind_speed is not None:
            zero_power_filter = weather.windspeed < cut_out_wind_speed
        if cut_in_wind_speed is not None:
            zero_power_filter &= weather.windspeed >= cut_in_wind_speed

        args = [
            (
                deepcopy(self.floris),
                weather.loc[f"{month}/{year}"],
                (year, month),
                reinitialize_kwargs,
                run_kwargs,
            )
            for month, year in product(month_list, year_list)
        ]
        return args, zero_power_filter

    def run_wind_rose_aep(
        self,
        full_wind_rose: bool = False,
        run_kwargs: dict = {},
    ):
        """Runs the custom FLORIS WindRose AEP methodology that allows for gathering of
        intermediary results.

        Args:
            full_wind_rose (bool, optional): If True, the full wind profile will be
                used, otherwise, if False, the wind profile will be limited to just the
                simulation period. Defaults to False.
            run_kwargs (dict, optional): Arguments that are provided to
                `FlorisInterface.get_farm_AEP_wind_rose_class()`. Defaults to {}.

                From FLORIS:

                - cut_in_wind_speed (float, optional): Wind speed in m/s below which
                    any calculations are ignored and the wind farm is known to
                    produce 0.0 W of power. Note that to prevent problems with the
                    wake models at negative / zero wind speeds, this variable must
                    always have a positive value. Defaults to 0.001 [m/s].
                - cut_out_wind_speed (float, optional): Wind speed above which the
                    wind farm is known to produce 0.0 W of power. If None is
                    specified, will assume that the wind farm does not cut out
                    at high wind speeds. Defaults to None.
                - yaw_angles (NDArrayFloat | list[float] | None, optional):
                    The relative turbine yaw angles in degrees. If None is
                    specified, will assume that the turbine yaw angles are all
                    zero degrees for all conditions. Defaults to None.
                - turbine_weights (NDArrayFloat | list[float] | None, optional):
                    weighing terms that allow the user to emphasize power at
                    particular turbines and/or completely ignore the power
                    from other turbines. This is useful when, for example, you are
                    modeling multiple wind farms in a single floris object. If you
                    only want to calculate the power production for one of those
                    farms and include the wake effects of the neighboring farms,
                    you can set the turbine_weights for the neighboring farms'
                    turbines to 0.0. The array of turbine powers from floris
                    is multiplied with this array in the calculation of the
                    objective function. If None, this  is an array with all values
                    1.0 and with shape equal to (n_wind_directions, n_wind_speeds,
                    n_turbines). Defaults to None.
                - no_wake: (bool, optional): When *True* updates the turbine
                    quantities without calculating the wake or adding the wake to
                    the flow field. This can be useful when quantifying the loss
                    in AEP due to wakes. Defaults to *False*.
        """
        if full_wind_rose:
            assert isinstance(self.weather, pd.DataFrame)  # mypy helper
            weather = self.weather.loc[:, [self.floris_wind_direction, self.floris_windspeed]]
        else:
            start = self.wombat.env.weather.index.min()
            stop = self.wombat.env.weather.index.max()

            assert isinstance(self.weather, pd.DataFrame)  # mypy helper
            weather = self.weather.loc[
                start:stop, [self.floris_wind_direction, self.floris_windspeed]
            ]

        # recreate the FlorisInterface object for the wind rose settings
        wd, ws = weather.values.T
        self.wind_rose = WindRose()
        wind_rose_df = self.wind_rose.make_wind_rose_from_user_data(
            wd, ws
        )  # noqa: F841  pylint: disable=W0612
        freq = self.wind_rose.df.set_index(["wd", "ws"]).unstack().values

        # Recreating FlorisInterface.get_farm_AEP() w/o some of the quality checks
        # because the parameters are coming directly from other FLORIS objects, and
        # not user inputs
        wd = wind_rose_df.wd.unique()
        ws = wind_rose_df.ws.unique()
        n_wd = wd.size
        n_ws = ws.size
        ix_evaluate = ws >= run_kwargs["cut_in_wind_speed"]
        if run_kwargs["cut_out_wind_speed"] is not None:
            ix_evaluate &= ws < run_kwargs["cut_out_wind_speed"]

        farm_power = np.zeros((n_wd, n_ws))
        turbine_power = np.zeros((n_wd, n_ws, self.floris.floris.farm.n_turbines))
        if np.any(ix_evaluate):
            ws_subset = ws[ix_evaluate]
            yaw_angles = run_kwargs.get("yaw_angles", None)
            if yaw_angles is not None:
                yaw_angles = yaw_angles[:, ix_evaluate]

            self.floris.reinitialize(wind_speeds=ws_subset, wind_directions=wd)
            if run_kwargs["no_wake"]:
                self.floris.calculate_no_wake(yaw_angles=yaw_angles)
            else:
                self.floris.calculate_wake(yaw_angles=yaw_angles)

            farm_power[:, ix_evaluate] = self.floris.get_farm_power(
                turbine_weights=run_kwargs["turbine_weights"]
            )
            turbine_power[:, ix_evaluate, :] = self.floris.get_turbine_powers()
            if (weights := run_kwargs["turbine_weights"]) is not None:
                turbine_power *= weights
        else:
            self.floris.reinitialize(wind_speeds=ws, wind_directions=wd)

        self.aep_mwh = np.sum(freq * farm_power) * 8760 / 1e6
        self.turbine_aep_mwh = (
            np.sum(freq.reshape((*freq.shape, 1)) * turbine_power, axis=(0, 1)) * 8760 / 1e6
        )

    def run_floris(
        self,
        which: str,
        reinitialize_kwargs: dict = {},
        run_kwargs: dict = {},
        full_wind_rose: bool = False,
        cut_in_wind_speed: float = 0.001,
        cut_out_wind_speed: float | None = None,
        nodes: int = -1,
    ) -> None:
        """Runs either a FLORIS wind rose analysis for a simulation-level AEP value
        (``which="wind_rose"``) or a turbine-level time series for the WOMBAT simulation
        period (``which="time_series"``).

        Args:
            which : str
                One of "wind_rose" or "time_series" to run either a simulation-level
                wind rose analysis for the
            reinitialize_kwargs : dict, optional
                Any keyword arguments to be assed to ``FlorisInterface.reinitialize()``.
                Defaults to {}.
            run_kwargs : dict, optional
                Any keyword arguments to be assed to ``FlorisInterface.calculate_wake()``.
                Defaults to {}.
            full_wind_rose : bool, optional
                Indicates, for "wind_rose" analyses ONLY, if the full weather profile
                from ``weather`` (True) or the limited, WOMBAT simulation period (False)
                should be used for analyis. Defaults to False.
            cut_in_wind_speed : float, optional
                The wind speed, in m/s, at which a turbine will start producing power.
                Should only be a value if running a time series analysis. Defaults to
                0.001.
            cut_out_wind_speed : float, optional
                The wind speed, in m/s, at which a turbine will stop producing power.
                Should only be a value if running a time series analysis. Defaults to
                None.
            nodes : int, optional
                The number of nodes to parallelize over. If -1, then it will use the
                floor of 80% of the available CPUs on the computer. Defaults to -1.

        Raises
        ------
            ValueError: _description_
        """
        if which == "wind_rose":
            # TODO: Change this to be modify the standard behavior, and get the turbine
            # powers to properly account for availability later

            # Set the FLORIS defaults
            run_kwargs.setdefault("cut_in_wind_speed", cut_in_wind_speed)
            run_kwargs.setdefault("cut_out_wind_speed", cut_out_wind_speed)
            run_kwargs.setdefault("turbine_weights", None)
            run_kwargs.setdefault("yaw_angles", None)
            run_kwargs.setdefault("no_wake", False)

            self.run_wind_rose_aep(full_wind_rose=full_wind_rose, run_kwargs=run_kwargs)
            self.floris_results_type = "wind_rose"

        elif which == "time_series":
            parallel_args, zero_power_filter = self.preprocess_monthly_floris(
                reinitialize_kwargs, run_kwargs, cut_in_wind_speed, cut_out_wind_speed
            )
            fi_dict, turbine_powers = run_parallel_floris(parallel_args, nodes)

            self._fi_dict = fi_dict
            self.turbine_aep_mwh = turbine_powers
            self.connect_floris_to_turbines(x_col=self.floris_x_col, y_col=self.floris_y_col)
            self.turbine_aep_mwh.columns = self.floris_turbine_order
            self.turbine_aep_mwh = (
                self.turbine_aep_mwh.where(
                    np.repeat(
                        zero_power_filter.reshape(-1, 1),
                        self.turbine_aep_mwh.shape[1],
                        axis=1,
                    ),
                    0.0,
                )
                / 1e6
            )

            n_years = self.turbine_aep_mwh.index.year.unique().size
            self.aep_mwh = self.turbine_aep_mwh.values.sum() / n_years
            self.floris_results_type = "time_series"
        else:
            raise ValueError(f"`which` must be one of: 'wind_rose' or 'time_series', not: {which}")

    def run(
        self,
        which_floris: str,
        floris_reinitialize_kwargs: dict = {},
        floris_run_kwargs: dict = {},
        full_wind_rose: bool = False,
        skip: list[str] = [],
        cut_in_wind_speed: float = 0.001,
        cut_out_wind_speed: float | None = None,
        nodes: int = -1,
    ) -> None:
        """Run all three models in serial, or a subset if ``skip`` is used.

        Args:
            which_floris : str
                One of "wind_rose" or "time_series" if computing the farm's AEP based on
                a wind rose, or based on time series corresponding to the WOMBAT
                simulation period, respectively.
            floris_reinitialize_kwargs : dict
                Any additional ``FlorisInterface.reinitialize`` keyword arguments.
            floris_run_kwargs : dict
                Any additional ``FlorisInterface.get_farm_AEP`` or
                ``FlorisInterface.calculate_wake()`` keyword arguments, depending on
                ``which_floris`` is used.
            full_wind_rose : bool, optional
                Indicates, for "wind_rose" analyses ONLY, if the full weather profile
                from ``weather`` (True) or the limited, WOMBAT simulation period (False)
                should be used for analyis. Defaults to False.
            skip : list[str], optional
                A list of models to be skipped. This is intended to be used after a model
                is reinitialized with a new or modified configuration. Defaults to [].
            cut_in_wind_speed : float, optional
                The wind speed, in m/s, at which a turbine will start producing power.
                Can also be provided in ``floris_reinitialize_kwargs`` for a wind rose
                analysis, but must be provided here for a time series analysis. Defaults
                to 0.001.
            cut_out_wind_speed : float, optional
                The wind speed, in m/s, at which a turbine will stop producing power.
                Can also be provided in ``floris_reinitialize_kwargs`` for a wind rose
                analysis, but must be provided here for a time series analysis. Defaults
                to None.
            nodes : int, optional
                The number of nodes to parallelize over. If -1, then it will use the
                floor of 80% of the available CPUs on the computer. Defaults to -1.

        Raises
        ------
            ValueError
                Raised if ``which_floris`` is not one of "wind_rose" or "time_series".
        """
        if which_floris not in ("wind_rose", "time_series"):
            raise ValueError(
                f"`which_floris` must be one of: 'wind_rose' or 'time_series', not: {which_floris}"
            )

        if which_floris == "wind_rose" and cut_in_wind_speed is not None:
            floris_reinitialize_kwargs.update({"cut_in_wind_speed": cut_in_wind_speed})
        if which_floris == "wind_rose" and cut_out_wind_speed is not None:
            floris_reinitialize_kwargs.update({"cut_out_wind_speed": cut_out_wind_speed})

        if "orbit" not in skip:
            self.orbit.run()
        if "wombat" not in skip:
            self.wombat.run()
        if "floris" not in skip:
            self.run_floris(
                which=which_floris,
                reinitialize_kwargs=floris_reinitialize_kwargs,
                run_kwargs=floris_run_kwargs,
                full_wind_rose=full_wind_rose,
                cut_in_wind_speed=cut_in_wind_speed,
                cut_out_wind_speed=cut_out_wind_speed,
                nodes=nodes,
            )

    def reinitialize(
        self,
        orbit_config: str | Path | dict | None = None,
        wombat_config: str | Path | dict | None = None,
        floris_config: str | Path | dict | None = None,
    ) -> None:
        """Enables a user to reinitialize one or multiple of the CapEx, OpEx, and AEP
        models.

        Args:
            orbit_config (str | Path | dict | None, optional): ORBIT configuration file
                or dictionary. Defaults to None.
            wombat_config (str | Path | dict | None, optional): WOMBAT configuation file
                or dictionary. Defaults to None.
            floris_config (str | Path | dict | None, optional): FLORIS configuration
                file or dictionary. Defaults to None.
        """
        if orbit_config is not None:
            self.orbit_config = orbit_config
            self.setup_orbit()

        if wombat_config is not None:
            self.wombat_config = wombat_config
            self.setup_wombat()

        if floris_config is not None:
            self.floris_config = floris_config
            self.setup_floris()

        return

    # Helper methods

    def generate_floris_positions_from_layout(
        self,
        x_col: str = "easting",
        y_col: str = "northing",
        update_config: bool = True,
        config_fname: str | None = None,
    ) -> None:
        """Updates the FLORIS layout_x and layout_y based on the relative coordinates
        from the WOMBAT layout file.

        Args:
            x_col : str, optional
                The relative, distance-based x-coordinate column name. Defaults to "easting".
            y_col : str, optional
                The relative, distance-based y-coordinate column name. Defaults to "northing".
            update_config : bool, optional
                Run ``FlorisInterface.reinitialize`` with the updated ``layout_x`` and
                ``layout_y`` values. Defaults to True.
            config_fname : str | None, optional
                Provide a file name if ``update_config`` and this new configuration
                should be saved. Defaults to None.
        """
        layout = self.wombat.windfarm.layout_df
        x_min = layout[x_col].min()
        y_min = layout[y_col].min()
        layout.assign(floris_x=layout[x_col] - x_min, floris_y=layout[y_col] - y_min)
        layout = layout.loc[
            layout.id.isin(self.wombat.windfarm.turbine_id), ["floris_x", "floris_y"]
        ]
        x, y = layout.values.T
        self.floris.reinitialize(layout_x=x, layout_y=y)
        if update_config:
            assert isinstance(self.floris_config_dict, dict)  # mypy helper
            self.floris_config_dict["farm"]["layout_x"] = x.tolist()
            self.floris_config_dict["farm"]["layout_y"] = y.tolist()
            if config_fname is not None:
                full_path = self.library_path / "project/config" / config_fname
                with open(full_path, "w") as f:
                    yaml.dump(self.floris_config_dict, f, default_flow_style=False)
                    print(f"Updated FLORIS configuration saved to: {full_path}.")

    # Results methods
    # TODO: Figure out the actual workflows requried to have more complete/easier reporting

    def plot_farm(
        self, figure_kwargs: dict = {}, draw_kwargs: dict = {}, return_fig: bool = False
    ) -> None | tuple[plt.figure, plt.axes]:
        """Plot the graph representation of the windfarm as represented through WOMBAT.

        Args:
            figure_kwargs : dict, optional
                Customized keyword arguments for matplotlib figure instantiation that
                will passed as ``plt.figure(**figure_kwargs). Defaults to {}.``
            draw_kwargs : dict, optional
                Customized keyword arguments for ``networkx.draw()`` that can will
                passed as ``nx.draw(**figure_kwargs). Defaults to {}.``
            return_fig : bool, optional
                Whether or not to return the figure and axes objects for further editing
                and/or saving. Defaults to False.

        Returns
        -------
            None | tuple[plt.figure, plt.axes]: _description_
        """
        figure_kwargs.setdefault("figsize", (14, 12))
        figure_kwargs.setdefault("dpi", 200)

        fig = plt.figure(**figure_kwargs)
        ax = fig.add_subplot(111)

        windfarm = self.wombat.windfarm
        positions = {
            name: np.array([node["longitude"], node["latitude"]])
            for name, node in windfarm.graph.nodes(data=True)
        }

        draw_kwargs.setdefault("with_labels", True)
        draw_kwargs.setdefault("font_weight", "bold")
        draw_kwargs.setdefault("node_color", "#E37225")
        nx.draw(windfarm.graph, pos=positions, ax=ax, **draw_kwargs)

        fig.tight_layout()
        plt.show()

        if return_fig:
            return fig, ax
        return None

    def capex(self, per_mw: bool = False, breakdown: bool = False) -> pd.DataFrame:
        """Provides a thin wrapper to ORBIT's ``ProjectManager`` CapEx calculations that
        can provide a breakdown of total or normalize it by the project's capacity, in MW.

        Args:
            per_mw : bool, optional
                Provide the CapEx normalized by the project's capacity, in MW. Defaults
                to False.
            breakdown : bool, optional
                Provide a detailed view of the CapEx breakdown, and a total, which is
                the sum of the BOS, turbine, project, and soft CapEx categories.
                Defaults to False.

        Returns
        -------
            pd.DataFrame
                Project CapEx in the base currency or normalized by project capacity, in MW.
        """
        if breakdown:
            capex = pd.DataFrame.from_dict(
                self.orbit.capex_breakdown, orient="index", columns=["CapEx"]
            )
            capex.loc["Total"] = self.orbit.total_capex
        else:
            capex = pd.DataFrame([], columns="CapEx", index="Total")

        if per_mw:
            capex["CapEx per MW"] = capex / self.orbit.capacity
        return capex

    def energy_production(self, frequency: str = "project", by: str = "windfarm") -> pd.DataFrame:
        """Computes the monthly power production for the simulation by extrapolating
        the AEP if FLORIS results were computed by a wind rose, or using the time series
        results, and multiplying it by the WOMBAT monthly
        ``Metrics.production_based_availability``.

        Args:
            frequency : str, optional
                One of "project" (project total), "annual" (annual total), or
                "month-year" (monthly totals for each year). For FLORIS analyses run on
                a wind rose basis, only "project" and "annual" is available, and for
                time-series based results, all options are available. Defaults to
                "project".
            by : str, optional
                One of "windfarm" (project level) or "turbine" (turbine level) to
                indicate what level to calculate the

        Raises
        ------
            ValueError
                Raised if ``frequency`` is not one of: "project", "annual", "month-year".

        Returns
        -------
            pd.DataFrame
                The wind farm-level power prodcution for the desired ``frequency``.
        """
        # Check the frequency input
        opts = ("project", "annual", "month-year")
        if frequency not in opts:
            raise ValueError(f"`frequency` must be one of {opts}.")  # type: ignore

        # For the wind rose outputs, only consider project-level availability because
        # wind rose AEP is a long-term estimation of energy production
        if self.floris_results_type == "wind_rose":
            if frequency not in ("project", "annual"):
                raise ValueError(
                    "Wind rose analyses only allow for 'annual' and 'project' level"
                    " energy production."
                )
            power_gwh = (
                self.wombat.metrics.production_based_availability(
                    frequency="annual", by="turbine"
                ).loc[:, self.floris_turbine_order]
                * self.turbine_aep_mwh
            )

            if frequency == "project":
                power_gwh = power_gwh.sum(axis=0).to_frame("Energy Production (GWh)")

            if by == "windfarm":
                return (
                    power_gwh.sum(axis=1)
                    .rename({0: "Energy Production (GWh)"})
                    .to_frame("wind_farm")
                )
            return power_gwh.rename(index={0: "Energy Production (GWh)"})

        if self.floris_results_type == "time_series":
            availability = self.wombat.metrics.production_based_availability(
                frequency="month-year", by="turbine"
            )
            power_gwh = self.turbine_aep_mwh / 1000
            power_gwh = (
                power_gwh.assign(year=power_gwh.index.year, month=power_gwh.index.month)
                .groupby(["year", "month"])
                .sum()
                .loc[availability.index]
            ) * availability.loc[:, self.floris_turbine_order]

            if by == "windfarm":
                power_gwh = power_gwh.sum(axis=1).to_frame("Energy Production (GWh)")

            # Aggregate to the desired frequency level
            if frequency == "month-year":
                return power_gwh
            elif frequency == "annual":
                return (
                    power_gwh.reset_index(drop=False).groupby("year").sum().drop(columns=["month"])
                )
            elif frequency == "project":
                return pd.DataFrame(
                    [power_gwh.reset_index(drop=False).values.sum()],
                    columns=["Energy Production (GWh)"],
                )

    def npv(
        self, frequency: str, discount_rate: float = 0.025, offtake_price: float = 80
    ) -> pd.DataFrame:
        """Calculates the net present value of the windfarm at a project, annual, or
        monthly resolution given a base discount rate and offtake price.

        .. note:: This function will be improved over time to incorporate more of the
            financial parameter at play, such as PPAs.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        discount_rate : float, optional
            The rate of return that could be earned on alternative investments, by
            default 0.025.
        offtake_price : float, optional
            Price of energy, per MWh, by default 80.

        Returns
        -------
        pd.DataFrame
            The project net prsent value at the desired time resolution.
        """
        # Check the frequency input
        opts = ("project", "annual", "month-year")
        if frequency not in opts:
            raise ValueError(f"`frequency` must be one of {opts}.")  # type: ignore

        if self.floris_results_type == "wind_rose":
            if frequency not in ("project", "annual"):
                raise ValueError(
                    "Wind rose analyses only allow for 'annual' and 'project' level"
                    " energy production."
                )

            # Gather the OpEx, and revenues
            expenditures = self.wombat.metrics.opex("annual")
            production = self.energy_production("annual")
            revenue: pd.DataFrame = production / 1000 * offtake_price  # MWh

        else:
            # Gather the OpEx, and revenues
            expenditures = self.wombat.metrics.opex("month-year")
            production = self.energy_production("month-year")
            revenue = production / 1000 * offtake_price  # MWh

        # Instantiate the NPV with the required calculated data and compute the result
        # change the column names according to the analysis type used
        npv = revenue.join(expenditures).rename(
            columns={"Energy Production (GWh)": "revenue", "wind_farm": "revenue"}
        )
        N = npv.shape[0]
        npv.loc[:, "discount"] = np.full(N, 1 + discount_rate) ** np.arange(N)
        npv.loc[:, "NPV"] = (npv.revenue.values - npv.OpEx.values) / npv.discount.values

        # Aggregate the results to the required resolution
        if frequency == "project":
            return pd.DataFrame(npv.reset_index().sum()).T[["NPV"]]
        elif frequency == "annual":
            return npv.reset_index().groupby("year").sum()[["NPV"]]
        elif frequency == "monthly":
            return npv.reset_index().groupby("month").sum()[["NPV"]]
        return npv[["NPV"]]
