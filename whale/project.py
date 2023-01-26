"""Provides the Project class that ties to together ORBIT (CapEx), WOMBAT (OpEx), and
FLORIS (AEP) simulation libraries for simplified modeling workflow.
"""

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

from wombat.core import Simulation
from wombat.core.library import load_yaml
from wombat.core.data_classes import FromDictMixin


def resolve_path(value: str | Path) -> Path:
    """Converts a user-input string to a ``Path`` object and resolves it.

    Args:
        value (str | Path): A string or Path to a configuration library.

    Raises:
        TypeError: Raised if the input to :py:attr:`value` is not either a ``str`` or ``pathlib.Path``.

    Returns:
        Path: The resolved Path object versio of the input library path.
    """
    if isinstance(value, str):
        value = Path(value)
    if isinstance(value, Path):
        return value.resolve()

    raise TypeError(
        f"The input path: {value}, must be of type `str` or `pathlib.Path`."
    )


def read_config(value: str | Path | dict) -> dict:
    """Reads the configuration file from a YAML to a dictionary.

    Args:
        value (str | Path | dict): The path to, or a dictionary of the configuration.

    Raises:
        TypeError: Raised if not a valid file name or already a dictionary.

    Returns:
        dict: The configuration dictionary.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, Path)):
        value = load_yaml(resolve_path(value))
    raise TypeError(
        (
            f"The input configuration is not a valid format: {type(value)}. Inputs must"
            " be of type `str`, `Path`, or `dict`."
        )
    )


def load_weather(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Loads in the weather file using PyArrow, but returing a ``pandas.DataFrame``
    object. Must have the column "datetime", which can be converted to a
    ``pandas.DatetimeIndex``.

    Args:
        value : str | Path | pd.DataFrame
            The input file name and path, or a ``pandas.DataFrame`` (gets passed back
            without modification).

    Returns:
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
    returns the individual turbine powers for each corresponding time

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

    Returns:
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

    Returns:
        tuple[dict[tuple[int, int], FlorisInterface], pd.DataFrame]
            A dictionary of the ``chunk_id`` and ``FlorisInterface`` object, and the
            full turbine power dataframe (without renamed columns).
    """
    nodes = int(mp.cpu_count() * 0.8) if nodes == -1 else nodes
    with mp.Pool(nodes) as pool:
        with tqdm(total=len(args_list), desc="Calculating turbine-level power") as pbar:
            df_list = []
            fi_dict = {}
            for chunk_id, fi, df in pool.imap_unordered(run_chunked_floris, args_list):
                df_list.append(df)
                fi_dict[chunk_id] = fi
                pbar.update()

    fi_dict = dict(sorted(fi_dict.items()))
    turbine_power_df = pd.concat(df_list).sort_index()
    return fi_dict, turbine_power_df


@define
class Project(FromDictMixin):
    """The unified interface for creating, running, and assessing analyses that combine
    ORBIT, WOMBAT, and FLORIS.

    Args:
        library_path : str | pathlib.Path
            The file path where the configuration data for ORBIT, WOMBAT, and FLORIS can
            be found.
        weather : str | pathlib.Path
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
    weather: str | Path | pd.DataFrame
    orbit_config: str | Path | dict | None = field(default=None)
    wombat_config: str | Path | dict | None = field(default=None)
    floris_config: str | Path | dict | None = field(default=None)
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
    orbit_config_dict: dict = field(factory=dict, init=False)
    wombat: Simulation = field(init=False)
    orbit: ProjectManager = field(init=False)
    floris: FlorisInterface = field(init=False)
    floris_wind_rose: WindRose = field(init=False)
    floris_turbine_order: list[str] = field(init=False, factory=list)
    aep_mwh: float = field(init=False)
    floris_turbine_powers: pd.DataFrame = field(init=False)
    _fi_dict: dict[tuple[int, int], FlorisInterface] = field(init=False, factory=dict)

    def __attrs_post_init__(self) -> None:
        if isinstance(self.weather, str | Path):
            weather_path = self.library_path / "weather" / self.weather
            self.weather = load_weather(weather_path)
        self.setup_orbit()
        self.setup_wombat()
        self.setup_floris()

    @library_path.validator  # type: ignore
    def library_exists(self, attribute: attrs.Attribute, value: Path) -> None:
        """Validates that the user input to :py:attr:`library_path` is a valid directory.

        Args:
            attribute (attrs.Attribute): The attrs Attribute information/metadata/configuration.
            value (str | Path): The user input.

        Raises:
            FileNotFoundError: Raised if :py:attr:`value` does not exist.
            ValueError: Raised if the :py:attr:`value` exists, but is not a directory.
        """
        if not value.exists():
            raise FileNotFoundError(
                f"The input path to {attribute.name} cannot be found: {value}"
            )
        if not value.is_dir():
            raise ValueError(
                f"The input path to {attribute.name}: {value} is not a directory."
            )

    def setup_orbit(self) -> None:
        """Creates the ORBIT Project Manager object and readies it for running an analysis."""
        if self.orbit_config is None:
            print("No ORBIT configuration provided, skipping model setup.")
            return

        if isinstance(self.orbit_config, (str, Path)):
            self.orbit_config = self.library_path / "project/config" / self.orbit_config
        self.orbit_config_dict = load_config(self.orbit_config)

        assert isinstance(self.weather, pd.DataFrame)  # mypy helper
        self.orbit = ProjectManager(
            self.orbit_config_dict,
            library_path=str(self.library_path),
            weather=self.weather.loc[:, self.orbit_weather_cols],
        )

    def setup_wombat(self) -> None:
        """Creates the WOMBAT Simulation object and readies it for running an analysis."""
        if self.orbit_config is None:
            print("No WOMBAT configuration provided, skipping model setup.")
            return

        if isinstance(self.wombat_config, (str, Path)):
            self.wombat_config = (
                self.library_path / "project/config" / self.wombat_config  # type: ignore
            )
        self.wombat = Simulation.from_config(self.wombat_config)

    def setup_floris(self) -> None:
        """Creates the FLORIS FlorisInterface object and readies it for running an
        analysis.
        """
        if self.floris_config is None:
            print("No FLORIS configuration provided, skipping model setup.")
            return

        if isinstance(self.floris_config, (str, Path)):
            self.floris_config = load_yaml(
                self.library_path / "project/config", self.floris_config
            )
        self.floris = FlorisInterface(configuration=self.floris_config)

    def connect_floris_to_turbines(
        self, x_col: str = "floris_x", y_col: str = "floris_y"
    ):
        """Generates ``floris_turbine_order`` from the WOMBAT ``Windfarm.layout_df``"""
        layout = self.wombat.windfarm.layout_df
        self.floris_turbine_order = [
            layout.loc[(layout[x_col] == x) & (layout[y_col] == y), "id"].values[0]
            for x, y in zip(self.floris.layout_x, self.floris.layout_y)
        ]

    def preprocess_monthly_floris(
        self, reinitialize_kwargs: dict = {}, run_kwargs: dict = {}
    ) -> list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]]:
        """Creates the monthly chunked inputs to run a parallelized FLORIS time series
        analysis.

        Args:
            reinitialize_kwargs : dict, optional
                Any keyword arguments to be assed to ``FlorisInterface.reinitialize()``.
                Defaults to {}.
            run_kwargs : dict, optional
                Any keyword arguments to be assed to ``FlorisInterface.calculate_wake()``.
                Defaults to {}.

        Returns:
            list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]]
                A list of tuples of:
                 - a copy of the ``FlorisInterface`` object
                 - tuple of year and month
                 - a copy of ``reinitialize_kwargs``
                 - c copy of ``run_kwargs``
        """
        start = self.wombat.env.weather.index.min().year
        end = self.wombat.env.weather.index.max().year
        month_list = range(1, 13)
        year_list = range(start, end + 1)

        assert isinstance(self.weather, pd.DataFrame)  # mypy helper
        weather = self.weather.loc[
            :, [self.floris_windspeed, self.floris_wind_direction]
        ].rename(
            columns={
                self.floris_windspeed: "windspeed",
                self.floris_wind_direction: "wind_direction",
            }
        )

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
        return args

    def run_floris(
        self,
        which: str,
        reinitialize_kwargs: dict = {},
        run_kwargs: dict = {},
        full_wind_rose: bool = False,
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
            nodes : int, optional
                The number of nodes to parallelize over. If -1, then it will use the
                floor of 80% of the available CPUs on the computer. Defaults to -1.

        Raises:
            ValueError: _description_
        """
        if which == "wind_rose":
            # Get the weather for the length of the WOMBAT simulation
            if full_wind_rose:
                assert isinstance(self.weather, pd.DataFrame)  # mypy helper
                weather = self.weather.loc[
                    :, [self.floris_wind_direction, self.floris_windspeed]
                ]
            else:
                start = self.wombat.env.weather.index.min()
                stop = self.wombat.env.weather.index.max()

                assert isinstance(self.weather, pd.DataFrame)  # mypy helper
                weather = self.weather.loc[
                    start:stop, [self.floris_wind_direction, self.floris_windspeed]
                ]
            wd, ws = weather.values.T
            wind_rose = WindRose()
            wind_rose_df = wind_rose.make_wind_rose_from_user_data(  # noqa: F841  pylint: disable=W0612
                wd, ws
            )

            self.floris_wind_rose = wind_rose

            self.aep_mwh = (
                self.floris.get_farm_AEP_wind_rose_class(
                    wind_rose=self.floris_wind_rose, **run_kwargs
                )
                / 1e6
            )
        elif which == "time_series":
            parallel_args = self.preprocess_monthly_floris(
                reinitialize_kwargs, run_kwargs
            )
            fi_dict, turbine_powers = run_parallel_floris(parallel_args, nodes)

            self._fi_dict = fi_dict
            self.floris_turbine_powers = turbine_powers
            self.floris_turbine_powers.colums = self.floris_turbine_order

            n_years = self.floris_turbine_powers.index.year.unique().size
            self.aep_mwh = self.floris_turbine_powers.values.sum() / n_years / 1e6
            # TODO: Calculate the availability x turbine powers based on coordinate re-matching

        else:
            raise ValueError(
                f"`which` must be one of: 'wind_rose' or 'time_series', not: {which}"
            )

    def run(
        self,
        which_floris: str,
        floris_reinitialize_kwargs: dict = {},
        floris_run_kwargs: dict = {},
        full_wind_rose: bool = False,
        skip: list[str] = [],
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
            nodes : int, optional
                The number of nodes to parallelize over. If -1, then it will use the
                floor of 80% of the available CPUs on the computer. Defaults to -1.

        Raises:
            ValueError
                Raised if ``which_floris`` is not one of "wind_rose" or "time_series".
        """
        if which_floris not in ("wind_rose", "time_series"):
            raise ValueError(
                f"`which_floris` must be one of: 'wind_rose' or 'time_series', not: {which_floris}"
            )

        if "orbit" not in skip:
            self.orbit.run()
        if "wombat" not in skip:
            self.wombat.run()
        if "floris" not in skip:
            self.run_floris(
                which_floris,
                floris_reinitialize_kwargs,
                floris_run_kwargs,
                full_wind_rose,
                nodes,
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

        Returns:
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
        self.floris.reinitialize(
            layout_x=layout.floris_x.values, layout_y=layout.floris_y.values
        )
        if update_config:
            assert isinstance(self.floris_config, dict)  # mypy helper
            self.floris_config["farm"]["layout_x"] = layout.floris_x.values.tolist()
            self.floris_config["farm"]["layout_y"] = layout.floris_y.values.tolist()
            if config_fname is not None:
                full_path = self.library_path / "project/config" / config_fname
                with open(full_path, "w") as f:
                    yaml.dump(self.floris_config, f, default_flow_style=False)
                    print(f"Updated FLORIS configuration saved to: {full_path}.")
