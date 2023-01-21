from lib2to3.pytree import convert
from pathlib import Path

import attrs
import pandas as pd
from attrs import define, field

from ORBIT import ProjectManager, load_config
from wombat.core import Simulation
from floris.tools import FlorisInterface
from floris.tools.wind_rose import WindRose
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
    
    raise TypeError(f"The input path: {value}, must be of type `str` or `pathlib.Path`.")


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
    if isinstance(value, pd.DataFrame):
        return value

    value = resolve_path(value)
    df = pd.read_csv(
        value,
        engine="pyarrow",
        parse_dates=["datetime"],
        index_col="datetime",
        dtype=float
    )
    return df

@define
class Project(FromDictMixin):
    """The unified interface for creating, running, and assessing analyses that combine
    ORBIT, WOMBAT, and FLORIS.

    Args:
        library_path(:obj:`str` | :obj:`pathlib.Path`): The file path where the configuration data
            for ORBIT, WOMBAT, and FLORIS can be found.
        weather(:obj:`str` | :obj:`pathlib.Path`): The file path where the weather
            profile data is located. The following columns must exist in the data: datetime, windspeed, wave_height, and wind_direction
        orbit_config(:obj:`str` | :obj:`pathlib.Path`): The ORBIT configuration file name or dictionary.
        wombat_config(:obj:`str` | :obj:`pathlib.Path`): The WOMBAT configuration file name or dictionary.
        floris_config(:obj:`str` | :obj:`pathlib.Path`): The FLORIS configuration file name or dictionary.
    """
    library_path: str | Path = field(converter=resolve_path)
    weather: str | Path | pd.DataFrame = field(converter=load_weather)
    orbit_config: str | Path | dict
    wombat_config: str | Path | dict
    floris_config: str | Path | dict

    # Internally created attributes, aka, no user inputs to these
    orbit_config_dict: dict = field(factory=dict, init=False)
    wombat: Simulation = field(init=False)
    orbit: ProjectManager = field(init=False)
    floris: FlorisInterface = field(init=False)
    floris_wind_rose: WindRose = field(init=False)
    aep: float = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.setup_orbit()
        self.setup_wombat()
        self.setup_floris()

    @library_path.validator
    def library_exists(self, attribute: attrs.Attribute, value: str | Path) -> None:
        """Validates that the user input to :py:attr:`library_path` is a valid directory.

        Args:
            attribute (attrs.Attribute): The attrs Attribute information/metadata/configuration.
            value (str | Path): The user input.

        Raises:
            FileNotFoundError: Raised if :py:attr:`value` does not exist.
            ValueError: Raised if the :py:attr:`value` exists, but is not a directory.
        """
        if not value.exists():
            raise FileNotFoundError(f"The input path to {attribute.name} cannot be found: {value}")
        if not value.is_dir():
            raise ValueError(f"The input path to {attribute.name}: {value} is not a directory.")

    def setup_orbit(self) -> None:
        """Creates the ORBIT Project Manager object and readies it for running an analysis."""
        self.orbit_config = self.library_path / "project/config" / self.orbit_config
        self.orbit_config_dict = load_config(self.orbit_config)
        self.orbit = ProjectManager(self.orbit_config_dict, library_path=str(self.library_path), weather=self.weather)

    def setup_wombat(self) -> None:
        """Creates the WOMBAT Simulation object and readies it for running an analysis."""
        self.wombat_config = self.library_path / "project/config" / self.wombat_config
        self.wombat = Simulation.from_config(self.wombat_config)

    def setup_floris(self, wind_rose: WindRose | None = None) -> None:
        """Creates the FLORIS FlorisInterface object and readies it for running an
        analysis.

        Args:
            wind_rose (WindRose | None, optional): A custom ``WindRose`` object.
            Defaults to None.
        """
        self.floris_config = self.library_path / "project/config" / self.floris_config

        if wind_rose is None:
            weather = self.wombat.env.weather
            wind_rose = WindRose()
            wind_rose_df = wind_rose.make_wind_rose_from_user_data(weather.wind_direction, weather.windspeed)
        
        self.floris_wind_rose = wind_rose
        self.floris = FlorisInterface(configuration=self.floris_config)

    def run_floris(self, which: str, floris_kwargs: dict) -> None:
        if which == "wind_rose":
            self.aep = self.floris.get_farm_AEP_wind_rose_class(wind_rose=self.floris_wind_rose, **floris_kwargs)
        elif which == "time_series":
            # Calculate the AEP
            self.floris.reinitialize(
                time_series=True,
                wind_directions=self.wombat.env.weather.wind_direction.values,
                wind_speeds=self.wombat.env.weather.windspeed.values,
            )
            self.floris.calculate_wake()
            self.floris_turbine_powers = self.floris.get_turbine_powers()

            self.aep = self.floris_turbine_powers.sum()
            print("WARNING: FLORIS TIME SERIES RESULTS ARE JUST SUMMED AT THE CURRENT MOMENT")
            # TODO: Calculate the availability x turbine powers based on coordinate re-matching

        else:
            raise ValueError(f"`which` must be one of: 'wind_rose' or 'time_series', not: {which}")

    def run(self, which_floris: str, floris_kwargs: dict) -> None:
        """Run all three models in serial."""
        if which_floris not in ("wind_rose", "time_series"):
            raise ValueError(f"`which_floris` must be one of: 'wind_rose' or 'time_series', not: {which_floris}")
        self.orbit.run()
        self.wombat.run()
        self.run_floris(which_floris, floris_kwargs)

    def reinitialize(
        self,
        orbit_config: str | Path | dict | None = None,
        wombat_config: str | Path | dict | None = None,
        floris_config: str | Path | dict | None = None,
        floris_wind_rose: WindRose | None = None,
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
            if floris_wind_rose is not None:
                self.setup_floris(wind_rose=floris_wind_rose)
            else:
                self.setup_floris()

        if floris_config is None and floris_wind_rose is not None:
            self.setup_floris(wind_rose=floris_wind_rose)
