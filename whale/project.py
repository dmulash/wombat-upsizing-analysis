from lib2to3.pytree import convert
from pathlib import Path

import attrs
import pandas as pd
from attrs import define, field

from ORBIT import ProjectManager, load_config
from wombat.core import Simulation
from floris.tools import FlorisInterface
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
    if isinstance(value, (str, Path)):
        value = load_yaml(resolve_path(value))
    if isinstance(value, dict):
        return value
    raise TypeError(
        (
            f"The input configuration is not a valid format: {type(value)}. Inputs must"
            " be of type `str`, `Path`, or `dict`."
        )
    )


@define
class Project(FromDictMixin):
    """The unified interface for creating, running, and assessing analyses that combine
    ORBIT, WOMBAT, and FLORIS.

    Args:
        library_path(:obj:`str` | :obj:`pathlib.Path`): The file path where the configuration data
            for ORBIT, WOMBAT, and FLORIS can be found.
        weather(:obj:`str` | :obj:`pathlib.Path`): The file path where the weather
            profile data is located.
        orbit_config(:obj:`str` | :obj:`pathlib.Path`): The ORBIT configuration file name or dictionary.
        wombat_config(:obj:`str` | :obj:`pathlib.Path`): The WOMBAT configuration file name or dictionary.
        floris_config(:obj:`str` | :obj:`pathlib.Path`): The FLORIS configuration file name or dictionary.
    """
    library_path: str | Path = field(converter=resolve_path)
    weather: str | Path = field(converter=resolve_path)
    orbit_config: str | Path | dict
    wombat_config: str | Path | dict
    floris_config: str | Path | dict

    # Internally created attributes, aka, no user inputs to these
    orbit_config_dict: dict = field(factory=dict, init=False)
    wombat: Simulation = field(init=False)
    orbit: ProjectManager = field(init=False)
    floris: FlorisInterface = field(init=False)

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
        self.orbit = ProjectManager(self.orbit_config_dict, library_path=str(self.library_path), weather=pd.read_csv(self.weather))

    def setup_wombat(self) -> None:
        """Creates the WOMBAT Simulation object and readies it for running an analysis."""
        self.wombat_config = self.library_path / "project/config" / self.wombat_config
        self.wombat = Simulation.from_config(self.wombat_config)

    def setup_floris(self) -> None:
        """Creates the FLORIS FlorisInterface object and readies it for running an analysis."""
        self.floris_config = self.library_path / "project/config" / self.floris_config
        self.floris = FlorisInterface(configuration=self.floris_config)

    def run_capex_opex(self) -> None:
        self.orbit.run()
        self.wombat.run()
