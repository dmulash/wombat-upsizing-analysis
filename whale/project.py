from lib2to3.pytree import convert
from pathlib import Path
from typing import Type

import attrs
from attrs import define, field
from numpy import isin

from orbit import ProjectManager
from wombat.core import Simulation
from wombat.core.library import load_yaml
from wombat.core.data_classes import FromDictMixin
from floris.tools import FlorisInterface


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


def read_config(value: str | Path | dict | ProjectManager | Simulation | FlorisInterface) -> dict | ProjectManager | Simulation | FlorisInterface:
    if isinstance(value, (ProjectManager, Simulation, FlorisInterface, dict)):
        return value
    if isinstance(value, (str, Path)):
        return resolve_path(value)
    raise TypeError(
        (
            f"The input configuration is not a valid format: {type(value)}. Inputs must"
            " be of type `str`, `Path`, `dict`, or simulation specific: "
            "`orbit.ProjectManager`, `wombat.core.Simulation`, or "
            "`floris.tools.FlorisInterface`."
        )
    )


@define
class Project(FromDictMixin):
    library_path: str | Path = field(converter=resolve_path)
    weather: str | Path = field(converter=resolve_path)
    orbit_config: Path | dict | ProjectManager = field(converter=read_config)
    wombat_config: Path | dict | Simulation = field(converter=read_config)
    floris_config: Path | dict | FlorisInterface = field(converter=read_config)

    def __attrs_post_init__(self) -> None:
        self.setup_orbit()
        self.setup_wombat()
        self.setup_floris()

    @library_path.validator
    def library_exists(self, attribute: attrs.Attribute, value: str | Path) -> None:
        if not value.exists():
            raise FileNotFoundError(f"The input path to {attribute.name} cannot be found: {value}")
        if not value.is_dir():
            raise ValueError(f"The input path to {attribute.name}: {value} is not a directory.")

    def setup_orbit(self):
        ...

    def setup_wombat(self):
        ...

    def setup_floris(self):
        ...

