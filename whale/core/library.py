"""Provides a consistent way to read and write YAML data, largely based on WOMBAT's
library module.

All library data should adhere to the followind directory structure where <library>
signifies the user's input library path:
```
<library>
  ├── project
    ├── config     <- Project-level configuration files
    ├── port       <- Port configuration files
    ├── plant      <- Wind farm layout files
  ├── cables       <- Export and Array cable configuration files
  ├── substations  <- Substation configuration files
  ├── turbines     <- Turbine configuration and power curve files
  ├── vessels      <- Land-based and offshore servicing equipment configuration files
  ├── weather      <- Weather profiles
  ├── results      <- The analysis log files and any saved output data
```
"""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path

import yaml


# YAML SafeLoader that is able to read scientific notation and Python Tuples
class CustomSafeLoader(yaml.SafeLoader):
    """Customized ``yaml.SafeLoader`` that adds custom constructors for consistent
    data loading in safe mode.
    """

    def construct_python_tuple(self, node):
        """Loads a YAML object to a Pytho Tuple.s"""
        return tuple(self.construct_sequence(node))


custom_loader = yaml.SafeLoader
custom_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)
custom_loader.add_constructor(
    "tag:yaml.org,2002:python/tuple", CustomSafeLoader.construct_python_tuple
)


def load_yaml(path: str | Path, fname: str | Path) -> Any:
    """Loads and returns the contents of the YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the file to be loaded.
    fname : str | Path
        Name of the file (ending in .yaml) to be loaded.

    Returns
    -------
    Any
        Whatever content is in the YAML file.
    """
    path = Path(path).resolve()
    return yaml.load(open(path / fname, "r"), Loader=custom_loader)
