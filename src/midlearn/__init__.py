# src/midlearn/__init__.py

from .api import MIDRegressor, MIDExplainer, MIDImportance, MIDBreakdown, MIDConditional
from .plotting import plot_effect, plot_importance, plot_breakdown, plot_conditional
from .plotting_theme import color_theme, scale_color_theme, scale_fill_theme, scale_colour_theme
from . import api
from . import plotting
from importlib.metadata import version, PackageNotFoundError

# plotting methods of each class
api.MIDRegressor.plot = plotting.plot_effect
api.MIDImportance.plot = plotting.plot_importance
api.MIDBreakdown.plot = plotting.plot_breakdown
api.MIDConditional.plot = plotting.plot_conditional

__all__ = [
    "MIDRegressor",
    "MIDExplainer",
    "MIDImportance",
    "MIDBreakdown",
    "MIDConditional",
    "plot_effect",
    "plot_importance",
    "plot_breakdown",
    "plot_conditional",
    "color_theme",
    "scale_color_theme",
    "scale_colour_theme",
    "scale_fill_theme"
]

try:
    __version__ = version("midlearn-learn")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
