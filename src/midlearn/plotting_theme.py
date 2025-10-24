# src/midlearn/plotting_theme.py

from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, InitVar, KW_ONLY

from plotnine.scales.scale_continuous import scale_continuous
from plotnine.scales.scale_discrete import scale_discrete
from plotnine._utils.registry import alias
from mizani.bounds import rescale_mid

from . import _r_interface
from . import utils

class color_theme(object):
    """Color themes for graphics.
    """
    def __init__(
        self,
        theme: str | list[str] | color_theme,
        theme_type: Literal['diverging', 'qualitative', 'sequential'] | None = None,
        **kwargs
    ):
        """Initialize color theme object

        Parameters
        ----------
        theme : str or list of str or color_theme
            The name of the theme (str), a list of custom colors (list[str]),
            or an existing color_theme object.
        theme_type: {'diverging', 'qualitative', 'sequential'}, default 'sequential'
            The type of the color theme.
        **kwargs : dict
            Additional keyword arguments passed to the `midr::color.theme()` function in R.
        """
        if theme_type is not None:
            theme_type = utils.match_arg(theme_type, ['diverging', 'qualitative', 'sequential'])
        self._obj = _r_interface._call_r_color_theme(
            theme = theme,
            theme_type = theme_type,
            **kwargs
        )
        self.name = self._obj['name'][0]
        self.type = self._obj['type'][0]
        self.theme_type = self.type # alias
        self._ramp = self._obj['ramp']
        self._palette = self._obj['palette']
    
    def palette(
        self,
        n: int
    ) -> list[str]:
        """Return a list of colors of specified length.

        Parameters
        ----------
        n : int
            The number of colors to retrieve from the color palette.

        Returns
        -------
        list[str]
            A list of hexadecimal color codes.
        """
        if not isinstance(n, int):
            n = int(n)
        return [_r_interface._convert_r_color(v) for v in self._palette(n)]

    def ramp(
        self,
        x: float | list[float]
    ) -> list[str]:
        """Return a list of colors based on the specified list of values.

        Parameters
        ----------
        x : float or list of float
            A number, or list of numbers, in the range [0, 1].

        Returns
        -------
        list[str]
            A list of hexadecimal color codes.
        """
        if isinstance(x, float):
            x = [x]
        x = _r_interface._as_r_vector(x, mode='numeric')
        return [_r_interface._convert_r_color(v) for v in self._ramp(x)]

    def __repr__(self) -> str:
        return f"<color_theme name='{self.name}' theme_type='{self.theme_type}'>"


def scale_color_theme(
    theme: str | list[str] | color_theme,
    midpoint: float = 0,
    **kwargs
) -> scale_color_theme_d | scale_color_theme_c:
    """Scale for 'color' aesthetics of plotnine graphics.

    Generate either a discrete or continuous color scale depending on the theme type.

    Parameters
    ----------
    theme : str or list of str or color_theme
        The name of the theme (str), a list of custom colors (list[str]),
        or an existing color_theme object.
    midpoint : float, default 0
        The value used as the center for the 'diverging' color scale.
    **kwargs : dict
        Additional keyword arguments passed to the constructor of scales.
    
    Returns
    -------
    scale_color_theme_d or scale_color_theme_c
        A plotnine color scale object based on the theme type.

    See Also
    --------
    scale_fill_theme : The corresponding scale function for the 'fill' aesthetic.

    Notes
    -----
    If the theme is 'qualitative', it returns :class:`scale_color_theme_d`.
    Otherwise, it returns :class:`scale_color_theme_c`.
    """
    theme = color_theme(theme=theme)
    if theme.theme_type == 'qualitative':
        return scale_color_theme_d(theme=theme, **kwargs)
    return scale_color_theme_c(theme=theme, midpoint=midpoint, **kwargs)


def scale_fill_theme(
    theme: str | list[str] | color_theme,
    midpoint: float = 0,
    **kwargs
) -> scale_fill_theme_d | scale_fill_theme_c:
    """Scale for 'fill' aesthetics of plotnine graphics.

    Generate either a discrete or continuous fill scale depending on the theme type.

    Parameters
    ----------
    theme : str or list of str or color_theme
        The name of the theme (str), a list of custom colors (list[str]),
        or an existing color_theme object.
    midpoint : float, default 0
        The value used as the center for the 'diverging' fill scale.
    **kwargs : dict
        Additional keyword arguments passed to the constructor of scales.
    
    Returns
    -------
    scale_fill_theme_d or scale_fill_theme_c
        A `plotnine` color scale object based on the theme type.

    See Also
    --------
    scale_color_theme : The corresponding scale function for the 'color' aesthetic.
    """
    theme = color_theme(theme=theme)
    if theme.theme_type == 'qualitative':
        return scale_fill_theme_d(theme=theme, **kwargs)
    return scale_fill_theme_c(theme=theme, midpoint=midpoint, **kwargs)


#alias
scale_colour_theme = scale_color_theme
"""Alias for :func:`~midlearn.plotting_theme.scale_color_theme`.
"""

@dataclass
class scale_color_theme_d(scale_discrete):
    theme: InitVar[color_theme]
    _aesthetics = ['color']
    _: KW_ONLY
    na_value: str = '#7F7F7F'
    def __post_init__(self, theme: color_theme):
        super().__post_init__()
        self.palette = theme.palette


@dataclass
class scale_color_theme_c(scale_continuous):
    theme: InitVar[str | color_theme]
    midpoint: InitVar[float] = 0
    _aesthetics = ['color']
    _: KW_ONLY
    guide: Literal['legend', 'colorbar'] | None = 'colorbar'
    na_value: str = '#7F7F7F'
    def __post_init__(self, theme: str | color_theme, midpoint: float):
        super().__post_init__()
        theme = color_theme(theme)
        if theme.theme_type == 'diverging':
            def _rescale_mid(*args, **kwargs):
                return rescale_mid(*args, mid=midpoint, **kwargs)
            self.rescaler = _rescale_mid
        self.palette = theme.ramp


@dataclass
class scale_fill_theme_d(scale_color_theme_d):
    _aesthetics = ['fill']


@dataclass
class scale_fill_theme_c(scale_color_theme_c):
    _aesthetics = ['fill']


@alias
class scale_colour_theme_d(scale_color_theme_d):
    pass


@alias
class scale_colour_theme_c(scale_color_theme_c):
    pass
