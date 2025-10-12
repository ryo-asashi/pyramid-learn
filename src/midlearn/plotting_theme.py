# src/midlearn/plotting_theme.py

from __future__ import annotations

from dataclasses import dataclass, InitVar, KW_ONLY
from plotnine.scales.scale_continuous import scale_continuous
from plotnine.scales.scale_discrete import scale_discrete
from plotnine._utils.registry import alias
from mizani.bounds import rescale_mid
from typing import Literal
from . import _r_interface

class color_theme(object):
    def __init__(self, theme: str | color_theme, **kwargs):
        self._obj = _r_interface._call_r_color_theme(
            theme = theme,
            **kwargs
        )
        self.name = self._obj['name'][0]
        self.type = self._obj['type'][0]
        self._ramp = self._obj['ramp']
        self._palette = self._obj['palette']
    
    def palette(self, n: int) -> list[str]:
        if not isinstance(n, int):
            n = int(n)
        return list(self._palette(n))

    def ramp(self, x: float | list[float]) -> list[str]:
        if isinstance(x, float):
            x = [x]
        x = _r_interface._as_r_vector(x, mode='numeric')
        return list(self._ramp(x))

    def __repr__(self) -> str:
        return f"<color_theme name='{self.name}' type='{self.type}'>"


def scale_color_theme(
    theme: str | color_theme,
    midpoint: float = 0,
    **kwargs
) -> scale_color_theme_d | scale_color_theme_c:
    theme = color_theme(theme=theme)
    if theme.type == 'qualitative':
        return scale_color_theme_d(theme=theme, **kwargs)
    return scale_color_theme_c(theme=theme, midpoint=midpoint, **kwargs)


def scale_fill_theme(
    theme: str | color_theme,
    **kwargs
) -> scale_fill_theme_d | scale_fill_theme_c:
    theme = color_theme(theme=theme)
    if theme.type == 'qualitative':
        return scale_fill_theme_d(theme=theme, **kwargs)
    return scale_fill_theme_c(theme=theme, **kwargs)


#alias
scale_colour_theme = scale_color_theme


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
        if theme.type == 'diverging':
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
