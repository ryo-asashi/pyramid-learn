# src/midlearn/plotting.py

from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from .api import (
        MIDRegressor, 
        MIDExplainer, 
        MIDImportance, 
        MIDBreakdown, 
        MIDConditional
    )

import numpy as np
import pandas as pd
import plotnine as p9

from . import plotting_theme as pt
from . import utils

def plot_effect(
    estimator: MIDRegressor | MIDExplainer,
    term: str,
    style: Literal['effect', 'data'] = 'effect',
    theme: str | pt.color_theme | None = None,
    intercept: bool = False,
    main_effects: bool = False,
    data: pd.DataFrame | None = None,
    jitter: float | list[float] = 0.3,
    resolution: int | tuple[int, int] = 100,
    **kwargs
):
    """Visualize the estimated main or interaction effect of a fitted MID model with plotnine.
    This is a porting function for the R function `midr::ggmid.mid()`.

    Parameters
    ----------
    estimator : MIDRegressor or MIDExplainer
        A fitted MIDRegressor or MIDExplainer object containing the model components.
    term : str
        The name of the component function (main effect or interaction term) to plot.
    style : {'effect', 'data'}, default 'effect'
        The plotting style. 
        'effect' plots the estimated component function as a line or a surface. 
        'data' plots the specified data points (jittered for factor variables) with MID values represented by color.
    theme : str or pt.color_theme or None, default None
        The color theme to use for the plot.
    intercept : bool, default False
        If True, the global intercept term is added to the component function values.
    main_effects : bool, default False
        If True, main effects are included when plotting two-way interaction terms.
        Ignored for single-term plots.
    data : pandas.DataFrame or None, default None
        The data frame to plot. Required only if `style='data'`.
    jitter : float or list of float, default 0.3
        The amount of jitter to apply to factor variables when `style='data'` is used.
    resolution : int or tuple[int, int], default 100
        The resolution (number of grid points) for calculating the effect. 
        If a single integer, it is used for both axes of a 2D interaction plot. 
        If a tuple (int, int), it specifies the resolution for the first and 
        second predictor in an interaction, respectively.
    **kwargs : dict
        Additional keyword arguments passed to the main layer of the plot.
    
    Returns
    -------
    plotnine.ggplot.ggplot
        A plotnine object representing the visualization of the component function.
    """
    style = utils.match_arg(style, ['effect', 'data'])
    tags = term.split(':')
    if style == 'data':
        if not isinstance(jitter, list):
            jitter = [jitter] * len(tags)
        if data is None:
            raise ValueError("The 'data' argument is required when style='data'. Please provide the pandas.DataFrame to use for plotting.")
        data = data.copy()
        terms = [term] + (tags if (len(tags) == 2 and main_effects) else [])
        data['mid'] = (
            estimator.r_predict(X=data, output_type='terms', terms=terms).sum(axis=1)
            + (estimator.intercept if intercept else 0)
        )
    if len(tags) == 1:
        eff_df = estimator.main_effects(term)
        if intercept:
            eff_df['mid'] += estimator.intercept
        p = p9.ggplot(data=eff_df, mapping=p9.aes(x=term, y='mid'))
        enc = estimator._encoding_type(tag=term, order=1)
        if style == 'effect':
            if enc == 'linear':
                p = p + p9.geom_line(**kwargs)
                if theme is not None:
                    p = p + p9.aes(color='mid') + pt.scale_color_theme(theme)
            elif enc == 'constant':
                xval = eff_df[[f'{term}_min', f'{term}_max']].to_numpy().ravel('C')
                yval = np.repeat(eff_df['mid'].to_numpy(), 2)
                path_df = pd.DataFrame({term: xval, 'mid': yval})
                p += p9.geom_path(data=path_df, **kwargs)
                if theme is not None:
                    p = p + p9.aes(color='mid') + pt.scale_color_theme(theme)
            else:
                p += p9.geom_col(**kwargs)
                if theme is not None:
                    p = p + p9.aes(fill='mid') + pt.scale_fill_theme(theme)
        if style == 'data':
            jit = jitter[0] if enc == 'factor' else 0
            p += p9.geom_jitter(p9.aes(y = "mid"), data=data, width=jit, height=0, **kwargs)
            if theme is not None:
                p = p + p9.aes(color='mid') + pt.scale_color_theme(theme)
    elif len(tags) == 2:
        xtag, ytag = tags[0], tags[1]
        try:
            eff_df = estimator.interactions(term)
        except KeyError as e:
            try:
                eff_df = estimator.interactions(f'{ytag}:{xtag}')
            except KeyError:
                raise e
        if intercept:
            eff_df['mid'] += estimator.intercept
        if main_effects:
            eff_df['mid'] += estimator.effect(term=xtag, x=eff_df) + estimator.effect(term=ytag, x=eff_df)
        p = p9.ggplot(eff_df, p9.aes(x=xtag, y=ytag))
        xenc = estimator._encoding_type(tag=xtag, order=2)
        yenc = estimator._encoding_type(tag=ytag, order=2)
        if style == 'effect':
            xres, yres = (resolution, resolution) if isinstance(resolution, int) else (resolution, resolution)
            if xenc == 'factor':
                xval = eff_df[xtag].unique()
            else:
                xmin, xmax = eff_df[f'{xtag}_min'].min(), eff_df[f'{xtag}_max'].max()
                xval = np.linspace(xmin, xmax, xres)
            if yenc == 'factor':
                yval = eff_df[ytag].unique()
            else:
                ymin, ymax = eff_df[f'{ytag}_min'].min(), eff_df[f'{ytag}_max'].max()
                yval = np.linspace(ymin, ymax, yres)
            grid_df = pd.DataFrame({
                xtag: np.repeat(xval, len(yval)),
                ytag: np.tile(yval, len(xval))
            })
            grid_df['mid'] = estimator.effect(term=term, x=grid_df)
            if intercept:
                grid_df['mid'] += estimator.intercept
            if main_effects:
                grid_df['mid'] += estimator.effect(term=xtag, x=grid_df) + estimator.effect(term=ytag, x=grid_df)
            p += p9.geom_raster(p9.aes(x=xtag, y=ytag, fill='mid'), data=grid_df)
            p += pt.scale_fill_theme(theme if theme is not None else 'midr')
        if style == 'data':
            xjit = jitter[0] if xenc == 'factor' else 0
            yjit = jitter[1] if yenc == 'factor' else 0
            p += p9.geom_jitter(
                mapping=p9.aes(color='mid'), data=data, width=xjit, height=yjit, **kwargs
            )
            if theme is not None:
                p += pt.scale_color_theme(theme)
            else:
                p += p9.scale_color_continuous()
    return p


def plot_importance(
    importance: MIDImportance,
    style: Literal['barplot', 'heatmap'] = 'barplot',
    theme: str | pt.color_theme | None = None,
    max_nterms: int | None = 30,
    **kwargs
):
    """Visualize the importance scores of the component functions from a fitted MID model with plotnine.
    This is a porting function for the R function `midr::ggmid.mid.importance()`.

    Parameters
    ----------
    importance : MIDImportance
        A fitted :class:`MIDImportance` object containing the component importance scores.
    style : {'barplot', 'heatmap'}, default 'barplot'
        The plotting style.
        'barplot' displays importance as horizontal bars, suitable for a large number of terms.
        'heatmap' displays importance in a matrix format, suitable for visualizing main effects and two-way interactions simultaneously.
    theme : str or pt.color_theme or None, default None
        The color theme to use for the plot.
    max_nterms : int or None, default 30
        The maximum number of terms to display when `style='barplot'`. 
        Terms are sorted by importance before truncation. If None, all terms are displayed.
    **kwargs : dict
        Additional keyword arguments passed to the main layer of the plot.

    Returns
    -------
    plotnine.ggplot.ggplot
        A plotnine object representing the visualization of component importance.
    """
    style = utils.match_arg(style, ['barplot', 'heatmap'])
    imp_df = importance.importance.copy()
    if style == 'barplot':
        if max_nterms is not None:
            imp_df = imp_df.head(max_nterms)
        p = (
            p9.ggplot(imp_df, p9.aes(x='term', y='importance'))
            + p9.geom_col(**kwargs)
            + p9.coord_flip()
            + p9.labs(x="")
        )
        if theme is not None:
            theme = pt.color_theme(theme)
            var_fill = 'order' if theme.type == 'qualitative' else 'importance'
            p = p + p9.aes(fill=var_fill) + pt.scale_fill_theme(theme)
    elif style == 'heatmap':
        terms = imp_df['term'].str.split(':', expand=True)
        if terms.shape[1] == 1:
            terms.loc[:, 1] = None
        terms[1] = terms[1].fillna(terms[0])
        df1 = pd.DataFrame({
            'x': terms[0], 'y':terms[1], 'importance': imp_df['importance']
        })
        df2 = pd.DataFrame({
            'x': terms[1], 'y':terms[0], 'importance': imp_df['importance']
        })
        df = pd.concat([df1, df2]).drop_duplicates(ignore_index=True)
        all_vars = pd.unique(np.concatenate([terms[0], terms[1]]))
        df['x'] = pd.Categorical(df['x'], categories=all_vars)
        df['y'] = pd.Categorical(df['y'], categories=all_vars)
        p = (
            p9.ggplot(df, p9.aes(x='x', y='y', fill='importance'))
            + p9.geom_tile(**kwargs)
            + p9.labs(x="", y="")
        )
        p += pt.scale_fill_theme(theme if theme is not None else 'grayscale')
    return p


def plot_breakdown(
    breakdown: MIDBreakdown,
    style: Literal['waterfall', 'barplot'] = 'waterfall',
    theme: str | pt.color_theme | None = None,
    max_nterms: int | None = 15,
    catchall: str = 'others',
    format: tuple[str, str] = ('%t=%v', '%t'),
    **kwargs
):
    """Visualize the decomposition of a single prediction into contributions from each component term with plotnine.
    This is a porting function for the R function `midr::ggmid.mid.breakdown()`.

    Parameters
    ----------
    breakdown : MIDBreakdown
        A fitted :class:`MIDBreakdown` object containing the term contributions for a specific data point.
    style : {'waterfall', 'barplot'}, default 'waterfall'
        The plotting style.
        'waterfall' displays contributions as a cascading plot, showing how each term adds to the final prediction, starting from the intercept.
        'barplot' displays contributions as simple horizontal bars, relative to zero.
    theme : str or pt.color_theme or None, default None
        The color theme to use for the plot.
    max_nterms : int or None, default 15
        The maximum number of terms to display. Terms beyond this limit are 
        grouped into a single 'catchall' category. If None, all terms are displayed.
    catchall : str, default 'others'
        The label used for the grouped category when the number of terms exceeds `max_nterms`.
    format : tuple[str, str], default ('%t=%v', '%t')
        A tuple of two format strings for labeling terms on the y-axis.
        The first string is for main effects (e.g., 'term=value'), and the second is for interaction terms (e.g., 'term').
        %t is replaced by the term name, and %v is replaced by the predictor value.
    **kwargs : dict
        Additional keyword arguments passed to the main layer of the plot.

    Returns
    -------
    plotnine.ggplot.ggplot
        A plotnine object representing the breakdown visualization.
    """
    style = utils.match_arg(style, ['waterfall', 'barplot'])
    brk_df = breakdown.breakdown.copy()
    if 'value' in brk_df.columns:
        def _format_row(row):
            _t = str(row['term'])
            _v = str(row['value'])
            fmt = format[1 if ':' in _t else 0]
            return fmt.replace('%t', _t).replace('%v', _v)
        brk_df['term'] = brk_df.apply(_format_row, axis=1)
    if max_nterms is not None and max_nterms < len(brk_df):
        resid = brk_df.iloc[max_nterms - 1:]['mid'].sum()
        brk_df = brk_df.head(max_nterms - 1)
        catchall_row = pd.DataFrame([{'term': catchall, 'mid': resid}])
        brk_df = pd.concat([brk_df, catchall_row], ignore_index=True)
    brk_df['term'] = pd.Categorical(
        brk_df['term'], categories=brk_df['term'].iloc[::-1]
    )
    if style == 'waterfall':
        intercept = breakdown.intercept
        cs = np.cumsum(np.r_[intercept, brk_df['mid']])
        brk_df['xmin'], brk_df['xmax'] = cs[:-1], cs[1:]
        brk_df['ymin'], brk_df['ymax'] = brk_df['term'].cat.codes + 1 - 0.4, brk_df['term'].cat.codes + 1 + 0.4
        brk_df['ymin2'] = (brk_df['ymin'] - 1).clip(lower=brk_df['ymin'].min())
        p = (
            p9.ggplot(brk_df, p9.aes(y='term'))
            + p9.geom_vline(xintercept=intercept, size=0.5)
            + p9.geom_rect(p9.aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax'), **kwargs)
            + p9.geom_linerange(p9.aes(x='xmax', ymax='ymax', ymin='ymin2'), size=0.5)
            + p9.labs(x='yhat')
            + p9.scale_y_discrete(name="")
        )
    elif style == 'barplot':
        p = (
            p9.ggplot(brk_df, p9.aes(x='term', y='mid'))
            + p9.geom_col(**kwargs)
            + p9.geom_hline(yintercept=0, linetype='dashed', color='#808080')
            + p9.coord_flip()
            + p9.labs(x="")
        )
    if theme is not None:
        theme = pt.color_theme(theme)
        if theme.type == 'qualitative':
            mid_sign = np.where(brk_df['mid'] > 0, '> 0', '< 0')
            p = p + p9.aes(fill=mid_sign) + pt.scale_fill_theme(theme) + p9.labs(fill='mid')
        else:
            p = p + p9.aes(fill='mid') + pt.scale_fill_theme(theme)
    return p


def plot_conditional(
    conditional: MIDConditional,
    style: Literal['ice', 'centered'] = 'ice',
    theme: str | pt.color_theme | None = None,
    var_color: str | None = None,
    dots: bool = True,
    reference: int = 0,
    **kwargs
):
    """Visualize Individual Conditional Expectation (ICE) plots or Centered ICE (c-ICE) plots with plotnine.
    This is a porting function for the R function `midr::ggmid.mid.conditional()`.

    Parameters
    ----------
    conditional : MIDConditional
        A fitted :class:`MIDConditional` object containing the ICE data.
    style : {'ice', 'centered'}, default 'ice'
        The plotting style.
        'ice' plots raw predicted values against the predictor variable.
        'centered' displays the **change in prediction** relative to a `reference` point,
        by subtracting the prediction at the `reference` point for each individual observation.
    theme : str or pt.color_theme or None, default None
        The color theme to use for the line colors.
    var_color : str or None, default None
        The name of a column (from the original data) to map to the color aesthetic of the ICE lines. This helps visualize heterogeneity.
    dots : bool, default True
        If True, plots points for the observed (original) predictions for each sample.
    reference : int, default 0
        The 0-indexed sample point used as the reference prediction for centering when `style='centered'` is used.
    **kwargs : dict
        Additional keyword arguments passed to the main layer of the plot.

    Returns
    -------
    plotnine.ggplot.ggplot
        A plotnine object representing the conditional expectation visualization.
    """
    style = utils.match_arg(style, ['ice', 'centered'])
    variable = conditional.variable
    obs_df = conditional.observed.copy()
    con_df = conditional.conditional.copy()
    if style == 'centered':
        values = conditional.values
        ref = values[min(len(values) - 1, max(0, reference))]
        ref_df = con_df.loc[con_df[variable] == ref, ['.id', 'yhat']].rename(columns={'yhat': 'yref'})
        obs_df = pd.merge(obs_df, ref_df, on='.id')
        con_df = pd.merge(con_df, ref_df, on='.id')
        obs_df['centered yhat'] = obs_df['yhat'] - obs_df['yref']
        con_df['centered yhat'] = con_df['yhat'] - con_df['yref']
    yvar = 'yhat' if style == 'ice' else 'centered yhat'
    p = (
        p9.ggplot(data=obs_df, mapping=p9.aes(x=variable, y=yvar))
        + p9.geom_line(p9.aes(group='.id'), data=con_df, **kwargs)
    )
    if dots:
        p += p9.geom_point()
    if var_color is not None:
        p += p9.aes(color=var_color)
    if theme is not None:
        p += pt.scale_color_theme(theme)
    return p
