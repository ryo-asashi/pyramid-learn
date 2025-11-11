<!-- README.md is generated from README.ipynb. Please edit that file -->

# pyramid-learn <img src="docs/logo/logo_hex.png" align="right" height="138"/>

A [{rpy2}](https://rpy2.github.io/doc/latest/html/)-based Python wrapper for the [{midr}](https://ryo-asashi.github.io/midr/) R package to explain black-box models, with a [{scikit-learn}](https://scikit-learn.org/stable/) compatible API.

The goal of {midr} is to provide a model-agnostic method for interpreting and explaining black-box predictive models by creating a globally interpretable surrogate model.
The package implements 'Maximum Interpretation Decomposition' (MID), a functional decomposition technique that finds an optimal additive approximation of the original model.
This approximation is achieved by minimizing the squared error between the predictions of the black-box model and the surrogate model.
The theoretical foundations of MID are described in Iwasawa & Matsumori (2025) \[Forthcoming\], and the package itself is detailed in [Asashiba et al. (2025)](https://arxiv.org/abs/2506.08338).

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/ryo-asashi/pyramid-learn.git
```

## Features

- **Scikit-learn Compatible API**: Fits seamlessly into your existing 'scikit-learn' workflows with a familiar .fit() and .predict() interface.

- **Model-Agnostic IML**: Explains any black-box model, from complex neural networks to gradient boosting machines.

- **Global Interpretability**: Generates a simple, additive surrogate model (MID) that provides a global understanding of the black-box model's behavior.

- **Direct Visualizations**: Easily creates plots for feature importance, component functions (dependence), prediction breakdowns, and conditional expectations using a plotnine-based interface.

## Requirements

This package is a {rpy2}-based Python wrapper and requires a working R installation on your system, as well as the {midr} R package.

You can install the R package from CRAN by running the following command in your R console:

```r
install.packages('midr')
```

## Quick Start

Here’s a basic example of how to use **pyramid-learn** (namespace: **midlearn**) to explain a trained LightGBM model.


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.datasets import fetch_openml
from sklearn import set_config

import lightgbm as lgb
import midlearn as mid 

# Set up plotnine theme for clean visualizations
import plotnine as p9
p9.theme_set(p9.theme_bw(base_family='serif'))
p9.options.figure_size = (5, 4)

# Configure scikit-learn display
set_config(display='text')
```

    Error importing in API mode: ImportError('On Windows, cffi mode "ANY" is only "ABI".')
    Trying to import in ABI mode.
    

## 1. Train a Black-Box Model
We use the California Housing dataset to train a LightGBM Regressor, which will serve as our black-box model.


```python
# Load and prepare data
bikeshare = fetch_openml(data_id=42712)
X = pd.DataFrame(bikeshare.data, columns=bikeshare.feature_names)
y = bikeshare.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit a LightGBM regression model
estimator = lgb.LGBMRegressor(
    force_col_wise=True,
    n_estimators=500,
    random_state=42
)
estimator.fit(X_train, y_train)
```

    [LightGBM] [Info] Total Bins 283
    [LightGBM] [Info] Number of data points in the train set: 13034, number of used features: 12
    [LightGBM] [Info] Start training from score 190.379623
    




    LGBMRegressor(force_col_wise=True, n_estimators=500, random_state=42)




```python
model_pred = estimator.predict(X_test)
rmse = root_mean_squared_error(model_pred, y_test)
print(f"RMSE: {round(rmse, 6)}")
```

    RMSE: 37.615267
    

## 2. Create an Explaination Model
We fit the `MIDExplainer` to the training data to create a globally faithful, interpretable surrogate model (MID).


```python
# Initialize and fit the MID model
explainer = mid.MIDExplainer(
    estimator=estimator,
    penalty=.05,
    singular_ok=True,
    interactions=True,
    encoding_frames={'hour':list(range(24))}
)
explainer.fit(X_train)
```

    Generating predictions from the estimator...
    

    R callback write-console: singular fit encountered
      
    




    MIDExplainer(encoding_frames={'hour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                           23]},
                 estimator=LGBMRegressor(force_col_wise=True, n_estimators=500,
                                         random_state=42),
                 penalty=0.05, singular_ok=True)




```python
# Check the fidelity of the surrogate model to the original model
p = p9.ggplot() \
    + p9.geom_abline(slope=1, color='gray') \
    + p9.geom_point(p9.aes(estimator.predict(X_test), explainer.predict(X_test)), alpha=0.5, shape=".") \
    + p9.labs(
        x='Prediction (LightGBM Regressor)',
        y='Prediction (Surrogate MID Regressor)',
        title='Surrogate Model Fidelity Check',
        subtitle=f'R-squared score: {round(explainer.fidelity_score(X_test), 6)}',
    )
p
```

    Generating predictions from the estimator...
    


    
![png](README_files/README_11_1.png)
    


## 3. Visualize the Explanation Model
The MID model allows for clear visualization of feature importance, individual effects, and local prediction breakdowns.


```python
# Calculate and plot overall feature importance (default bar plot and heatmap)
imp = explainer.importance()
display(
    imp.plot(max_nterms=20) +
    p9.ggtitle("Importance Plot")
)
display(
    imp.plot(style='heatmap', color='black', linetype='dotted') +
    p9.ggtitle("Importance Heatmap") +
    p9.theme(legend_key_height=225)
)
```


    
![png](README_files/README_13_0.png)
    



    
![png](README_files/README_13_1.png)
    



```python
# Plot the top 3 important main effects (Component Functions)
for i, t in enumerate(imp.terms(interactions=False)[:3]):
    p = (
        explainer.plot(term=t) +
        p9.ggtitle(f"Main Effect of {t.capitalize()}")
    )
    display(p)
```


    
![png](README_files/README_14_0.png)
    



    
![png](README_files/README_14_1.png)
    



    
![png](README_files/README_14_2.png)
    



```python
# Plot the interaction of pairs of variables (Component Functions)
display(
    explainer.plot(
        "hour:workingday",
        theme='mako',
        main_effects=True
    ) +
    p9.ggtitle("Total Effect of Hour and Workingday") +
    p9.theme(legend_key_height=225)
)
display(
    explainer.plot(
        "hour:feel_temp",
        style='data',
        theme='mako',
        data=X_train,
        main_effects=True
    ) +
    p9.ggtitle("Total Effect of Hour and Feeling Temperature") +
    p9.theme(legend_key_height=225)
)
```


    
![png](README_files/README_15_0.png)
    



    
![png](README_files/README_15_1.png)
    



```python
# Plot prediction breakdowns for the first three test samples (Local Interpretability)
for i in range(3):
    p = (
        explainer.breakdown(row=i, data=X_test).plot() +
        p9.ggtitle(f"Breakdown Plot for Row {i}")
    )
    display(p)
```


    
![png](README_files/README_16_0.png)
    



    
![png](README_files/README_16_1.png)
    



    
![png](README_files/README_16_2.png)
    



```python
# Plot individual conditional expectations (ICE) with color encoding
ice = explainer.conditional(
    variable='hour',
    data=X_train.head(500)
)
display(
    ice.plot(alpha=.1) +
    p9.ggtitle("ICE Plot of Hour")
)
display(
    ice.plot(
        style='centered',
        var_color='workingday',
        theme='muted'
    ) +
    p9.labs(
        title="Centered ICE Plot of Hour",
        subtitle="Colored by the value of Workingday"
    ) +
    p9.theme(legend_position="bottom")
)
```


    
![png](README_files/README_17_0.png)
    



    
![png](README_files/README_17_1.png)
    

