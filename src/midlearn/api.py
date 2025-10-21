# src/midlearn/api.py

from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score

from . import _r_interface

class MIDRegressor(BaseEstimator, RegressorMixin):
    """Stand-alone Maximum Interpretation Decomposition regressor.
    """
    def __init__(
        self,
        params_main=None,
        params_inter=None,
        penalty=0,
        **kwargs
    ):
        """Create a MID model.

        Parameters
        ----------
        params_main : int, optional
            An integer specifying the maximum number of sample points for main effects.
            This corresponds to the 'k[1]' argument in R's `midr::interpret()`.
        params_inter : int, optional
            An integer specifying the maximum number of sample points for interactions.
            This corresponds to the 'k[2]' argument in R's `midr::interpret()`.
        penalty : float, optional
            The regularization penalty for pseudo-smoothing, corresponding to the
            'lambda' argument in R's `midr::interpret()`. Defaults to 0.
        **kwargs : dict
            Additional keyword arguments to be passed directly to the underlying
            `midr::interpret()` function for advanced fitting options.
        """
        self.params_main = params_main
        self.params_inter = params_inter
        self.penalty = penalty
        self.kwargs = kwargs

    def fit(
        self,
        X,
        y,
        sample_weight=None
    ) -> MIDRegressor:
        """Fit the MID model to the response y on predictors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used to train the MID model.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            The fitted estimator instance.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        self.mid_ = _r_interface._call_r_interpret(
            X=X,
            y=y,
            sample_weight=sample_weight,
            params_main=self.params_main,
            params_inter=self.params_inter,
            penalty=self.penalty,
            **self.kwargs
        )
        self.is_fitted_ = True
        return self

    def r_predict(
        self,
        X,
        output_type: str = 'response',
        terms: list[str] | None = None,
        **kwargs
    ) -> np.ndarray:
        """A low-level method to call the R predict.mid function.

        This method provides a direct interface to the R function, accepting common
        arguments explicitly and passing any others via keyword arguments.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            New data for which to make predictions.
        output_type : str, optional
            The type of prediction to return. Possible values are 'response', 'terms', or 'link'.
            Defaults to 'response'.
        terms : list of str, optional
            A list of specific term names to get predictions.
            If None, predictions for all terms are returned. Defaults to None.
        **kwargs : dict
            Additional keyword arguments to be passed directly to the underlying
            `midr::predict.mid()` function for advanced options.

        Returns
        -------
        np.ndarray
            The prediction result from R.
        """
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        try:
            X = X[self.feature_names_in_]
        except KeyError as e:
            missing_cols = set(self.feature_names_in_) - set(X.columns)
            raise ValueError(f"The following columns are missing: {list(missing_cols)}") from e
        res = _r_interface._call_r_predict(
            r_object=self.mid_,
            X=X,
            output_type=output_type,
            terms=terms,
            **kwargs
        )
        return np.asarray(res)

    def predict(
        self,
        X
    ) -> np.ndarray:
        """Predict target values for new data X using the fitted MID model.
        """
        return self.r_predict(X, type='response')

    def predict_terms(
        self,
        X
    ) -> np.ndarray:
        """Predict the contribution of each term for new data X.
        """
        return self.r_predict(X, type='terms')

    def effect(
        self,
        term: str,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | None = None    
    ) -> np.ndarray:
        """Evaluate a single MID component function for new data.

        Parameters
        ----------
        term : str
            The name of the model term to evaluate (e.g., 'x1', 'x1:x2').
        x : pd.DataFrame or np.ndarray
            New data for the first variable in the term. If a pd.DataFrame is
            provided, values of the related variables are extracted from it.
        y : np.ndarray, optional
            Values for the second variable in an interaction term. This is only
            required when evaluating a two-way interaction term.

        Returns
        -------
        np.ndarray
            A NumPy array of the calculated term contributions, with the same
            length as x.
        """
        check_is_fitted(self)
        res = _r_interface._call_r_mid_effect(
            r_object=self.mid_,
            term=term,
            x=x,
            y=y
        )
        return np.asarray(res)

    @property
    def intercept(self):
        """The intercept of the fitted model.
        """
        return _r_interface._extract_and_convert(r_object=self.mid_, name='intercept').item()

    @property
    def weights(self):
        """Sample weights used to fit the model.
        """
        return _r_interface._extract_and_convert(r_object=self.mid_, name='weights')

    @property
    def fitted_matrix(self):
        """A pandas DataFrame showing the breakdown of the fitted values into the effects of the component functions.
        """
        return _r_interface._extract_and_convert(r_object=self.mid_, name='fitted.matrix')

    @property
    def fitted_values(self):
        """A NumPy array of the fitted values.
        """
        return _r_interface._extract_and_convert(r_object=self.mid_, name='fitted.values')
    
    @property
    def residuals(self):
        """A NumPy array of the working residuals.
        """
        return _r_interface._extract_and_convert(r_object=self.mid_, name='residuals')

    @property
    def ratio(self):
        """The ratio of the sum of squared error between the target model predictions and the fitted values, to the sum of squared deviations of the target model predictions. Corresponds to 1 - R squared.
        """
        return _r_interface._extract_and_convert(r_object=self.mid_, name='ratio').item()

    def terms(self, **kwargs):
        """Extract term labels from the fitted model. See midr's mid.terms().
        """
        return list(_r_interface._call_r_mid_terms(r_object=self.mid_, **kwargs))

    def main_effects(self, term: str):
        """Extract a pd.DataFrame representing the main effect of the specified 'term'.
        """
        effects = _r_interface._extract_and_convert(r_object=self.mid_, name='main.effects')
        return _r_interface._extract_and_convert(r_object=effects, name=term)

    def interactions(self, term: str):
        """Extract a pd.DataFrame representing the interaction of the specified 'term'.
        """
        effects = _r_interface._extract_and_convert(r_object=self.mid_, name='interactions')
        return _r_interface._extract_and_convert(r_object=effects, name=term)
    
    def _encoding_type(self, tag: str, order: int = 1):
        obj = _r_interface._extract_and_convert(r_object=self.mid_, name='encoders')
        obj = _r_interface._extract_and_convert(r_object=obj, name='main.effects' if order == 1 else 'interactions')
        obj = _r_interface._extract_and_convert(r_object=obj, name=tag)
        return _r_interface._extract_and_convert(r_object=obj, name='type')[0]

    def importance(self, **kwargs):
        """Create MIDImportance object from the fitted estimator. Refer to midr's mid.importance().
        """
        return MIDImportance(estimator=self, **kwargs)

    def breakdown(self, **kwargs):
        """Create MIDBreakdown object from the fitted estimator. Refer to midr's mid.breakdown().
        """
        return MIDBreakdown(estimator=self, **kwargs)

    def conditional(self, variable: str, **kwargs):
        """Create MIDConditional object from the fitted estimator. Refer to midr's mid.conditional().
        """
        return MIDConditional(estimator=self, variable=variable, **kwargs)


class MIDExplainer(MIDRegressor, MetaEstimatorMixin):
    """Surrogate Maximium Interpretation Decomposition explainer.
    """
    def __init__(
        self,
        estimator,
        target_classes: str | list[str] | None = None,
        params_main=None,
        params_inter=None,
        penalty=0,
        **kwargs
    ):
        """Create a surrogate MID model to explain a pre-trained black-box model.
        
        Parameters
        ----------
        estimator : object
            The pre-trained black-box model to be explained.
        target_classes: list of str, optional
            For classification estimators only.
            Specifies the target class or classes for which the probability is to be explained.
            If a list is provided, the sum of probabilities is used.
            If None (the default), the model explains 1 - P(class 0).
        params_main : int, optional
            An integer specifying the maximum number of sample points for main effects.
            This corresponds to the 'k[1]' argument in R's `midr::interpret()`.
        params_inter : int, optional
            An integer specifying the maximum number of sample points for interactions.
            This corresponds to the 'k[2]' argument in R's `midr::interpret()`.
        penalty : float, optional
            The regularization penalty for pseudo-smoothing, corresponding to the
            'lambda' argument in R's `midr::interpret()`. Defaults to 0.
        **kwargs : dict
            Additional keyword arguments to be passed directly to the underlying
            `midr::interpret()` function for advanced fitting options.
        """
        self.estimator = estimator
        self.target_classes = target_classes
        if not is_classifier(self.estimator) and self.target_classes is not None:
            warnings.warn(
                "'target_classes' is specified but will be ignored because the estimator is not a classifier.",
                UserWarning
            )
        super().__init__(
            params_main = params_main,
            params_inter = params_inter,
            penalty = penalty,
            **kwargs
        )

    def _predict_y_estimator(
        self,
        X
    ) -> np.ndarray:
        """Generate a unified continuous prediction from the estimator, handling both classifiers and regressors.
        """
        if is_classifier(self.estimator):
            if not hasattr(self.estimator, "predict_proba"):
                raise TypeError("The provided estimator must have a 'predict_proba' method.")
            probas = self.estimator.predict_proba(X)
            if self.target_classes is not None:
                if not hasattr(self.estimator, 'classes_'):
                    raise TypeError(
                        "Estimator must have a 'classes_' attribute to use 'target_classes'."
                    )
                class_labels = np.asarray(self.estimator.classes_)
                target_classes = self.target_classes
                if not isinstance(target_classes, list):
                    target_classes = [target_classes]
                target_indices = np.where(np.isin(class_labels, target_classes))[0]
                if len(target_indices) != len(set(target_classes)):
                    raise ValueError(
                        "The 'target_classes' were not appropriately found in the estimator's classes."
                    )
                return probas[:, target_indices].sum(axis=1)
            else:
                return 1 - probas[:, 0]
        else:
            if not hasattr(self.estimator, "predict"):
                raise TypeError("The provided estimator must have a 'predict' method.")
            return self.estimator.predict(X)

    def fit(
        self,
        X,
        y=None,
        sample_weight=None
    ) -> MIDExplainer:
        """Fit the surrogate MID model to the predictions of the estimator on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used to train the MID model.
        y : array-like of shape (n_samples,), optional
            Predictions obtained from the original estimator. If None (the default),
            predictions are generated automatically from the estimator using X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            The fitted estimator (explainer) instance.
        """
        if y is None:
            print("Generating predictions from the estimator...")
            y = self._predict_y_estimator(X)
        super().fit(X=X, y=y, sample_weight=sample_weight)
        self.estimator_ = self.estimator
        return self
    
    def fidelity_score(
        self,
        X,
        y=None,
        sample_weight=None
    ) -> float:
        """Calculate the fidelity of the surrogate model.

        This score (R-squared) measures how well this explainer's predictions
        match the original estimator's predictions on the data X. A score
        close to 1.0 means the explainer is a faithful surrogate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to evaluate fidelity on.
        y : array-like of shape (n_samples,), optional
            Predictions obtained from the original estimator. If None (the default),
            predictions are generated automatically from the estimator using X.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights to apply when calculating the score. Defaults to None.

        Returns
        -------
        score : float
            The R-squared score between the original estimator's predictions and
            this surrogate model's predictions.
        """
        check_is_fitted(self)
        if y is None:
            print("Generating predictions from the estimator...")
            y = self._predict_y_estimator(X)
        y_explainer = self.predict(X)
        return r2_score(y_true=y, y_pred=y_explainer, sample_weight=sample_weight)



class MIDImportance(object):
    """MID Importance.

    This object is returned by the `MIDRegressor.importance()` method and holds
    the results of the feature importance calculation.
    """
    def __init__(
        self,
        estimator: MIDRegressor | MIDExplainer,
        **kwargs
    ):
        """Initialize the MIDImportance object.

        Parameters
        ----------
        estimator : MIDRegressor or MIDExplainer
            The fitted MID model instance from which to calculate importance.
        **kwargs : dict
            Additional keyword arguments passed to the `midr::mid.importance()` function in R.
        """
        self._obj = _r_interface._call_r_mid_importance(
            r_object = estimator.mid_,
            **kwargs
        )
    
    @property
    def importance(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='importance')

    @property
    def predictions(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='predictions')

    @property
    def measure(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='measure')

    def terms(self, **kwargs):
        return list(_r_interface._call_r_mid_terms(r_object=self._obj, **kwargs))


class MIDBreakdown(object):
    """MID Breakdown.

    This object is returned by the `MIDRegressor.breakdown()` method and provides
    a detailed breakdown of a single prediction.
    """
    def __init__(
        self,
        estimator: MIDRegressor | MIDExplainer,
        row: int | None = None,
        **kwargs
    ):
        """Initialize the MIDBreakdown object.

        Parameters
        ----------
        estimator : MIDRegressor or MIDExplainer
            The fitted MID model instance to use for the breakdown.
        row : int, optional
            The specific row index (observation) in the data for which
            to create the breakdown. If None (the default), the breakdown for the
            first instance is calculated.
        **kwargs : dict
            Additional keyword arguments passed to the `midr::mid.breakdown()` function in R.
        """
        self._obj = _r_interface._call_r_mid_breakdown(
            r_object = estimator.mid_,
            row = row,
            **kwargs
        )
    
    @property
    def breakdown(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='breakdown')

    @property
    def data(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='data')

    @property
    def intercept(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='intercept')

    @property
    def prediction(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='prediction')

    def terms(self, **kwargs):
        return list(_r_interface._call_r_mid_terms(r_object=self._obj, **kwargs))


class MIDConditional(object):
    """
    MID Conditional Expectations.

    This object is returned by the `MIDRegressor.conditional()` method and
    contains data for plotting conditional dependence.
    """
    def __init__(
        self,
        estimator: MIDRegressor | MIDExplainer,
        variable: str,
        **kwargs
    ):
        """Initialize the MIDConditional object.

        Parameters
        ----------
        estimator : MIDRegressor or MIDExplainer
            The fitted MID model instance to use.
        variable : str
            The name of the feature for which to calculate conditional dependence.
        **kwargs : dict
            Additional keyword arguments passed to the `midr::mid.conditional()` function in R.
        """
        self._obj = _r_interface._call_r_mid_conditional(
            r_object = estimator.mid_,
            variable = variable,
            **kwargs
        )
        self.variable = variable

    @property
    def observed(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='observed')

    @property
    def conditional(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='conditional')

    @property
    def values(self):
        return _r_interface._extract_and_convert(r_object=self._obj, name='values')
    
    def terms(self, **kwargs):
        return list(_r_interface._call_r_mid_terms(r_object=self._obj, **kwargs))
