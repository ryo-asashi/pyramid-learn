# src/midlearn/api.py

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score

from . import _r_interface

class MIDRegressor(BaseEstimator, RegressorMixin):
    """
    Class for stand-alone Maximum Interpretation Decomposition models.
    """
    def __init__(
        self,
        params_main=None,
        params_inter=None,
        penalty=0,
        **kwargs
    ):
        """
        Creates a MID model.

        Args:
            params_main, params_inter, penalty: Hyperparameters for the MID fitting process.
            kwargs: Advanced fitting options passed to midr's interpret().
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
        """
        Fit the MID model to the response y on predictors X.

        Args:
            X: Data used to train the MID model.
            y: Target values.
            sample_weights: Sample weights.
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
        """
        A low-level method to call the R predict.mid function with arbitrary arguments. The kwargs are passed directly to the R function.

        Args:
            X (pd.DataFrame or np.ndarray): New data for which to make predictions.
            **kwargs: Arguments passed directly to R's predict.mid (e.g., type, terms).

        Returns:
            np.ndarray: The prediction result from R.
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
        """
        Predicts target values for new data X using the fitted MID model.
        """
        return self.r_predict(X, type='response')

    def predict_terms(
        self,
        X
    ) -> np.ndarray:
        """
        Predicts the contribution of each term for new data X.
        """
        return self.r_predict(X, type='terms')

    def effect(
        self,
        term: str,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | None = None    
    ) -> np.ndarray:
        """
        Evaluate single MID component functions for new data.

        Args:
            x (pd.DataFrame or np.ndarray): New data for the first variable in the term. If a pd.DataFrame is provided, values of the related variables are extracted from it.
            y: An optional np.ndarray of values for the second variable in an interaction term.

        Returns:
            np.ndarray: A NumPy array of the calculated term contributions, with the same length as x (and y).
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
        return _r_interface._extract_and_convert(r_object=self.mid_, name='intercept').item()

    @property
    def weights(self):
        return _r_interface._extract_and_convert(r_object=self.mid_, name='weights')

    @property
    def fitted_matrix(self):
        return _r_interface._extract_and_convert(r_object=self.mid_, name='fitted.matrix')

    @property
    def fitted_values(self):
        return _r_interface._extract_and_convert(r_object=self.mid_, name='fitted.values')
    
    @property
    def residuals(self):
        return _r_interface._extract_and_convert(r_object=self.mid_, name='residuals')

    @property
    def ratio(self):
        return _r_interface._extract_and_convert(r_object=self.mid_, name='ratio').item()

    def terms(self, **kwargs):
        return list(_r_interface._call_r_mid_terms(r_object=self.mid_, **kwargs))

    def main_effects(self, term: str):
        effects = _r_interface._extract_and_convert(r_object=self.mid_, name='main.effects')
        return _r_interface._extract_and_convert(r_object=effects, name=term)

    def interactions(self, term: str):
        effects = _r_interface._extract_and_convert(r_object=self.mid_, name='interactions')
        return _r_interface._extract_and_convert(r_object=effects, name=term)
    
    def _encoding_type(self, tag: str, order: int = 1):
        obj = _r_interface._extract_and_convert(r_object=self.mid_, name='encoders')
        obj = _r_interface._extract_and_convert(r_object=obj, name='main.effects' if order == 1 else 'interactions')
        obj = _r_interface._extract_and_convert(r_object=obj, name=tag)
        return _r_interface._extract_and_convert(r_object=obj, name='type')[0]

    def importance(self, **kwargs):
        return MIDImportance(estimator=self, **kwargs)

    def breakdown(self, **kwargs):
        return MIDBreakdown(estimator=self, **kwargs)

    def conditional(self, variable: str, **kwargs):
        return MIDConditional(estimator=self, variable=variable, **kwargs)


class MIDExplainer(MIDRegressor, MetaEstimatorMixin):
    """
    Class for surrogate Maximium Interpretation Decomposition models.
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
        """
        Creates a MID model.
        
        Args:
            estimator: The pre-trained black-box model to be explained.
            params_main, params_inter, penalty: Hyperparameters for the MID fitting process.
            kwargs: Advanced fitting options passed to midr's interpret().
        """
        self.estimator = estimator
        if is_classifier(self.estimator):
            self.target_classes = target_classes
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
        """
        Generates a unified continuous prediction from the estimator, handling both classifiers and regressors.
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
        """
        Fit the surrogate MID model to the predictions of the estimator on X.

        Args:
            X: Data used to train the surrogate MID model.
            y: Predictions obtained from the estimator (optional).
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
        """
        Calculate the fidelity of the surrogate model.

        This score (R-squared) measures how well this explainer's predictions match the original estimator's predictions on the data X.
        A score close to 1.0 means the explainer is a faithful surrogate.

        Args:
            X: The data to evaluate fidelity on.
            y: Predictions obtained from the estimator (optional).

        Returns:
            float: The R-squared score between the original estimator's predictions and this surrogate model's predictions.
        """
        check_is_fitted(self)
        if y is None:
            print("Generating predictions from the estimator...")
            y = self._predict_y_estimator(X)
        y_explainer = self.predict(X)
        return r2_score(y_true=y, y_pred=y_explainer, sample_weight=sample_weight)



class MIDImportance(object):
    """
    Class for MID Importance
    """
    def __init__(
        self,
        estimator: MIDRegressor | MIDExplainer,
        **kwargs
    ):
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
    """
    Class for MID Breakdown
    """
    def __init__(
        self,
        estimator: MIDRegressor | MIDExplainer,
        row: int | None = None,
        **kwargs
    ):
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
    Class for MID Conditional
    """
    def __init__(
        self,
        estimator: MIDRegressor | MIDExplainer,
        variable: str,
        **kwargs
    ):
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
