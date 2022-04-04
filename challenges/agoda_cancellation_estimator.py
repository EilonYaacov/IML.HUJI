from __future__ import annotations
from typing import NoReturn
from sklearn.ensemble import RandomForestClassifier
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import accuracy_score


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        # self.models = [RandomForestClassifier(random_state=42), GradientBoostingClassifier(), LogisticRegression()]
        self.estimator = RandomForestClassifier(random_state=42)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        # for c in self.models:
        #     c.fit(X,y)
        self.estimator.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # for c in self.models:
        #     c.predict(X)
        return self.estimator.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        return accuracy_score(y, self.estimator.predict(X))
        # return roc_auc_score(y, self.estimator.predict(X))

        # for c in self.models:
        #     print(accuracy_score(y,c.predict(X)))
