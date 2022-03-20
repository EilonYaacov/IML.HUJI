from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet
import plotly.graph_objects as go
import plotly.io as pio


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_, self.var_ = np.mean(X), np.var(X)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdfs = np.ndarray(shape=(len(X),))
        for i in range(len(X)):
            pdfs[i] = (1 / math.sqrt(2 * math.pi * self.var_)) * math.exp(
                math.pow(X[i] - self.mu_, 2) / self.var_ * (-1 / 2))
        return pdfs

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        scalar = (-len(X) / 2) * np.log(2 * np.pi * sigma)
        X_centered = X - np.tile(mu, len(X))
        exp = (-0.5 * sigma) * np.sum(np.power(X_centered, 2))
        log_likelihood = scalar + exp
        return log_likelihood


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.zeros(len(X[0]))
        for i in range(len(self.mu_)):
            self.mu_[i] = np.mean(X.transpose()[i])
        X_mu_s = np.tile(self.mu_, [len(X), 1])
        X_centered = np.subtract(X_mu_s, X)
        self.cov_ = 1 / (len(X) - 1) * np.matmul(np.transpose(X_centered), X_centered)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdfs = np.ndarray(shape=(len(X),))
        scalar = 1 / np.sqrt(np.power(2 * np.pi, len(self.cov_)) * np.linalg.slogdet(self.cov_)[1])
        X_centered = X - np.tile(self.mu_, [len(X), 1])
        inverse_cov = np.linalg.inv(self.cov_)
        for i in range(len(X)):
            pdfs[i] = scalar * np.exp(-0.5 * (np.sum(X_centered[i] @ inverse_cov * X_centered[i])))
        return pdfs

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        res = len(X) * np.log(1 / np.sqrt(np.power(2 * np.pi, len(cov)) * np.linalg.slogdet(cov)[1]))
        X_centered = X - np.tile(mu, [1000, 1])
        inverse_cov = np.linalg.inv(cov)
        log_likelihood = res - 0.5 * (np.sum(X_centered @ inverse_cov * X_centered))
        return log_likelihood
