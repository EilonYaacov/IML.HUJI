from plotly.subplots import make_subplots
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = 1000
    mu, sigma = 10, 1
    X = np.random.normal(mu, sigma, samples)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    print((univariate_gaussian.mu_, univariate_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    x = []
    y = []
    for i in range(10, 1000, 10):
        x.append(i)
        y.append(abs(np.mean(X[:i]) - mu))

    go.Figure([go.Scatter(x=x, y=y, mode='markers+lines')],
              layout=go.Layout(title=r"$\text{The absolute distance between the estimated- and true value of the "
                                     r"expectation, as a function of the sample size}$",
                               xaxis_title="$\\text{Number of samples}$",
                               yaxis_title="$\\text{The absolute distance between the estimated- and true value of the "
                                           r"expectation}$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = univariate_gaussian.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdfs, mode='markers')],
              layout=go.Layout(title=r"$\text{PDF function of the samples}$",
                               xaxis_title="$\\text{sample values}$",
                               yaxis_title="$\\text{PDF value}$",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples = 1000
    mu, sigma = [0, 0, 4, 0], [[1, 0.2, 0, 0.5],
                               [0.2, 2, 0, 0],
                               [0, 0, 1, 0],
                               [0.5, 0, 0, 1]]
    X = np.random.multivariate_normal(mu, sigma, samples)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(X)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    X_axis = np.linspace(-10, 10, 200)
    Y_axis = np.linspace(-10, 10, 200)
    res = np.ndarray(shape=(len(X_axis), len(Y_axis)))
    min_likelihood = -np.infty
    min_likelihood_index = (0,0)
    for i in range(len(Y_axis)):
        for j in range(len(X_axis)):
            mu_hat = [Y_axis[i], 0, X_axis[j], 0]
            res[i][j] = MultivariateGaussian.log_likelihood(mu_hat, sigma, X)
            min_likelihood_index = (i,j) if min_likelihood < res[i][j] else min_likelihood_index
            min_likelihood = max(min_likelihood,res[i][j])
    fig = make_subplots(rows=1, cols=1).add_traces(go.Heatmap(x=X_axis, y=Y_axis, z=res))
    fig.update_xaxes(title_text="f1 values", row=1, col=1)
    fig.update_yaxes(title_text="f3 values", row=1, col=1)
    fig.update_layout(title_text=r"$\text{HeatMap of log likelihood of f1 and f3 - Question 5}$", height=800)
    fig.show()

    # Question 6 - Maximum likelihood
    print((X_axis[min_likelihood_index[0]],Y_axis[min_likelihood_index[1]]))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
