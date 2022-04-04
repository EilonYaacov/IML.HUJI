
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).drop_duplicates().dropna()
    full_data = full_data.loc[full_data["price"] >= 0]
    full_data = full_data.loc[full_data["yr_built"] >= 1800]
    full_data = full_data.loc[full_data["sqft_living"] >= 0]
    full_data = full_data.loc[full_data["sqft_lot"] >= 0]
    full_data = full_data.loc[full_data["sqft_living15"] >= 0]
    full_data = full_data.loc[full_data["sqft_lot15"] >= 0]
    full_data = full_data.loc[full_data["sqft_above"] >= 0]
    full_data = full_data.loc[full_data["sqft_basement"] >= 0]
    response = full_data["price"]
    full_data["is_renovated"] = full_data["yr_renovated"].apply(lambda x: 1 if x > 2000 else 0)
    full_data = pd.get_dummies(full_data, prefix='',
                               columns=[
                                   "zipcode"])
    full_data.drop(["id", "date", "long", "lat", "price", "yr_renovated"], inplace=True,
                   axis=1)
    return full_data, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_of_y = np.std(y)
    for feature in X:
        pearson_corr = np.cov(X[feature], y)[0][1] / (np.std(X[feature]) * std_of_y)
        fig = px.scatter(x=X[feature], y=y, width=1000, opacity=0.65,
                         trendline='ols', trendline_color_override='darkblue')
        fig.update_layout(
            title="Pearson Correlation of " + feature + " and the response: " + str(round(pearson_corr, 3)),
            xaxis={"title": feature},
            yaxis={"title": "Response"})
        fig.write_image(output_path + "" + feature + "_correlation.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, price = load_data("datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, price, "exercises/ex_2_plots")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, price, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_regression = LinearRegression(True)
    average_loss = []
    variance_of_loss = []
    upper_bound = []
    for i in range(10, 101):
        loss = 0
        for j in range(10):
            p_train_X, p_train_y, p_test_X, p_test_y = split_train_test(train_X, train_y, i / 100)
            linear_regression.fit(p_train_X, p_train_y)
            loss += linear_regression.loss(test_X, test_y)
        average_loss.append(loss/10)
        # upper_bound = average_loss + 2*variance_of_loss


    fig = px.scatter(x=np.arange(10, 101, step=1), y=average_loss)
    fig.update_layout(title="Average loss as function of training size", xaxis={"title": 'Percentage of training size'},
                      yaxis={"title": "Average loss"})
    fig.write_image("Average loss as function of training size.png")
