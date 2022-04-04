import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename).drop_duplicates().dropna()
    full_data = full_data.loc[full_data["Month"] >= 1]
    full_data = full_data.loc[full_data["Day"] >= 1]
    full_data = full_data.loc[full_data["Year"] >= 1995]
    full_data = full_data.loc[full_data["Temp"] >= -10]
    full_data['Date'] = pd.to_datetime(full_data['Date'], format='%Y-%m-%d')
    full_data['DayOfYear'] = full_data['Date'].dt.dayofyear
    full_data.drop(["Date", "Day"], inplace=True,
                   axis=1)
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df["Year"] = df["Year"].astype(str)
    fig = px.scatter(df.loc[(df['Country'] == "Israel")], x="DayOfYear", y="Temp", color="Year",
                     title="Temp in israel as day of year")
    # fig.write_image("Temp_as_day_of_year_israel.png")
    df_month = df.loc[(df['Country'] == "Israel")].groupby("Month").agg(np.std, ddof=0)
    fig_2 = px.bar(df_month, x=np.arange(1, 13, step=1), y="Temp", labels={
        "x": "Month",
        "Temp": "Std of month and temp"},
                   title="Std of month and Temp Israel")
    # fig_2.write_image("std_of_month_israel.png")

    # Question 3 - Exploring differences between countries
    df_3 = df.groupby(["Month", "Country"]).agg({"Temp": [np.std, 'mean']})
    df_3.columns = ['std', 'mean']
    df_3 = df_3.reset_index()

    fig_3 = px.line(df_3, x="Month", y="mean", error_y="std", color="Country", labels={
        "x": "Month",
        "Temp": "mean of month and temp"},
                    title="mean of month and Temp of all countries")
    fig_3.write_image("std_of_month_all_countries.png")

    # Question 4 - Fitting model for different values of `k`
    loss = []
    df_israel = df.loc[(df['Country'] == "Israel")]
    temp = df_israel["Temp"]
    df_israel.drop(["Month", "Country", "Year", "City", "Temp"], inplace=True,
                   axis=1)
    train_X, train_y, test_X, test_y = split_train_test(df_israel, temp, .75)
    for i in range(1, 11):
        poly = PolynomialFitting(k=i)
        poly.fit(train_X.to_numpy(), train_y)
        loss.append(poly.loss(test_X, test_y))
    fig_4 = px.bar(x=np.arange(1, 11, step=1), y=loss, labels={
        "x": "k", "y": "Loss"},
                   title="loss for each k")
    fig_4.write_image("loss_for_each_k.png")

    # Question 5 - Evaluating fitted model on different countries
    # df.drop(["Month", "Country", "Year", "City"], inplace=True,
    #                axis=1)
    loss_between_countries = []
    poly = PolynomialFitting(k=5)
    poly.fit(train_X.to_numpy(), train_y)
    loss_between_countries.append(poly.loss(test_X, test_y))
    Countries = ["Israel", "Jordan", "South Africa", "The Netherlands"]
    for country in Countries:
        if country == "Israel":
            continue
        df_country = df.loc[(df['Country'] == country)]
        temp_country = df_country["Temp"]
        df_country = df_country["DayOfYear"]
        loss_between_countries.append(poly.loss(df_country.to_frame(), temp_country))

    fig_5 = px.bar(x=Countries, y=loss_between_countries, labels={
        "x": "Country", "y": "Loss"},
                   title="Loss between fitted k=5 Israel and other countries")
    fig_5.write_image("loss_between_israel_fitted_5_and_countries.png")
