from sklearn.preprocessing import StandardScaler
from challenges.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd


def load_data(is_train_data: bool, filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """

    full_data = pd.read_csv(filename).drop_duplicates()
    full_data["duration"] = (pd.to_datetime(full_data["checkout_date"], format="%Y-%m-%d %H:%M:%S") - pd.to_datetime(
        full_data["checkin_date"], format="%Y-%m-%d %H:%M:%S")) / np.timedelta64(1, 'D')
    full_data["time_of_reservation_before_checkin"] = (pd.to_datetime(full_data["checkin_date"],
                                                                      format="%Y-%m-%d %H:%M:%S") - pd.to_datetime(
        full_data["booking_datetime"], format="%Y-%m-%d %H:%M:%S")) / np.timedelta64(1, 'D')

    if is_train_data:
        full_data["cancellation_datetime"].apply(lambda x: "0000-00-00 00:00:00" if x is np.nan else x)
        full_data["time_of_reservation_before_cancellation"] = (pd.to_datetime(full_data["cancellation_datetime"],
                                                                               format="%Y-%m-%d %H:%M:%S") - pd.to_datetime(
            full_data["booking_datetime"], format="%Y-%m-%d %H:%M:%S")) / np.timedelta64(1, 'D')
        full_data["did_cancel_within_a_month"] = full_data["time_of_reservation_before_cancellation"].apply(
            lambda x: 1 if (x > -1 and x < 45) else 0)
    full_data.drop(
        ["h_booking_id", "h_customer_id", "booking_datetime", "checkin_date", "checkout_date", "hotel_id",
         "hotel_country_code",
         "hotel_live_date", "origin_country_code",
         "original_payment_method", "original_payment_currency", "request_nonesmoke", "request_highfloor",
         "request_latecheckin", "request_largebed", "request_highfloor", "request_earlycheckin",
         "hotel_area_code", "hotel_chain_code", "request_twinbeds", "hotel_city_code", "customer_nationality",
         "language", "hotel_brand_code", "no_of_extra_bed", "request_airport"],
        inplace=True, axis=1)
    full_data = pd.get_dummies(full_data, prefix='',
                               columns=["guest_nationality_country_name","accommadation_type_name",
                                        "original_payment_type"])
    full_data["charge_option"] = full_data["charge_option"].apply(lambda x: 1 if x == "Pay Later" else -1)

    full_data["days_without_payment"] = full_data.apply(
        lambda x: func(x["cancellation_policy_code"], int(x["time_of_reservation_before_checkin"]), x["duration"])[0],
        axis=1)
    full_data["percentage"] = full_data.apply(
        lambda x: func(x["cancellation_policy_code"], int(x["time_of_reservation_before_checkin"]), x["duration"])[1],
        axis=1)
    full_data.replace({True: 1, False: 0, np.nan: 0}, inplace=True)
    if is_train_data:
        labels = full_data["did_cancel_within_a_month"]
        full_data.drop(["cancellation_policy_code", "did_cancel_within_a_month", "cancellation_datetime",
                        "time_of_reservation_before_cancellation"], inplace=True,
                       axis=1)
    else:
        full_data.drop(["cancellation_policy_code"], inplace=True,
                       axis=1)
        labels = None
    features = full_data
    return features, labels


def func(cancellation_policy_code, days_before_reservation, duration):
    cancellation_policy_array = cancellation_policy_code.split("_")
    days_without_payment = 0
    days, percentage = [0] * len(cancellation_policy_array), [0] * len(cancellation_policy_array)
    for i in range(len(cancellation_policy_array)):
        if "D" not in cancellation_policy_array[i]:
            break
        days[i], percentage[i] = parser(cancellation_policy_array[i], duration)
        if i == 0:
            days_without_payment = max(-1, days_before_reservation - days[i])
        else:
            days[i - 1] = days[i - 1] - days[i]
    final_percentage = 0
    if len(cancellation_policy_array) == 2 and days[0] != 0:
        final_percentage = days[0] / (days[0] + days[1]) * percentage[0] + days[1] / (days[0] + days[1]) * percentage[1]
    else:
        final_percentage = percentage[0]
    final_percentage = -1 if final_percentage == 0 else final_percentage
    if days_without_payment > 45:
        final_percentage = 0
        days_without_payment = -1
    return days_without_payment, final_percentage


def parser(policy_code, duration):
    cancellation_policy_array = policy_code.split("D")
    days = cancellation_policy_array[0]
    percentage = 0
    if "N" in cancellation_policy_array[1]:
        percentage = (int(cancellation_policy_array[1][:-1]) / int(duration)) * 100
    else:
        percentage = float(cancellation_policy_array[1][:-1])

    return int(days), percentage


def evaluate_and_export(estimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame({"predicted_values": estimator.predict(X)}).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data(True, "datasets/agoda_cancellation_train.csv")
    load_real_test_x = load_data(False, "datasets/test_set_week_1.csv")[0]
    train_df, real_test_x = df.align(load_real_test_x, join='outer', axis=1, fill_value=0)
    fitted_scaler = StandardScaler().fit(train_df)
    x_df_update = pd.DataFrame(fitted_scaler.transform(train_df),
                               columns=train_df.columns)
    train_X, train_y, test_X, test_y = split_train_test(x_df_update, cancellation_labels, .75)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # real_test_x = real_test_x[df.columns]
    fitted_scaler = StandardScaler().fit(real_test_x)
    real_test_x_scaled = pd.DataFrame(fitted_scaler.transform(real_test_x),
                               columns=real_test_x.columns)
    # Store model predictions over test set
    evaluate_and_export(estimator, real_test_x_scaled, "208562405_208251975_316010636.csv")
