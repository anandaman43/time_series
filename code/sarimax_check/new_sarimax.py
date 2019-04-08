import properties as p_model
import statsmodels.api as sm

def sarimax(cus_no, mat_no, pdq, seasonal_pdq, prod, post_outlier_period_flag, run_locally = False, **kwargs):
    """
    Fits sarimax model on the weekly data(cat I, II and III) for the given parameter set,
    performs CV, calculates CV error and makes future prediction.
    :param cus_no: customer number
    :param mat_no: material number
    :param pdq: pdq parameter tuple(e.g. (1,0,1))
    :param seasonal_pdq: seasonal pdq(e.g. (1,1,0,52)
    :param prod: time series data frame for a material for the given customer
        (structure:- ds: date(datetime), y: quantity(float))
    :param kwargs:
        :min_train_days: minimum training period for the CV to start for the remain test data
        :test_points: number of points ahead to make prediction for the each CV step
        :pred_points: future prediction points
        :pdt_cat: Product category object
    :return: ((cus_no, mat_no),
    (_criteria, output_error_dict, _output_pred, list(pdq), list(seasonal_pdq), pdt_category))
    """

    import pandas as pd
    import numpy as np
    import warnings
    import statsmodels.api as sm
    from dateutil import parser

    if 'min_train_days' in kwargs.keys():
        min_train_days = kwargs.get('min_train_days')
    else:
        min_train_days = p_model.min_train_days

    if 'test_points' in kwargs.keys():
        test_points = kwargs.get('test_points')
    else:
        test_points = p_model.test_points

    if 'pred_points' in kwargs.keys():
        pred_points = kwargs.get('pred_points')
    else:
        pred_points = p_model.pred_points

    if 'seasonality_flag' in kwargs.keys():
        seasonality_flag = kwargs.get('seasonality_flag')
    else:
        seasonality_flag = False

    if 'seasonality_component' in kwargs.keys():
        seasonality_component = kwargs.get('seasonality_component')

    if 'image_dir' in kwargs.keys():
        image_dir = kwargs.get('image_dir')

    pdq = pdq
    seasonal_pdq = seasonal_pdq

    ################################################################
    # First split of test and train data
    ################################################################
    # train = prod[prod.ds <= (np.amax(prod.ds) - pd.DateOffset(days=(np.amax(prod.ds) - np.amin(prod.ds)).days
    #                                                                - min_train_days))]
    train = prod.iloc[:prod.shape[0] - 26]
    test = prod[(np.amax(np.array(train.index)) + 1):(np.amax(np.array(train.index)) + 1 + test_points)]
    # rem_data = prod[(np.amax(np.array(train.index)) + test_points):]
    #################################################################

    #################################################################
    # Cross validation step: looping through all the test data points step by step through redefining the
    # train and test set iteratively
    #################################################################
    output_result = pd.DataFrame()  # Data frame to store actual and predicted quantities for cross validation set
    fit_counter = 0
    if seasonality_flag:
        while len(test) > 0:
            # Changing the index to date column to make it model consumable
            train_arima = train.set_index('ds', drop=True)
            test_arima = test.set_index('ds', drop=True)

            seasonality = seasonality_component.sort_values(["ds"]).set_index('ds')
            # print(train_arima)
            # print(seasonality.shape[0])
            train_seasonality = seasonality_expansion(train_arima, seasonality)
            # print(len(train_seasonality))
            test_seasonality = seasonality_expansion(test_arima, seasonality)
            ##########################################################################
            # Model fitting
            ##########################################################################
            warnings.filterwarnings("ignore")  # specify to ignore warning messages

            mod = sm.tsa.statespace.SARIMAX(train_arima, exog=train_seasonality,
                                            order=pdq, seasonal_order=seasonal_pdq,
                                            enforce_stationarity=False, enforce_invertibility=False,
                                            measurement_error=False, time_varying_regression=False,
                                            mle_regression=True)

            results = mod.fit(disp=False)
            ##########################################################################

            ##########################################################################
            # generating forecast for test data points
            ##########################################################################
            pred_test = results.get_prediction(start=pd.to_datetime(np.amax(train_arima.index)),
                                               end=pd.to_datetime(np.amax(test_arima.index)),
                                               exog=test_seasonality, dynamic=True)

            result_test = test
            result_test['y_ARIMA'] = np.array(pred_test.predicted_mean)[1:]
            result_test.loc[(result_test['y_ARIMA'] < 0), 'y_ARIMA'] = 0
            ##########################################################################

            ##########################################################################
            # save plots for CV fit and predictions at each CV step
            ##########################################################################
            if run_locally == True:
                pred_train = results.get_prediction(start= np.amin(np.array(train_arima.index)),dynamic=False)
                result_train = train
                result_train['y_ARIMA'] = np.array(pred_train.predicted_mean)
                three_dim_save_plot(x1= prod.ds, y1= prod.y, y1_label= "Actual",
                                    x2= result_test.ds, y2= result_test.y_ARIMA, y2_label='Predicted',
                                    x3= result_train.ds, y3= result_train.y_ARIMA, y3_label='Model_fit',
                                    text="pdq:" + str(pdq) + " pdq_seasonal:" + str(seasonal_pdq),
                                    xlable= "Date", ylable= "Quantity",
                                    title= "CV_fit_" + str(fit_counter),
                                    dir_name= image_dir, cus_no= cus_no, mat_no= mat_no)
            ##########################################################################

            ##########################################################################
            # recreating test and train data set for next step of CV
            ##########################################################################
            train = prod[:(np.amax(np.array(train.index)) + 1 + test_points)]
            test = prod[(np.amax(np.array(train.index)) + 1):(np.amax(np.array(train.index)) + 1 + test_points)]
            # rem_data = prod[(np.amax(np.array(train.index)) + test_points):]
            ##########################################################################

            # appending the cross validation results at each step
            output_result = pd.concat([output_result, result_test], axis=0)
            fit_counter += 1

        ##############################################################################
        # save plot for complete CV predictions
        ##############################################################################
        if run_locally == True:
            data_baseline = prod.copy()
            data_baseline['rolling_mean'] = pd.rolling_mean(data_baseline['y'].shift(), window=6, min_periods=1)
            baseline_res = data_baseline.loc[-len(output_result):]
            baseline_res = baseline_res.reset_index(drop=True)
            three_dim_save_plot(x1 = prod.ds, y1 = prod.y, y1_label= "Actual",
                              x2 = output_result["ds"], y2= output_result["y_ARIMA"], y2_label= "ARIMA",
                              x3=baseline_res.ds, y3=baseline_res.rolling_mean, y3_label="Baseline", y3_color='purple',
                              xlable="Date", ylable="Quantity",
                              text="pdq:" + str(pdq) + " pdq_seasonal:" + str(seasonal_pdq),
                              title="Baseline_vs_ARIMA_Prediction",
                              dir_name=image_dir, cus_no=cus_no, mat_no=mat_no)
        ##############################################################################

        ##############################################################################
        # Model building on complete data set to generate out of sample prediction
        ##############################################################################
        prod_arima = prod.set_index('ds', drop=True)
        prod_seasonality = seasonality_expansion(prod_arima, seasonality)
        mod = sm.tsa.statespace.SARIMAX(prod_arima, exog=prod_seasonality, order=pdq, seasonal_order=seasonal_pdq,
                                        enforce_stationarity=False, enforce_invertibility=False,
                                        measurement_error=False, time_varying_regression=False,
                                        mle_regression=True)

        results_arima = mod.fit(disp=False)
        pred_seasonality = seasonality_expansion_fwd(prod_arima, prod_seasonality, pred_points)
        pred_arima = results_arima.get_prediction(start=pd.to_datetime(np.amax(prod_arima.index)),
                                                  end=len(prod_arima.y) + pred_points - 1,
                                                  exog=pred_seasonality, dynamic=True)

        # making out of sample predictions
        pred_out_sample = pred_arima.predicted_mean
        pred_out_sample[pred_out_sample < 0] = 0
        _output_pred = _get_pred_dict_sarimax(pred_out_sample)  # # get a dict {(weekNum,year):pred_val}

    else:
        while len(test) > 0:
            # Changing the index to date column to make it model consumable
            train_arima = train.set_index('ds', drop=True)
            test_arima = test.set_index('ds', drop=True)

            ##########################################################################
            # Model fitting
            ##########################################################################
            warnings.filterwarnings("ignore")  # specify to ignore warning messages

            mod = sm.tsa.statespace.SARIMAX(train_arima, order=pdq, seasonal_order=seasonal_pdq,
                                            enforce_stationarity=False, enforce_invertibility=False,
                                            measurement_error=False, time_varying_regression=False,
                                            mle_regression=True)

            results = mod.fit(disp=False)
            ##########################################################################

            ##########################################################################
            # generating forecast for test data points
            ##########################################################################
            pred_test = results.get_prediction(start=pd.to_datetime(np.amax(train_arima.index)),
                                               end=pd.to_datetime(np.amax(test_arima.index)), dynamic=True)

            result_test = test
            result_test['y_ARIMA'] = np.array(pred_test.predicted_mean)[1:]
            result_test.loc[(result_test['y_ARIMA'] < 0), 'y_ARIMA'] = 0
            ##########################################################################

            ##########################################################################
            # save plots for CV fit and predictions at each CV step
            ##########################################################################
            if run_locally == True:
                pred_train = results.get_prediction(start=np.amin(np.array(train_arima.index)), dynamic=False)
                result_train = train
                result_train['y_ARIMA'] = np.array(pred_train.predicted_mean)
                three_dim_save_plot(x1=prod.ds, y1=prod.y, y1_label="Actual",
                                    x2=result_test.ds, y2=result_test.y_ARIMA, y2_label='Predicted',
                                    x3=result_train.ds, y3=result_train.y_ARIMA, y3_label='Model_fit',
                                    text="pdq:" + str(pdq) + " pdq_seasonal:" + str(seasonal_pdq),
                                    xlable="Date", ylable="Quantity",
                                    title="CV_fit_" + str(fit_counter),
                                    dir_name=image_dir, cus_no=cus_no, mat_no=mat_no)
            ##########################################################################

            ##########################################################################
            # recreating test and train data set for next step of CV
            ##########################################################################
            train = prod[:(np.amax(np.array(train.index)) + 1 + test_points)]
            test = prod[(np.amax(np.array(train.index)) + 1):(np.amax(np.array(train.index)) + 1 + test_points)]
            # rem_data = prod[(np.amax(np.array(train.index)) + test_points):]
            ##########################################################################

            # appending the cross validation results at each step
            output_result = pd.concat([output_result, result_test], axis=0)
            fit_counter += 1

            ##############################################################################
            # save plot for complete CV predictions
            ##############################################################################
        if run_locally == True:
            data_baseline = prod.copy()
            data_baseline['rolling_mean'] = data_baseline['y'].shift().rolling(window=6, min_periods=1).mean()
            baseline_res = data_baseline.loc[-len(output_result):]
            baseline_res = baseline_res.reset_index(drop=True)
            three_dim_save_plot(x1=prod.ds, y1=prod.y, y1_label="Actual",
                                x2=output_result["ds"], y2=output_result["y_ARIMA"], y2_label="ARIMA",
                                x3=baseline_res.ds, y3=baseline_res.rolling_mean, y3_label="Baseline",
                                y3_color='purple',
                                xlable="Date", ylable="Quantity",
                                text="pdq:" + str(pdq) + " pdq_seasonal:" + str(seasonal_pdq),
                                title="Baseline_vs_ARIMA_Prediction",
                                dir_name=image_dir, cus_no=cus_no, mat_no=mat_no)
            ##############################################################################

            ##############################################################################
            # Model building on complete data set to generate out of sample prediction
            ##############################################################################
        prod_arima = prod.set_index('ds', drop=True)
        mod = sm.tsa.statespace.SARIMAX(prod_arima, order=pdq, seasonal_order=seasonal_pdq,
                                        enforce_stationarity=False, enforce_invertibility=False,
                                        measurement_error=False, time_varying_regression=False,
                                        mle_regression=True)

        results_arima = mod.fit(disp=False)
        pred_arima = results_arima.get_prediction(start=pd.to_datetime(np.amax(prod_arima.index)),
                                                  end=len(prod_arima.y) + pred_points - 1, dynamic=True)

        # making out of sample predictions
        pred_out_sample = pred_arima.predicted_mean
        pred_out_sample[pred_out_sample < 0] = 0
        _output_pred = _get_pred_dict_sarimax(pred_out_sample)  # # get a dict {(weekNum,year):pred_val}

    ###############################################################################

    ###############################################################################
    # Error calculation
    ###############################################################################
    output_result = weekly_arima_error_calc(output_result)

    output_error = pd.DataFrame(data=[[cus_no, mat_no, rmse_calculator(output_result.y_ARIMA, output_result.y),
                                       mape_calculator(output_result.y_ARIMA, output_result.y),
                                       mae_calculator(output_result.y_ARIMA, output_result.y),
                                       np.nanmedian(
                                           np.absolute(np.array(output_result.rolling_6week_percent_error_arima))),
                                       np.nanmax(
                                           np.absolute(np.array(output_result.rolling_6week_percent_error_arima))),
                                       np.nanmean(
                                           np.absolute(np.array(output_result.rolling_6week_percent_error_arima))),
                                       np.nanmean(
                                           np.absolute(np.array(output_result.rolling_6week_quantity))),
                                       np.nanmedian(
                                           np.absolute(np.array(output_result.rolling_12week_percent_error_arima))),
                                       np.nanmax(
                                           np.absolute(np.array(output_result.rolling_12week_percent_error_arima))),
                                       np.nanmean(
                                           np.absolute(np.array(output_result.rolling_12week_percent_error_arima))),
                                       np.nanmean(
                                           np.absolute(np.array(output_result.rolling_12week_quantity))),
                                       np.nanmedian(
                                           np.absolute(np.array(output_result.rolling_24week_percent_error_arima))),
                                       np.nanmax(
                                           np.absolute(np.array(output_result.rolling_24week_percent_error_arima))),
                                       np.nanmean(
                                           np.absolute(np.array(output_result.rolling_24week_percent_error_arima))),
                                       np.nanmean(
                                           np.absolute(np.array(output_result.rolling_24week_quantity))),
                                       np.nanmedian(
                                           np.absolute(np.array(output_result.rolling_48week_percent_error_arima))),
                                       np.nanmax(
                                           np.absolute(np.array(output_result.rolling_48week_percent_error_arima))),
                                       np.nanmean(
                                           np.absolute(np.array(output_result.rolling_48week_percent_error_arima))),
                                       np.nanmean(
                                           np.absolute(np.array(output_result.rolling_48week_quantity))),
                                       output_result['Error_Cumsum_arima'].iloc[-1],
                                       output_result['cumsum_quantity'].iloc[-1],
                                       ((np.amax(output_result.ds) - np.amin(output_result.ds)).days + 7)]],
                                columns=['cus_no', 'mat_no', 'rmse', 'mape', 'mae',
                                         'wre_med_6', 'wre_max_6', 'wre_mean_6', 'quantity_mean_6',
                                         'wre_med_12', 'wre_max_12', 'wre_mean_12', 'quantity_mean_12',
                                         'wre_med_24', 'wre_max_24', 'wre_mean_24', 'quantity_mean_24',
                                         'wre_med_48', 'wre_max_48', 'wre_mean_48', 'quantity_mean_48',
                                         'cum_error', 'cum_quantity',
                                         'period_days'])
    ##############################################################################

    ##############################################################################
    # Output Preparation
    ##############################################################################
    output_error_dict = pd_func.extract_elems_from_dict(output_error.to_dict(orient='index'))
    _criteria = output_error_dict.get(SARIMAX_W_MODEL_SELECTION_CRITERIA)
    pdt_category = kwargs.get('pdt_cat')
    _result = (
        (cus_no, mat_no), (_criteria, output_error_dict, _output_pred, list(pdq), list(seasonal_pdq), pdt_category,
                           post_outlier_period_flag))
    ##############################################################################

    return _result



def seasonality_expansion(df_series, seasonality):
    """
    expand the seasonality for customer-material data
    :param df_series:
    :param seasonality:
    :return: expanded seasonality
    """
    # df_series_copy = df_series.copy().set_index("dt_week")
    seasonality_copy = seasonality.copy()
    seasonality_copy.columns = ["seasonal_quantity"]
    print(df_series)
    print(seasonality_copy)
    print("sgs", df_series.shape[0])
    print("afef", seasonality_copy.shape[0])
    seasonality_copy = pd.concat([df_series, seasonality_copy], axis=1)
    print(seasonality_copy)
    print("uyguyguy", seasonality_copy.shape[0])
    seasonality_copy = seasonality_copy.reset_index()
    start_iloc, end_iloc = seasonality_copy.dropna(subset=["seasonal_quantity"]).index[0], \
                           seasonality_copy.dropna(subset=["seasonal_quantity"]).index[-1]
    for i in range(start_iloc-1, -1, -1):
        seasonality_copy["seasonal_quantity"].iloc[i] = seasonality_copy.iloc[i + 52]["seasonal_quantity"]
    for i in range(end_iloc + 1, df_series.shape[0]):
        seasonality_copy["seasonal_quantity"].iloc[i] = seasonality_copy.iloc[i - 52]["seasonal_quantity"]

    seasonality_expanded = seasonality_copy[["index", "seasonal_quantity"]]
    seasonality_expanded.columns = ["dt_week", "quantity"]
    seasonality_expanded = seasonality_expanded.set_index("dt_week")
    return sm.add_constant(seasonality_expanded['quantity'].values)


def seasonality_expansion_fwd(df_series, seasonality, pred_points):
    """
    expand the seasonality for customer-material data
    :param df_series:
    :param seasonality:
    :return: expanded seasonality
    """
    # df_series_copy = df_series.copy().set_index("dt_week")
    seasonality_copy = seasonality.copy()
    seasonality_copy.columns = ["seasonal_quantity"]
    seasonality_copy = pd.concat([df_series, seasonality_copy], axis=1)
    seasonality_copy = seasonality_copy.reset_index()
    end_index = df_series.index[-1]
    for i in range(end_index + 1, end_index + 1 + pred_points):
        seasonality_copy["seasonal_quantity"].iloc[i] = seasonality_copy.iloc[i - 52]["seasonal_quantity"]
    seasonality_expanded = seasonality_copy[["dt_week", "seasonal_quantity"]]
    seasonality_expanded.columns = ["dt_week", "quantity"]
    return sm.add_constant(seasonality_expanded['quantity'].values)


if __name__ == "__main__":
    import pandas as pd
    prod = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/code/sarimax_check/133102_production.csv")
    seasonality = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/code/sarimax_check/seasonality.csv")
    print(sarimax(cus_no=121321, mat_no=15234541, pdq=(0,1,1), seasonal_pdq=(0,0,0,52), prod=prod,
                  post_outlier_period_flag=False, run_locally=False, seasonality_component=seasonality,
                  seasonality_flag=True))