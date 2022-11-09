import pandas as pd
from matplotlib import pylab

import model
import config
import utils
import calendar
import sys
import traceback
from pylab import *

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('no argument sent')
        n_days_to_feed_model = 14
        n_days_to_predict = 14
        n_features = 11
        cash_in = True
    else:
        n_days_to_feed_model = sys.argv[1]
        n_days_to_predict = sys.argv[2]
        n_features = sys.argv[3]
        cash_in = sys.argv[4]
    try:
        # Get Raw Data
        df = pd.read_csv('./ds_exercise_data.csv')
        df = df.fillna(0)

        # Data Manipulations & Adding New Features
        df['DateTime'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='DateTime', ascending=True)

        df["Weekday"] = df["DateTime"].apply(lambda x: calendar.day_name[x.weekday()])
        df["DayOfMonth"] = df["DateTime"].apply(lambda x: x.day)
        df["Month"] = df["DateTime"].apply(lambda x: x.month)
        df["MonthName"] = df["DateTime"].apply(lambda x: calendar.month_name[x.month])
        df["Year"] = df["DateTime"].apply(lambda x: x.year)
        df["WeekOfMonth"] = df["DateTime"].apply(lambda x: utils.week_of_month(x))
        df["yearlyDeviation"] = df["Year"].apply(lambda x: config.yearlyDeviationDict[x]["ratio"])
        df["yearlyDeviationChange"] = df["Year"].apply(lambda x: config.yearlyDeviationDict[x]["changeRatio"])
        # Working Day Or Not
        df["workingDayOrNot"] = df["Weekday"].apply(lambda x: True if x in config.weekdaysIndexList else False)
        df["Season"] = df["Month"].apply(lambda x: utils.getSeason(x))

        df["isReligiousHoliday"] = df["DateTime"].apply(lambda x: 1 if x.strftime("%Y-%m-%d") in config.religiousHolidayList else 0)
        df["isNationalHoliday"] = df["DateTime"].apply(lambda x: 1 if x.strftime("%m-%d") in config.nationalHolidayList else 0)
        df["isSpecialDay"] = df["DateTime"].apply(lambda x: 1 if x.strftime("%m-%d") in config.specialDaysList else 0)

        selectedFeatures = df[[
            "CashIn",
            "CashOut",
            "Weekday",
            "DayOfMonth",
            "yearlyDeviation",
            "yearlyDeviationChange",
            "workingDayOrNot",
            "Season",
            "isReligiousHoliday",
            "isNationalHoliday",
            "isSpecialDay"
        ]]

        # "previousWeekCashInAvg",
        # "previousWeekCashInStdDev"

        categoricalFeatureNameList = list(selectedFeatures.select_dtypes(exclude=["number","datetime"]))
        categoricalFeatureIndexList = [selectedFeatures.columns.get_loc(c) for c in categoricalFeatureNameList]

        scaledArray = model.preprocessModelDf(selectedFeatures, categoricalFeatureIndexList)
        model_, history, test_X, test_y, prediction_values = model.runModel(scaledArray, n_features, n_days_to_feed_model, n_days_to_predict, cash_in)

        model.plotLossHistory(history)
        rmse = model.makePrediction( test_X, test_y, prediction_values, model_)
        print("rmse : {}".format(rmse))

    except Exception as e:
        errorLog = str(traceback.format_exc())
        print(errorLog)









