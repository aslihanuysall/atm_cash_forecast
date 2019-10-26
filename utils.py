import config
from math import ceil
import calendar
import matplotlib.pyplot as plt

def getSeason(month):
    """ Returns the season of given month.

        This function is for dataframe lambda operations.

        Parameters:
        month (int): specific month

        Returns:
        str: season
    """
    if month in config.seasonDict["Spring"]:
        return "Spring"

    elif month in config.seasonDict["Summer"]:
        return "Summer"

    elif month in config.seasonDict["Autumn"]:
        return "Autumn"
    else:
        return "Winter"


def week_of_month(dt):
    """ Returns the week of the month for the specified date.

        This function is for dataframe lambda operations.

        Parameters:
        dt (datetime): specific date

        Returns:
        int: week of the month
    """
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom / 7.0))
