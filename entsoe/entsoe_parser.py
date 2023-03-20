"""
Разработчик: Д.Девяткин, А.Афанасьев, А.Гладыщев
Описание:
Файл содержит класс для получения данных источника entsoe
с сайта.
Входные данные: данные из источника https://transparency.entsoe.eu/generation/r2/dayAheadAggregatedGeneration/
Выходные данные: pd.DataFrame с преобразованными данными из источника
"""
# """
# Data parcer for Entsoe transparency platform
# https://transparency.entsoe.eu/generation/r2/dayAheadAggregatedGeneration/
# TODO:perfect-
# Business goal:
# Use https://github.com/EnergieID/entsoe-py/tree/master/entsoe api to get data.
# Requirement:
# add to Entsoe library in mapping.py:
# GE =            '10Y1001A1001B012', 'CTA | GE, BZN | GE, Georgia(GE)',              'Europe/Brussels',
# """
import os
from datetime import timedelta, datetime
from entsoe import EntsoePandasClient, exceptions
import pandas as pd
import logging
import numpy as np
from dotenv import load_dotenv

load_dotenv('/srv/sstd/.env')

logging.basicConfig(filename="log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)


class Entsoe:
    def __init__(self):
        """
        choice
        params for scraping
        and date

        """
        end_date = datetime.today().strftime("%Y%m%d")
        start_date_frc = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
        start_date = (datetime.today() - timedelta(days=2)).strftime("%Y%m%d")
        self.country_list = {'AL': 'Albania', 'AT': 'Austria', 'BY': 'Belarus',
                             'BE': 'Belgium', 'BA': 'Bosnia and Herz.', 'BG': 'Bulgaria', 'HR': 'Croatia',
                             'CY': 'Cyprus',
                             'CZ': 'Czech Republic', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
                             'DE': 'Germany', 'GR': 'Greece', 'HU': 'Hungary', 'IS': 'Iceland',
                             'IE': 'Ireland', 'IT': 'Italy', 'XK': 'Kosovo', 'LV': 'Latvia', 'LT': 'Lithuania',
                             'LU': 'Luxembourg', 'MT': 'Malta', 'MD': 'Moldova', 'ME': 'Montenegro',
                             'NL': 'Netherlands',
                             'MK': 'North Macedonia', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
                             'RU': 'Russia', 'RS': 'Serbia', 'SK': 'Slovakia', 'SI': 'Slovenia', 'ES': 'Spain',
                             'SE': 'Sweden', 'CH': 'Switzerland', 'TR': 'Turkey', 'UA': 'Ukraine',
                             'UK': 'United Kingdom'}
        self.client = EntsoePandasClient(api_key=os.environ.get("ENTSOE_API_KEY"))
        self.start = pd.Timestamp(f'{start_date}', tz='Europe/Brussels')
        self.start_frc = pd.Timestamp(f'{start_date_frc}', tz='Europe/Brussels')
        self.end = pd.Timestamp(f'{end_date}', tz='Europe/Brussels')

    def entsoe_get_values(self):
        """
        get data and convert to file
        :return: csv files
        """
        forecast_dfs = []
        actual_dfs = []
        for i, k in self.country_list.items():
            print(k)
            try:
                api_forecast = self.client.query_generation_forecast(f'{i}', start=self.start_frc, end=self.end)
                table_forecast = pd.DataFrame(api_forecast)
                frc_cols = table_forecast.columns.values
                table_forecast['date'] = table_forecast.index
                table_forecast = pd.melt(table_forecast, id_vars=['date'], value_vars=frc_cols)
                table_forecast['delivery_point'] = k
                forecast_dfs.append(table_forecast)
            except exceptions.NoMatchingDataError:
                print(i, 'missing', k)
            try:
                api_actual_generation = self.client.query_generation(f'{i}', start=self.start, end=self.end,
                                                                     psr_type=None)
                table_actual_generation = pd.DataFrame(api_actual_generation)
                if len(table_actual_generation.columns.values[0]) == 2:
                    print(2)
                    column_all = np.unique(list(map(lambda x: table_actual_generation.columns.to_numpy()[x][0],
                                                    range(len(table_actual_generation.columns.to_numpy())))))
                else:
                    column_all = table_actual_generation.columns.values
                for j in range(len(column_all)):
                    temp = table_actual_generation[column_all[j]]
                    if type(temp) == pd.core.frame.DataFrame:
                        temp_cols = temp.columns.values
                    else:
                        temp = pd.DataFrame(temp.to_numpy(), columns=[column_all[j]], index=temp.index)
                        temp_cols = temp.columns.values
                    temp['date'] = temp.index
                    if len(temp_cols) >= 2:

                        temp = pd.melt(temp, id_vars=['date'], value_vars=temp_cols)
                    else:
                        temp = pd.melt(temp, id_vars=['date'], value_vars=temp_cols)
                        temp['variable'] = 'generation'
                    temp['gen_unit'] = column_all[j]
                    temp = temp.rename(columns={'variable': 'gen_type', 'country_2': 'delivery_point'})
                    temp = temp.replace('Actual Aggregated', 'generation').replace('Actual Consumption', 'consumption')
                    temp['delivery_point'] = k
                    actual_dfs.append(temp)
            except exceptions.NoMatchingDataError:
                print(i, 'missing', k)
        d1 = pd.concat(forecast_dfs)
        d1['variable'] = d1['variable'].replace(['Actual Aggregated', 'Actual Consumption'],
                                                ['generation', 'consumption'])
        d1 = d1.rename(columns={'variable': 'gen_type', 'country_2': 'delivery_point'})

        d1['gen_unit'] = 'forecast'
        d2 = pd.concat(actual_dfs)
        d2 = d2.rename(columns={'variable': 'gen_type', 'country_2': 'delivery_point'})
        # d1.to_csv(r'C:\Users\Dmitry\Downloads\frc.csv',index=False)
        # d2.to_csv(r'C:\Users\Dmitry\Downloads\actual.csv',index=False)
        return {'frc': d1, 'fact': d2}


if __name__ == '__main__':
    result = Entsoe()
    result.entsoe_get_values()
