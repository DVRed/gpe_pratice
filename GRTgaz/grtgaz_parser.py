import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
import unicodedata
from typing import Literal, Union


Data_Type = Literal['consumptions', 'commercial_flow', 'physical_flow']
columns = ['date', 'delivery_point', 'from_country', 'to_country',
           'curve_type', 'flow_type', 'value', 'curve_name']


def strip_accents(string_: str):
    """
    function to remove accents from strings
    """
    return ''.join(char_ for char_ in unicodedata.normalize('NFD', string_)
                   if unicodedata.category(char_) != 'Mn')


def get_data_from_resource(file_name: str, url: str, params: dict):
    """
    function to get data as a file from resource set with url with set parameters
    """
    result = requests.get(url, params=params)
    with open(file_name, 'wb') as f:
        f.write(result.content)


class DataRow:
    """
    class to set the data row
    """
    def __init__(self, data_type: Data_Type,
                 date_: datetime,
                 value: Union[int, float],
                 curve_type: str = None,
                 delivery_point: str = None,
                 curve_name: str = None):
        self._date = date_
        self._value = value
        self._flow_type = 'physical_flow'
        self._from_country = self._to_country = 'FR'
        if data_type == 'consumptions':
            self._delivery_point = f'{self._get_dp(curve_type)} {self._to_country}'
            self._curve_type = curve_type
            self._curve_name = f"{self._delivery_point}_forecast_exit_" \
                               f"{self._date.strftime('%d.%m.%Y')}"
        else:
            self._delivery_point = delivery_point
            self._get_countries()
            self._curve_type = data_type
            if data_type == 'commercial_flow':
                self._curve_name = curve_name
                if self._curve_name == 'exit':
                    self._from_country, self._to_country = self._to_country, self._from_country
            else:
                self._curve_name = f"({self._to_country}) {self._delivery_point}" \
                                   f"_forecast_entry_{self._date.strftime('%d.%m.%Y')}"

    def _get_countries(self):
        """
        function to get 'from' and 'to' countries depending on delivery point
        """
        delivery_points_countries = {'Alveringem': {'from': 'BE', 'to': 'FR'},
                                     'Dunkerque': {'from': 'NO', 'to': 'FR'},
                                     'Jura': {'from': 'CH', 'to': 'FR'},
                                     'Midi': {'from': 'FR', 'to': 'FR'},  # to change
                                     'Obergailbach': {'from': 'DE', 'to': 'FR'},  # in db entry/ net/ reverse
                                     'Oltingue': {'from': 'CH', 'to': 'FR'},  # Oltingue (FR) / Rodersdorf (CH)
                                     'TIGF interconnection': {'from': 'FR', 'to': 'FR'},  # to change
                                     'Taisnieres B': {'from': 'BE', 'to': 'FR'},  # no in db
                                     'Taisnieres H': {'from': 'BE', 'to': 'FR'},  # no in db
                                     'VIP Virtualys': {'from': 'BE', 'to': 'FR'},   # no in db
                                     }
        countries = delivery_points_countries[self._delivery_point]
        self._from_country = countries['from']
        self._to_country = countries['to']

    @staticmethod
    def _get_dp(curve_type: str) -> str:
        dps = {'industrial_demand': 'Industrial demand',
               'LDZ_demand': 'LDZ demand',
               'power_demand': 'Power production demand',
               'other_demand': 'Other demand'}
        return dps[curve_type]

    def set_df_row(self):
        """
        function to set the data row as dictionary
        """
        df_row = {
            'date': self._date,
            'delivery_point': self._delivery_point,
            'from_country': self._from_country,
            'to_country': self._to_country,
            'curve_type': self._curve_type,
            'flow_type': self._flow_type,
            'value': self._value,
            'curve_name': self._curve_name
        }
        return df_row


class XLSData:
    """
    class to get all the data from .xlsx in the DataFrame
    """
    def __init__(self, file_name: str,
                 data_type: Data_Type):
        self.data_type = data_type
        self.df = pd.DataFrame(columns=columns)
        if self.data_type == 'consumptions':
            self.read_consumptions(file_name)
        else:
            self.read_flows(file_name)
        Path(file_name).unlink()

    def read_consumptions(self, file_name):
        """
        function to read data from .xlsx  one sheet 3-6 columns and put it in DataFrame
        (consumptions)
        """
        new_column_names = ['date', 'industrial_demand', 'LDZ_demand', 'power_demand', 'other_demand']
        xls_df = pd.read_excel(file_name, header=2, usecols='A,C:F', names=new_column_names)
        curve_types = set(xls_df.columns) - {'date'}
        for row in range(len(xls_df)):
            date_ = xls_df.loc[row, 'date']
            for curve_type in curve_types:
                value = xls_df.loc[row, curve_type]
                if not np.isnan(value):
                    df_row = DataRow(data_type=self.data_type, date_=date_, value=value, curve_type=curve_type)
                    self.df.loc[len(self.df.index)] = df_row.set_df_row()

    def read_flows(self, file_name):
        """
        function to go through all the sheets in .xlsx
        (for flows)
        """
        xls = pd.ExcelFile(file_name)
        for sheet_num in range(len(xls.sheet_names)):
            delivery_point = xls.sheet_names[sheet_num]
            if self.data_type == 'commercial_flow':  # for commercial flows
                sheet = xls.parse(delivery_point, header=2)
                self.drop_unnecessary_columns(sheet)
            else:  # for physical flows
                sheet = xls.parse(delivery_point, header=3, usecols='A:B')
            delivery_point = strip_accents(delivery_point)
            self.go_through_df_rows(sheet, delivery_point)

    @staticmethod
    def drop_unnecessary_columns(sheet):
        """
        function to drop all columns where is no necessary data
        (for commercial flows)
        """
        first_value_col = -1
        for col in range(1, len(sheet.columns), 2):  # drop all columns that do not contain "Shippers' nomination"
            if sheet.columns[col].find("Shippers' nomination") != -1:
                first_value_col = col
                break
        if first_value_col != -1:
            sheet.drop(sheet.iloc[:, 1:first_value_col], inplace=True, axis=1)
        else:
            raise IndexError("there are no Shippers' nomination columns")

    def go_through_df_rows(self, sheet, delivery_point):
        """
        function to go through all DataFrameRows
        (for flows)
        """
        for row in range(1, len(sheet)):
            date_ = sheet.iloc[row, 0]
            if self.data_type == 'commercial_flow':  # for commercial flows
                self.get_last_not_empty_col(sheet, row, date_, delivery_point)
            else:   # for physical flows
                value = sheet.iloc[row, 1]
                if not np.isnan(value):
                    df_row = DataRow(data_type=self.data_type,
                                     date_=date_,
                                     value=value,
                                     delivery_point=delivery_point)
                    df_row.delivery_point = delivery_point
                    self.df.loc[len(self.df.index)] = df_row.set_df_row()

    def get_last_not_empty_col(self, sheet, row, date_, delivery_point):
        """
        function to find last not empty column
        (for commercial flows)
        """
        def get_df_row(value, curve_name):
            """
            inner function to set row and put it in DataFrame
            """
            # df_row = BiggerDataRow(dt, value, delivery_point, bool_entry)
            df_row = DataRow(data_type=self.data_type,
                             date_=date_,
                             value=value,
                             curve_name=curve_name,
                             delivery_point=delivery_point)
            self.df.loc[len(self.df.index)] = df_row.set_df_row()

        value_entry = value_exit = 0
        for col in range(len(sheet.columns) - 2, 0, -2):
            not_empty_bool = False
            value = sheet.iloc[row, col]  # for curve_name = 'entry'
            if not np.isnan(value):
                value_entry = value
                not_empty_bool = True

            value = sheet.iloc[row, col + 1]   # for curve_name = 'exit'
            if not np.isnan(value):
                value_exit = value
                not_empty_bool = True

            if not_empty_bool:
                break

        get_df_row(value=value_entry, curve_name='entry')
        get_df_row(value=value_exit, curve_name='exit')


class GRTgazParser:
    """
    class to get data from GRTgaz in the DataFrame form
    """
    def __init__(self, data_type: Data_Type):
        type_param = param = ''
        self.data_type = data_type
        match self.data_type:
            case 'consumptions':
                type_param = 'consommation'
                param = 'Zone'
            case 'commercial_flow':
                type_param = 'flux_commerciaux'
                param = 'PIR'
            case 'physical_flow':
                type_param = 'flux_physiques'
                param = 'PIR'
        self.url = f'https://www.smart.grtgaz.com/api/v1/en/{type_param}/export/{param}.xls'
        self.dt_today = date.today()
        self.dt_format = '%-Y-%-m-%-d'

    def get_current_data(self):
        """
        function to get data for the past two days in the DataFrame form
        """
        file_name = 'current.xlsx'
        params = {
            'startDate': (self.dt_today - timedelta(days=2)).strftime(self.dt_format),
            'endDate': self.dt_today.strftime(self.dt_format)
        }
        if self.data_type == 'commercial_flow':
            params['range'] = 'daily'
        get_data_from_resource(file_name, self.url, params)
        return XLSData(file_name, self.data_type).df.reset_index(drop=True, inplace=True)

    def get_historical_data(self, start_date=date(2015, 4, 1), end_date=date.today()):
        """
        function to get historical data from start date till end date in the DataFrame form
        (default from 01.04.2015 till now)
        """
        df = pd.DataFrame(columns=['date', 'delivery_point', 'from_country',
                                   'to_country', 'curve_type', 'flow_type', 'value'])
        for year in range(start_date.year, end_date.year + 1):
            params = {
                'startDate': date(year, 1, 1).strftime(self.dt_format),
                'endDate': date(year, 12, 31).strftime(self.dt_format)
            }
            if self.data_type == 'commercial_flow':
                params['range'] = 'daily'

            match year:
                case start_date.year:
                    params['startDate'] = start_date.strftime(self.dt_format)

                case end_date.year:
                    params['endDate'] = end_date.strftime(self.dt_format)

            file_name = 'temp.xlsx'
            get_data_from_resource(file_name, self.url, params)
            temp_df = XLSData(file_name, self.data_type).df
            df = pd.concat([df, temp_df], sort=False, axis=0)
            df.reset_index(drop=True, inplace=True)
        return df


if __name__ == '__main__':
    data_types = ('consumptions', 'commercial_flow', 'physical_flow')
    data_type = data_types[2]
    cparser = GRTgazParser(data_type)
    historical_data = cparser.get_historical_data()
    # # current_data = cparser.get_current_data()
    folder = 'parsed_data/'
    historical_data.to_excel(f'{folder}all_GRTgaz_{data_type}.xlsx', index=False)
    # # current_date = date.today().strftime('%d.%m.%Y')
    # # current_data.to_excel(f'{folder}GRTgaz_{data_type}_{current_date}.xlsx')








