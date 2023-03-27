import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta


def get_data_from_resource(file_name, url, params):
    """
    function to get data as a file from resource set with url with set parameters
    """
    r = requests.get(url, params=params)
    with open(file_name, 'wb') as f:
        f.write(r.content)


class DataRow:
    """
    class to set the data row
    """
    def __init__(self, dt, curve_type, value):
        self.date = dt
        self.curve_type = curve_type
        self.value = value
        self.delivery_point = self.from_country = self.to_country = 'FR'
        self.flow_type = 'physical_flow'

    def set_df_row(self):
        """
        function to set the data row as dictionary
        """
        df_row = {
            'date': self.date,
            'delivery_point': self.delivery_point,
            'from_country': self.from_country,
            'to_country': self.to_country,
            'curve_type': self.curve_type,
            'flow_type': self.flow_type,
            'value': self.value,
        }
        return df_row

    def get_countries(self):
        """
        function to get 'from' and 'to' countries depending on delivery point
        """
        delivery_points_countries = {'Alveringem': {'from': 'BE', 'to': 'FR'},
                                     'Dunkerque': {'from': 'NO', 'to': 'FR'},
                                     'Jura': {'from': 'CH', 'to': 'FR'},
                                     'Midi': {'from': 'FR', 'to': 'FR'},  # to change
                                     'Obergailbach': {'from': 'FR', 'to': 'DE'},
                                     'Oltingue': {'from': 'FR', 'to': 'CH'},
                                     'TIGF interconnection': {'from': 'FR', 'to': 'FR'},  # to change
                                     'Taisnières B': {'from': 'BE', 'to': 'FR'},
                                     'Taisnières H': {'from': 'BE', 'to': 'FR'},
                                     'VIP Virtualys': {'from': 'BE', 'to': 'FR'},
                                     }
        countries = delivery_points_countries[self.delivery_point]
        self.from_country = countries['from']
        self.to_country = countries['to']


class BiggerDataRow(DataRow):
    """
    class expansion of class DataRow with additional column 'curve_name'
    (for commercial flows)
    """
    def __init__(self, dt, value, delivery_point, bool_entry):
        super().__init__(dt, 'physical_flow', value)
        self.delivery_point = delivery_point
        self.curve_name = 'entry' if bool_entry else 'exit'
        self.curve_type = 'commercial_flow'

    def set_df_row(self):
        df_row = super().set_df_row()
        df_row['curve_name'] = self.curve_name
        return df_row


class XLSData:
    """
    class to get all the data from .xlsx in the DataFrame
    """
    def __init__(self, file_name, data_type):
        self.data_type = data_type
        self.df = pd.DataFrame(columns=['date', 'delivery_point', 'from_country', 'to_country',
                                        'curve_type', 'flow_type', 'value'])
        if self.data_type == 'commercial_flows':
            self.df['curve_name'] = []
        if self.data_type == 'consumption':
            self.read_xls(file_name)
        else:
            self.go_through_sheets(file_name)
        Path(file_name).unlink()

    def read_xls(self, file_name):
        """
        function to read data from .xlsx  one sheet 3-6 columns and put it in DataFrame
        (consumptions)
        """
        new_column_names = ['date', 'industrial_demand', 'LDZ_demand', 'power_demand', 'other_demand']
        xls_df = pd.read_excel(file_name, header=2, usecols='A,C:F', names=new_column_names)
        curve_types = set(xls_df.columns) - {'date'}
        for row in range(len(xls_df)):
            dt = xls_df.loc[row, 'date']
            for curve_type in curve_types:
                value = xls_df.loc[row, curve_type]
                if not np.isnan(value):
                    df_row = DataRow(dt, curve_type, value)
                    self.df.loc[len(self.df.index)] = df_row.set_df_row()

    def go_through_sheets(self, file_name):
        """
        function to go through all the sheets in .xlsx
        (for flows)
        """
        xls = pd.ExcelFile(file_name)
        for sheet_num in range(len(xls.sheet_names)):
            delivery_point = xls.sheet_names[sheet_num]
            if self.data_type == 'commercial_flows':  # for commercial flows
                sheet = xls.parse(delivery_point, header=3)
            else:  # for physical flows
                sheet = xls.parse(delivery_point, header=3, usecols='A:B')
            self.go_through_df_rows(sheet, delivery_point)

    def go_through_df_rows(self, sheet, delivery_point):
        """
        function to go through all DataFrameRows
        (for flows)
        """
        for row in range(len(sheet)):
            dt = sheet.iloc[row, 0]
            if self.data_type == 'commercial_flows':  # for commercial flows
                self.get_last_not_empty_col(sheet, row, dt, delivery_point)
            else:   # for physical flows
                value = sheet.iloc[row, 1]
                if not np.isnan(value):
                    df_row = DataRow(dt, 'physical_flow', value)
                    df_row.delivery_point = delivery_point
                    df_row.get_countries()
                    self.df.loc[len(self.df.index)] = df_row.set_df_row()

    def get_last_not_empty_col(self, sheet, row, dt, delivery_point):
        """
        function to find last not empty column
        (for commercial flows)
        """
        def get_df_row(value, bool_entry):
            """
            inner function to set row and put it in DataFrame
            """
            df_row = BiggerDataRow(dt, value, delivery_point, bool_entry)
            df_row.get_countries()
            self.df.loc[len(self.df.index)] = df_row.set_df_row()

        for col in range(len(sheet.columns) - 2, 0, -2):
            value = sheet.iloc[row, col]  # for curve_name = 'entry'
            if not np.isnan(value):
                get_df_row(value, True)

                value = sheet.iloc[row, col + 1]   # for curve_name = 'exit' only if entry is not empty
                if not np.isnan(value):
                    get_df_row(value, False)
                break


class GRTgazParser:
    """
    class to get data from GRTgaz in the DataFrame form
    """
    def __init__(self, data_type):
        data_types = {'consumption', 'commercial_flows', 'physical_flows'}
        if data_type not in data_types:
            raise ValueError("results: status must be one of %r." % data_types)
        self.data_type = data_type
        match self.data_type:
            case 'consumption':
                self.url = 'https://www.smart.grtgaz.com/api/v1/en/consommation/export/Zone.xls'
            case 'commercial_flows':
                self.url = 'https://www.smart.grtgaz.com/api/v1/en/flux_commerciaux/export/PIR.xls'
            case 'physical_flows':
                self.url = 'https://www.smart.grtgaz.com/api/v1/en/flux_physiques/export/PIR.xls?'
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
        if self.data_type == 'commercial_flows':
            params['range'] = 'daily'
        get_data_from_resource(file_name, self.url, params)
        return XLSData(file_name, self.data_type).df

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
            if self.data_type == 'commercial_flows':
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
        return df


if __name__ == '__main__':
    data_types = ('consumption', 'commercial_flows', 'physical_flows')
    data_type = data_types[2]
    cparser = GRTgazParser(data_type)
    historical_data = cparser.get_historical_data()
    current_data = cparser.get_current_data()

    historical_data.to_csv(f'all_GRTgaz_{data_type}.csv')
    current_date = date.today().strftime('%d.%m.%Y')
    current_data.to_csv(f'GRTgaz_{data_type}_{current_date}.csv')


