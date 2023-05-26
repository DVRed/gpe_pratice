import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Literal, List, Mapping
from fluxys_collector import Data_Type, collect_fluxys_data


POINTS_COUNTRIES: Mapping[str, Mapping[Literal['from', 'to'], str]] = {
    'Alveringem': {
        'from': 'FR',
        'to': 'BE'
    },
    'Blaregnies L': {
        'from': 'FR',
        'to': 'BE'
    },
    'Blaregnies Segeo': {
        'from': 'FR',
        'to': 'BE'
    },
    'Blaregnies Troll': {
        'from': 'FR',
        'to': 'BE'
    },
    'Bras-Petange': {
        'from': 'LU',
        'to': 'BE'
    },
    'Dunkerque LNG Terminal': {
        'from': 'FR',
        'to': 'BE'
    },
    'Eynatten 1': {
        'from': 'DE',
        'to': 'BE'
    },
    'Eynatten 2': {
        'from': 'DE',
        'to': 'BE'},
    'Hilvarenbeek': {
        'from': 'NL',
        'to': 'BE'},
    'IZT': {'from': 'GB',
            'to': 'BE'
            },
    'LNG Terminal': {
        'from': 'NO',
        'to': 'BE'
    },  # ????
    'Loenhout': {
        'from': 'BE',
        'to': 'BE'},
    'Poppel': {
        'from': 'NL',
        'to': 'BE'
    },
    'Quality Conversion H': {
        'from': 'BE',
        'to': 'BE'
    },
    'Quality Conversion L': {
        'from': 'BE',
        'to': 'BE'
    },
    "'s Gravenvoeren": {
        'from': 'NL',
        'to': 'BE'
    },
    'Zandvliet H': {
        'from': 'NL',
        'to': 'BE'
    },
    'Zeebrugge Beach': {
        'from': 'BE',
        'to': 'BE'
    },
    'Zeebrugge LNG Terminal': {
        'from': 'BE',
        'to': 'BE'
    },
    'Zelzate 1': {
        'from': 'NL',
        'to': 'BE'
    },
    'Zelzate 2': {
        'from': 'NL',
        'to': 'BE'
    },
    'ZPT': {
        'from': 'NO',
        'to': 'BE'
    },
    'Hilvarenbeek L': {
        'from': 'NL',
        'to': 'BE'
    },
    'VIP BENE': {
        'from': 'NL',
        'to': 'BE'
    },
    'VIP THE-ZTP': {
        'from': 'BE',
        'to': 'BE'
    },
    'Virtualys': {
        'from': 'FR',
        'to': 'BE'
    },
    'Zeebrugge': {
        'from': 'BE',
        'to': 'BE'
    }
}

columns_to_drop: Mapping['Data_Type', List] = {
    'domestic': ['Balancing Zone ', 'Client Type', 'Subgrid', 'Allocations (kWh)', 'GCV'],
    'interconnection': ['Allocations (kWh)', 'Measured GCV\n(kWh/m³(n))']
}

COLUMNS_TO_RENAME: Mapping['Data_Type', Mapping[str, str]] = {
    'domestic': {
        'Gas day': 'date',
        'Nature': 'curve_name'
    },
    'interconnection': {
        'Interconnection Point': 'delivery_point',
        'Direction': 'curve_name',
        'Gas day': 'date'
    }
}

FILE_NAMES: Mapping['Data_Type', str] = {
    'domestic': 'FlowsNominations_DE.xlsx',
    'interconnection': 'FlowsNominations_IP.xlsx'
}

DATE_FORMAT = '%d/%m/%Y'


def change_dates_format(start_date: date, end_date: date) -> tuple[str, str]:
    """
    Change the format of the dates to fit the fluxys
    """
    from_date = start_date.strftime(DATE_FORMAT)
    to_date = end_date.strftime(DATE_FORMAT)
    return from_date, to_date


def collect_data(data_type: Data_Type, start_date: date, end_date: date) -> pd.DataFrame:
    """
    collects data from fluxys with set data_type, start_date and end_date
    """
    from_date, to_date = change_dates_format(start_date=start_date, end_date=end_date)
    collect_fluxys_data(data_type=data_type,
                        from_date=from_date,
                        to_date=to_date,
                        file_name=FILE_NAMES[data_type])
    df = FluxysDataFrame(data_type).data_frame
    os.unlink(FILE_NAMES[data_type])
    return df


def get_current_data(data_type: Data_Type) -> pd.DataFrame:
    """
    function to get data for the past two days in the DataFrame form
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=2)
    print(f'Collecting {data_type} data for the past 2 days...')
    df = collect_data(data_type=data_type,
                      start_date=start_date,
                      end_date=end_date)
    return df


def get_historical_data(data_type, start_year: int = 2015, end_year: int = date.today().year):
    """
    function to get historical data with set data type from start_year till end_year
    default: start_year = 2015, end_year = current year
    """
    df = pd.DataFrame()
    for year in range(start_year, end_year + 1):
        start_date = date(year, 1, 1)
        end_date = date.today() if year == date.today().year else date(year, 12, 31)

        print(f'Collecting {data_type} data for {year} year...')
        temp_df = collect_data(data_type=data_type,
                               start_date=start_date,
                               end_date=end_date)
        df = pd.concat([df, temp_df], sort=False, axis=0)
        df.reset_index(drop=True, inplace=True)
    return df


class FluxysDataFrame:
    """
    class to set dataframe for the fluxys
    """
    def __init__(self, data_type: Data_Type):
        self._df = pd.read_excel(FILE_NAMES[data_type])
        self._columns_to_rename: Mapping['Data_Type', Mapping[str, str]] = {
            'domestic': {
                'Gas day': 'date',
                'Nature': 'curve_name'
            },
            'interconnection': {
                'Interconnection Point': 'delivery_point',
                'Direction': 'curve_name',
                'Gas day': 'date'
            }
        }
        self._get_countries(data_type=data_type)
        self._rename_columns(data_type=data_type)
        self._change_view()
        self._melt_flow_types()
        self._divide_flow_types()
        
        # replace NaN values with 0
        # self._df.value = self._df.apply(lambda x: 0 if np.isnan(x.value) else x.value, axis=1)
        # drop rows with NaN values
        self._df.drop(self._df[np.isnan(self._df.value)].index, inplace = True)

    def _get_countries(self, data_type: Data_Type) -> None:
        """
        gets from and to countries for delivery points
        also gets delivery_point column from Balancing Zone and Client Type columns
        """
        if data_type == 'domestic':
            self._df[['from_country', 'to_country']] = ['BE', 'BE']
            self._df['delivery_point'] = self._df['Balancing Zone '] + '_' + self._df['Client Type']
        else:
            self._df[['from_country', 'to_country']] = self._df.apply(lambda row: pd.Series(
                [
                    POINTS_COUNTRIES[row['Interconnection Point']]['from'],
                    POINTS_COUNTRIES[row['Interconnection Point']]['to']
                ]), axis=1)

    def _rename_columns(self, data_type: Data_Type) -> None:
        """
        renames columns as set in database
        """
        self._df = self._df.rename(columns=self._columns_to_rename[data_type])

    def _change_view(self) -> None:
        """
        changes the view of date and curve_name columns
        """
        # convert date column type to datetime
        self._df['date'] = pd.to_datetime(self._df['date'], dayfirst=True)
        # get values in curve_name column in lowercase
        self._df['curve_name'] = self._df['curve_name'].str.lower()

    def _melt_flow_types(self) -> None:
        """
        renames columns with values and puts them as rows
        """
        com_columns_to_rename = {
            'Day Ahead Nominations (kWh)': 'day_ahead_nominations',
            'Final Nominations (kWh)': 'final_nominations',
            'Physical Flow (kWh)': 'physical_flow'}
        id_vars = ['delivery_point', 'date', 'curve_name', 'from_country', 'to_country']
        self._df = self._df.rename(columns=com_columns_to_rename)
        self._df = pd.melt(frame=self._df,
                           id_vars=id_vars,
                           value_vars=list(com_columns_to_rename.values()),
                           var_name='flow_type', value_name='value')

    def _divide_flow_types(self) -> None:
        """
        extracts nomination type and puts it in curve_name column
        """
        self._df[['curve_name', 'flow_type']] = self._df.apply(lambda row: pd.Series([
            f'{row.curve_name}_{row.flow_type[:row.flow_type.rfind("_")]}',
            'nomination']) if row.flow_type.find('nominations') != -1
            else pd.Series([row.curve_name, row.flow_type]), axis=1)

    @property
    def data_frame(self) -> pd.DataFrame:
        """
        returns changed data frame
        """
        return self._df


if __name__ == '__main__':
    data_types = ('domestic', 'interconnection')
    data_type = data_types[0]
    folder = 'parsed_data'
    
    # to get all data
    # historical_data = get_historical_data(data_type)
    # historical_data.to_csv(f'{folder}/all_Fluxys_{data_type}.csv', index=False)
    
    # to get current data
    current_data = get_current_data(data_type)
    current_date = date.today().strftime('%d.%m.%Y')
    current_data.to_csv(f'{folder}/Fluxys_{data_type}_{current_date}.csv', index=False)
 
