import pandas as pd
from datetime import date, datetime, timedelta
import os

DATE_FORMAT = '%d-%b-%Y'
FIRST_MONTHS = ['Today', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ALL_MONTHS = FIRST_MONTHS + ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']

columns_to_drop = ['Available Capacity', 'Injectability', 'Deliverability']
columns_to_rename = {
    'Gasday': 'date',
    'Site Name': 'delivery_point', 
    'Operator Type': 'point_type'
    }
point_type_to_replace = {
    'LNG': 'lng_terminal',
    'STORAGE': 'ugs'
}


def delete_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    deletes rows with nan
    """
    nans = list(df.loc[pd.isna(df["Gasday"]), :].index)
    df = df.drop(labels=range(nans[0], len(df.index)), axis=0)
    return df


def get_previous_date(df: pd.DataFrame, days: int = 1) -> datetime:
    """
    gets the date 'days' days ago (default is 1)
    """
    cur_date_str = df.loc[1, 'Gasday']
    cur_date_obj = datetime.strptime(cur_date_str, DATE_FORMAT)
    prev_date_obj = cur_date_obj - timedelta(days=days)
    return prev_date_obj


def cut_data_with_date(df: pd.DataFrame, date_obj: datetime) -> pd.DataFrame:
    """
    cuts the data from DataFrame with the date 'date_obj'
    """
    return df.loc[df['Gasday'] == date_obj.strftime(DATE_FORMAT)]


def join_dfs(df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    """
    joins two dataframes
    """
    joined_df = pd.concat([df, temp_df], sort=False, axis=0)
    joined_df.reset_index(drop=True, inplace=True)
    return joined_df


def change_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    changes the view of the DataFrame as in the database
    """
    # rename columns
    df = df.rename(columns=columns_to_rename)
    # convert date column type to datetime
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    # put columns with values as rows
    id_vars = list(columns_to_rename.values())
    value_vars = set(df.columns) - set(columns_to_rename.values())
    df = pd.melt(frame=df,
                 id_vars=id_vars,
                 value_vars=value_vars,
                 var_name='curve_name', value_name='value')
    # replace point_type as in the database
    df = df.replace({'point_type': point_type_to_replace}) 
    return df


def get_current_data() -> pd.DataFrame:
    """
    gets the current data in the DataFrame form
    """
    # read current data from file Todays sheet
    current_date = date.today().strftime('%d.%m.%Y')
    current_file_name = f'current_data_{current_date}.xls'
    df = pd.read_excel(current_file_name, sheet_name='Today')

    # delete rows with nan
    df = delete_nan(df)
    
    # read sheet with a month of a previous day
    prev_date = get_previous_date(df)
    temp_df = pd.read_excel(current_file_name, sheet_name=prev_date.strftime("%b"))
    
    # cut data only with a previous date
    temp_df = cut_data_with_date(temp_df, prev_date)
    
    # join two dataframes
    df = join_dfs(df, temp_df)

    # drop unnecessary columns
    df = df.drop(columns_to_drop, axis=1)
    df = change_view(df)
    # os.unlink(current_file_name)
    return df


def go_through_sheets(file_name: str, sheet_names: list) -> pd.DataFrame:
    """
    gets data from sheets in the file 'file_name' into one DataFrame
    """
    all_sheet_df = pd.DataFrame()
    for sheet_name in sheet_names:
        sheet_df = pd.read_excel(file_name, sheet_name=sheet_name)
        if sheet_name == 'Today':
            sheet_df = delete_nan(sheet_df)
    all_sheet_df = join_dfs(all_sheet_df, sheet_df)
    return all_sheet_df


def get_historical_data(file_name: str, start_year: int = 2015, end_year: int = date.today().year) -> pd.DataFrame:
    """
    gets data from start_year till end_year from the files 'file_name' + year into one DataFrame
    """
    out_df = pd.DataFrame()
    for year in range(start_year, end_year + 1):
        current_file_name = f'{file_name}_{year}.xls'
        xls = pd.ExcelFile(current_file_name)
        all_sheet_names = xls.sheet_names
        if year == 2015:
            sheet_names = list(set(all_sheet_names) & set(FIRST_MONTHS)) 
        else:
            sheet_names = list(set(all_sheet_names) & set(ALL_MONTHS)) 
        year_df = go_through_sheets(file_name=current_file_name, sheet_names=sheet_names)
        out_df = join_dfs(out_df, year_df)
        out_df = out_df.drop(columns_to_drop, axis=1)
    out_df = change_view(out_df)
    return out_df
                              

if __name__ == '__main__':
    folder = 'parsed_data'

    # current_date = date.today().strftime('%d.%m.%Y')
    # current_file_name = f'{folder}/NG_{current_date}.csv'
    # df = get_current_data()
    # df.to_csv(current_file_name, index=False)
    
    df = get_historical_data(f'{folder}/NG')
    df.to_csv(f'{folder}/NG.csv', index=False)
    