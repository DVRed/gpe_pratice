import sys
import pandas as pd
import numpy as np
import tqdm
from typing import Literal, Mapping, Union
from connection import connect, session, base
from table_classes import CurvesDict, Curves, FlowCurves

attr_dict: Mapping[str, 'one_attr_tables'] = {
    'unit': 'units_dict', 'delivery_point': 'delivery_point_dict',
    'from_country': 'country_dict', 'to_country': 'country_dict',
    'source': 'source_dict', 'flow_type': 'flow_types',
    'sector': 'sector_dict'}

new_columns: Mapping[str, str] = {
    'unit': 'id_unit', 'delivery_point': 'id_point',
    'from_country': 'from_country', 'to_country': 'to_country',
    'source': 'id_source', 'flow_type': 'id_type',
    'sector': 'id_sector'}

one_attr_tables = Literal['units_dict', 'delivery_point_dict', 'country_dict',
'source_dict', 'flow_types', 'sector_dict']

additional_columns: Mapping['str', Union[str, int]] = {'unit': 'kWh',
                      'source': 'GRTgaz',
                      'sector': 'flows',
                      'from_company': 999999,
                      'to_company': 999999
                                                       }


def search_record_in_table(table_name: one_attr_tables, attr: str) -> int:
    table = field = None
    match table_name:
        case 'units_dict':
            table = base.classes.units_dict
            field = table.unit_name
        case 'delivery_point_dict':
            table = base.classes.delivery_point_dict
            field = table.point_name
        case 'country_dict':
            table = base.classes.country_dict
            field = table.country_2
        case 'flow_types':
            table = base.classes.flow_types
            field = table.flow_type
        case 'source_dict':
            table = base.classes.source_dict
            field = table.source_name
        case 'sector_dict':
            table = base.classes.sector_dict
            field = table.sector_name

    search_record = session.query(
        table.id).filter(
        field == attr).all()

    match len(search_record):
        case 0:
            raise IndexError(f'there is no record  with a name {attr} in {table_name}')
        case 1:
            return search_record[0][0]
        case _:
            raise IndexError(f'there are more than one record in {table_name} with a name {attr}')


class GRTgazLoader:
    def __init__(self):
        self.curves = Curves()
        self.curves_dict = CurvesDict()
        self.flow_curves = FlowCurves()

    def insert_grtgaz(self, df_fs: pd.DataFrame = None):
        data = df_fs

        # convert all string data to id where possible
        for col_name, value in additional_columns.items():
            data[col_name] = value
        for old_col, new_col in new_columns.items():
            data[new_col] = data.apply(
                lambda x: search_record_in_table(
                    table_name=attr_dict[old_col], attr=x[old_col]), axis=1)

        # get id_flow_curves from table flow_curves
        data['id_flow_curves'] = data.apply(
            lambda x: self.flow_curves.search_data(
                id_source=x.id_source, id_point=x.id_point,
                id_unit=x.id_unit, from_country=x.from_country,
                from_company=x.from_company, to_company=x.to_company,
                to_country=x.to_country, id_type=x.id_type,
                curve_name=x.curve_name), axis=1)

        # get id_curve from table curves_dict
        data['id_curve'] = data.apply(
            lambda x: self.curves_dict.search_data(
                id_sector=x.id_sector, id_flow_curves=x.id_flow_curves), axis=1)

        # insert data into table curves
        data = data[['id_curve', 'date', 'value']].to_numpy()
        for row in tqdm.tqdm(data):
            try:
                self.curves.insert_new_data(id_curve=row[0],
                                            date=row[1],
                                            value=np.float64(row[2]))
            except:
                sys.exit(1)
            connect.close()
        return 'ok'


if __name__ == '__main__':
    data_types = ('consumptions', 'commercial_flow', 'physical_flow')
    data_type = data_types[2]
    folder = 'parsed_data/'
    result = pd.read_excel(f'{folder}/all_GRTgaz_{data_type}.xlsx', index_col=False)
    GRTgazLoader().insert_grtgaz(df_fs=result)
