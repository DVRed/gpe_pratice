from __future__ import annotations

from datetime import timedelta

import pandas
from tqdm import tqdm
from exxeta_loader import *
from eex_ng.eex_ng_futures_parser import EexNaturalGasFuturesParser
from eex_ng.eex_ng_indices_parser import EexNaturalGasIndicesParser
from eex_ng.eex_ng_spot_parser import EexNaturalGasSpotParser
from snam.db_config import DBConfig, DBConfigInstance
from psycopg2.extensions import register_adapter

# TODO set config as environment variable
testing_config = DBConfig(DBMS='postgresql',
                          DRIVER='psycopg2',
                          HOSTNAME='localhost',
                          DATABASE='analytics_relized',
                          USERNAME='postgres',
                          PASSWORD='postgres',
                          config_name='My_test_config')

"""
Whole code except __main__ is referred from eex_loader.loader.py
"""

# действия по согласованию числовых форматов
register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)


class Price:
    def __init__(self, in_data_row: dict):
        self.__dict__.update(in_data_row)
        self.product_1 = self._get_product()

    def _get_product(self):
        product = {
            'in_delivery_point': {'point_name': self.hub, 'point_type': 'Natural Gas'},
            'in_currency': self.currency,
            'in_unit': self.unit,
            'in_market': 'Natural Gas',
            'in_product_type': self.product_type,
            'in_beg_date': self.beg_date,
            'in_end_date': self.end_date,
            'in_product_name': self.products,
            'in_comment': 'm_aleksandrov'
        }
        return product

    def __str__(self):
        result = []
        for item in self.__dir__():
            if not item.startswith('_'):
                result.append(f"{item}: {getattr(self, item)}")
            else:
                continue
        return '\n'.join(result)

    __repr__ = __str__


class DBLoaderPricesType(DBLoaderDeliveryPointType):
    table_name = 'prices_type_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = 'price_type'


class DBLoaderPriceCurveDict(DBLoaderDeliveryPointType):
    table_name = 'prices_curve_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = [
            'id_source', 'id_instrument', 'id_type'
        ]

    def insert_item(self, in_value: dict, in_table=None) -> int:
        in_table = self.table
        params = (self.base, self.session)
        value = {
            'id_source': in_value['id_source'],
            'id_instrument': DBLoaderInstrument(*params).insert_item(in_value['product_1'], None, 'Single'),
            'id_type': DBLoaderPricesType(*params).insert_item(in_value['price_type'])
        }
        value_id = DBLoader(*params).insert_item(value, in_table, self.check_column_name)
        return value_id


class DBLoaderCurvesDict(DBLoaderDeliveryPointType):
    table_name = 'curves_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = [
            'id_sector', 'time_period', 'id_prices_curves'
        ]

    def insert_item(self, in_value: dict, in_table=None) -> int:
        in_table = self.table
        params = (self.base, self.session)
        value = {
            'id_sector': 1,  # 'forward prices'
            'time_period': 1,  # 'THICK'
            'id_prices_curves': DBLoaderPriceCurveDict(*params).insert_item(in_value)
        }
        value_id = DBLoader(*params).insert_item(value, in_table, self.check_column_name)
        return value_id


class DBLoaderCurves(DBLoaderDeliveryPointType):
    table_name = 'curves'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()

    def _get_curves_value(self, in_curve: Price) -> dict:
        params = (self.base, self.session)
        value = {
            'id_curve': DBLoaderCurvesDict(*params).insert_item({
                'product_1': in_curve.product_1,
                'price_type': in_curve.price_type,
                'id_source': in_curve.id_source
            }),
            'date': in_curve.date,
            'value': in_curve.price
        }
        return value

    def insert_item(self, in_curve: Price, in_table=None) -> int:
        in_table = self.table
        params = (self.base, self.session)
        value = self._get_curves_value(in_curve)
        value_id = DBLoader(*params).insert_item(value, in_table)

        return value_id


def clean_prices(price: str | datetime) -> float:
    """Очищает цены от значений со сбившимся форматоми"""
    try:
        correct_price = float(price)
    except TypeError:
        # для возможности вытащить отдельные атрибуты даты и слепить потерянное значение цены
        date = datetime.strptime(str(price), '%Y-%m-%d %H:%M:%S')
        if date.day == 1:
            correct_price = f'{date.month}.{date.year}'
        else:
            correct_price = f'{date.day}.{date.month}'
        correct_price = float(correct_price)
    return correct_price


if __name__ == '__main__':
    futures = EexNaturalGasFuturesParser(end_date=datetime.today(),
                                         start_date=(datetime.today() - timedelta(days=3))).parse()
    indices = EexNaturalGasIndicesParser(end_date=datetime.today(),
                                         start_date=datetime.today() - timedelta(days=3)).parse()
    spots = EexNaturalGasSpotParser(end_date=datetime.today(),
                                    start_date=datetime.today() - timedelta(days=3)).parse()

    res = pandas.concat([indices, spots, futures])
    # res.to_excel('./output/parse_result.xlsx')

    connector = DBConnector()
    base = connector.connect_to_base()
    session = connector.create_session()
    for price_index, row in tqdm(res.iterrows(), ncols=100, total=res.shape[0],
                                 desc='EEX prices loader', disable=False):
        curve = Price(row.to_dict())
        DBLoaderCurves(base, session).insert_item(curve)

    session.close()
    connector.engine.dispose()
