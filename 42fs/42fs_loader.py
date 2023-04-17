import sys
from db_initial_connection import base, session
from table_classes import ProductsDict, InstrumentsDict, PricesCurveDict, CurvesDict, Curves, DeliveryPoint
import pandas as pd
import numpy as np
from gas_parser import GASParser
from power_parser import POWERParser
import datetime
import tqdm

import os
from dotenv import load_dotenv

load_dotenv('/srv/sstd/.env')

class Loader42fs:
    def __init__(self):
        self.curves_class = Curves()
        self.products_class = ProductsDict()
        self.instruments_class = InstrumentsDict()
        self.prices_curve_class = PricesCurveDict()
        self.curves_dict_class = CurvesDict()
        self.delivery_point = DeliveryPoint()

        self._countries = {'Czech_base': 'Czechia', 'Hungary_base': 'Hungary',
                           'Poland_base_PLN': 'Poland', 'Slovak_base': 'Slovakia'}

    def validate_dp(self, point_name: str, point_type: str = 'electricity_region',
                    source: str = '42 Financial Services'):
        country = self._countries[point_name]

        search_country = session.query(
            base.classes.country_dict.id).filter(
            base.classes.country_dict.country_name == country).all()

        if len(search_country) != 1:
            raise IndexError(country)
        else:
            id_country = search_country[0]

        search_point_type = session.query(
            base.classes.delivery_point_types_dict.id).filter(
            base.classes.delivery_point_types_dict.point_type == point_type).all()

        if len(search_point_type) != 1:
            raise IndexError(point_type)
        else:
            id_type = search_point_type[0]

        search_source = session.query(
            base.classes.source_dict.id).filter(
            base.classes.source_dict.source_name == source).all()

        if len(search_source) != 1:
            raise IndexError(source)
        else:
            id_source = search_source[0]

        inserted_id = self.delivery_point.insert_new_data(id_type[0], id_country[0], id_source[0], point_name)
        return inserted_id

    def validate_product(self, point_name: str, currency: str, unit: str, product_type: str, market: str,
                         code: str, date: datetime):
        beg_date = date
        end_date = date

        search_dp = session.query(
            base.classes.delivery_point_dict.id).filter(
            base.classes.delivery_point_dict.point_name == point_name).all()

        match len(search_dp):
            case 0:
                id_dp = self.validate_dp(point_name)
            case 1:
                id_dp = search_dp[0][0]
            case _:
                raise IndexError(f'there are more than one record in delivery_point_dict with a name {point_name}')

        search_market = session.query(
            base.classes.markets_dict.id).filter(
            base.classes.markets_dict.market_name == market).all()

        if len(search_market) != 1:
            raise IndexError(market)
        else:
            id_market = search_market[0]

        search_currency = session.query(
            base.classes.currencies_dict.id).filter(
            base.classes.currencies_dict.currency_code == currency).all()

        if len(search_currency) != 1:
            raise IndexError(currency)
        else:
            id_currency = search_currency[0]

        search_unit = session.query(
            base.classes.units_dict.id).filter(
            base.classes.units_dict.unit_name == unit).all()

        if len(search_unit) != 1:
            raise IndexError(unit)
        else:
            id_unit = search_unit[0]

        search_product_type = session.query(
            base.classes.product_types_dict.id).filter(
            base.classes.product_types_dict.product_type == product_type).all()

        if len(search_product_type) != 1:
            raise IndexError(product_type)
        else:
            id_product_type = search_product_type[0]

        products_table = base.classes.products_dict
        check_exists = session.query(products_table.id).filter(
            products_table.id_delivery_point == id_dp,
            products_table.id_currency == id_currency[0],
            products_table.id_unit == id_unit[0],
            products_table.id_market == id_market[0],
            products_table.id_product_type == id_product_type[0],
            products_table.code == code,
            products_table.beg_date == beg_date,
            end_date == end_date).all()

        match len(check_exists):
            case 0:
                inserted_id = self.products_class.insert_new_data(id_dp, id_currency[0], id_unit[0], id_market[0],
                                                                  id_product_type[0], beg_date, end_date, code)
                return inserted_id
            case 1:
                return check_exists[0][0]
            case _:
                raise IndexError(f'there are more than one record in products_dict with fields: '
                                 f'\ncode = {code}'
                                 f'\nid_product_type = {product_type}'
                                 f'\nid_delivery_point = {id_dp}'
                                 f'\nid_currency = {id_currency}'
                                 f'\nid_unit = {id_unit}'
                                 f'\nid_market = {id_market}'
                                 f'\nbeg_date = {beg_date}'
                                 f'\nend_date = {end_date}')

    def validate_instrument(self, hub: str, hub2, currency: str, unit: str, product_type: str, market: str,
                            code: str, date: datetime):
        id_product1 = self.validate_product(hub, currency, unit, product_type, market, code, date)

        if hub2 == '-':
            instrument_type = 'Single'
            id_product2 = None
        else:
            instrument_type = 'Spread'
            id_product2 = self.validate_product(hub2, currency, unit, product_type, market, code, date)

        search_instrument_type = session.query(
            base.classes.instrument_types_dict.id).filter(
            base.classes.instrument_types_dict.instrument_type == instrument_type).all()

        if len(search_instrument_type) != 1:
            raise IndexError(instrument_type)
        else:
            id_instrument_type = search_instrument_type[0]

        instruments_table = base.classes.instruments_dict
        check_exists = session.query(instruments_table.id).filter(
            instruments_table.id_product_1 == id_product1,
            instruments_table.id_product_2 == id_product2,
            instruments_table.id_instrument_type == id_instrument_type[0]).all()

        match len(check_exists):
            case 0:
                inserted_id = self.instruments_class.insert_new_data(id_product1, id_product2, id_instrument_type[0])
                return inserted_id
            case 1:
                return check_exists[0][0]
            case _:
                raise IndexError(f'there are more than one record in instruments_dict with fields: '
                                 f'\nid_instrument_type = {id_instrument_type}'
                                 f'\nid_product_1 = {id_product1}'
                                 f'\nid_product_2 = {id_product2}')

    def validate_prices_curve(self, source: str, id_instrument: int, price_type: str, description: str):
        search_source = session.query(
            base.classes.source_dict.id).filter(
            base.classes.source_dict.source_name == source).all()

        if len(search_source) != 1:
            raise IndexError(source)
        else:
            id_source = search_source[0]

        search_price_type = session.query(
            base.classes.prices_type_dict.id).filter(
            base.classes.prices_type_dict.price_type == price_type).all()

        if len(search_price_type) != 1:
            raise IndexError(price_type)
        else:
            id_type = search_price_type[0]

        prices_curve_table = base.classes.prices_curve_dict
        check_exists = session.query(prices_curve_table.id).filter(
            prices_curve_table.id_source == id_source[0],
            prices_curve_table.id_instrument == id_instrument,
            prices_curve_table.id_type == id_type[0]).all()

        match len(check_exists):
            case 0:
                inserted_id = self.prices_curve_class.insert_new_data(id_source[0], id_instrument, id_type[0], description)
                return inserted_id
            case 1:
                return check_exists[0][0]
            case _:
                raise IndexError(f'there are more than one record in prices_curve_dict with fields: '
                                 f'\nid_source = {id_source}'
                                 f'\nid_type = {id_type}'
                                 f'\nid_instrument = {id_instrument}')

    def validate_curves_dict(self, id_prices_curves: int, sector_name: str):
        search_sector = session.query(
            base.classes.sector_dict.id).filter(
            base.classes.sector_dict.sector_name == sector_name).all()

        if len(search_sector) != 1:
            raise IndexError(sector_name)
        else:
            id_sector = search_sector[0]

        curves_dict_table = base.classes.curves_dict
        check_exists = session.query(curves_dict_table.id).filter(
            curves_dict_table.id_prices_curves == id_prices_curves,
            curves_dict_table.id_sector == id_sector[0]).all()

        match len(check_exists):
            case 0:
                inserted_id = self.curves_dict_class.insert_new_data(id_sector[0], id_prices_curves)
                return inserted_id
            case 1:
                return check_exists[0][0]
            case _:
                raise IndexError(f'there are more than one record in curves_dict with fields: '
                                 f'\nid_prices_curves = {id_prices_curves}'
                                 f'\nid_sector = {id_sector}')

    def insert_42fs(self, source: str = '42 Financial Services', sector: str = 'currency prices',
                    df_fs: pd.DataFrame = None, market_type: str = None):
        match market_type:
            case 'gas':
                market = 'Natural Gas'
            case 'power':
                market = 'Electricity'
            case _:
                raise ValueError('wrong market_type, it can only be "gas" or "power"')

        data = df_fs
        data['id_instrument'] = data.apply(
            lambda x: self.validate_instrument(
                x.hub, x.hub2, x.currency, x.unit, x.product_type, market, x.products, x.date), axis=1)
        data['id_prices_curves'] = data.apply(
            lambda x: self.validate_prices_curve(
                source, x.id_instrument, x.price_type, x.prices_name), axis=1)
        data['id_curve'] = data.apply(
            lambda x: self.validate_curves_dict(x.id_prices_curves, sector), axis=1)
        data = data[['id_curve', 'date', 'price']].to_numpy()
        for row in tqdm.tqdm(data):
            try:
                self.curves_class.insert_new_data(row[0], row[1], np.float64(row[2]))
            except:
                sys.exit(1)
        return 'ok'


if __name__ == '__main__':
    file_name_gas = r'ClosingDayPricesGAS2023.xlsx'
    result_gas = GASParser(file_name_gas)
    Loader42fs().insert_42fs(df_fs=result_gas.df, market_type='gas')
    file_name_power = r'ClosingDayPricesPOWER2023.xlsx'
    result_power = POWERParser(file_name_power)
    result_gas = pd.read_csv(r'ClosingDayPricesGAS2023.csv')
    Loader42fs().insert_42fs(df_fs=result_power.df, market_type='power')
