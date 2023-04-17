from db_initial_connection import engine, base, session
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
import numpy as np


class ProductsDict:
    def __init__(self):
        # products_dict
        self._products_table = base.classes.products_dict

    @staticmethod
    def next_table_id():
        return session.execute('SELECT nextval('"'curves_id_seq'"');').scalar()

    def insert_new_data(self, id_dp: int, id_currency: int, id_unit: int, id_market: int, id_product_type: int,
                        beg_date: datetime, end_date: datetime, code: str):
        insert_statement = insert(self._products_table, bind=engine).values(
            id_delivery_point=id_dp,
            id_currency=id_currency,
            id_unit=id_unit,
            id_market=id_market,
            id_product_type=id_product_type,
            beg_date=beg_date,
            end_date=end_date,
            code=code,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            index_elements=['id_delivery_point', 'id_currency', 'id_unit', 'id_market',
                            'id_product_type', 'beg_date', 'end_date', 'code'],
            set_={'update_time': insert_statement.excluded.update_time},
        )
        inserted_row = session.execute(update_statement)
        session.commit()
        return inserted_row.inserted_primary_key[0]


class InstrumentsDict:
    def __init__(self):
        # instruments_dict
        self._instruments_table = base.classes.instruments_dict

    def insert_new_data(self, id_product_1: int, id_product_2: int, id_instrument_type: int):
        insert_statement = insert(self._instruments_table, bind=engine).values(
            id_product_1=id_product_1,
            id_product_2=id_product_2,
            id_instrument_type=id_instrument_type,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            index_elements=['id_product_1', 'id_product_2', 'id_instrument_type'],
            set_={'update_time': insert_statement.excluded.update_time},
        )
        inserted_row = session.execute(update_statement)
        session.commit()
        return inserted_row.inserted_primary_key[0]


class PricesCurveDict:
    def __init__(self):
        # prices_curve_dict
        self._prices_curve_table = base.classes.prices_curve_dict

    def insert_new_data(self, id_source: int, id_instrument: int, id_type: int, description: str):
        insert_statement = insert(self._prices_curve_table, bind=engine).values(
            id_source=id_source,
            id_instrument=id_instrument,
            id_type=id_type,
            description=description,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            index_elements=['id_source', 'id_instrument', 'id_type'],
            set_={'update_time': insert_statement.excluded.update_time},
        )
        inserted_row = session.execute(update_statement)
        session.commit()
        return inserted_row.inserted_primary_key[0]


class CurvesDict:
    def __init__(self):
        # curves_dict
        self._curves_dict_table = base.classes.curves_dict

    def insert_new_data(self, id_sector, id_prices_curves: int):
        insert_statement = insert(self._curves_dict_table, bind=engine).values(
            id_sector=id_sector,
            id_prices_curves=id_prices_curves,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            index_elements=['id'],
            set_={'update_time': insert_statement.excluded.update_time},
        )
        inserted_row = session.execute(update_statement)
        session.commit()
        return inserted_row.inserted_primary_key[0]


class Curves:
    def __init__(self):
        # curves
        self._curves_table = base.classes.curves

    def insert_new_data(self, id_curve, date: datetime, value: np.float64):
        insert_statement = insert(self._curves_table, bind=engine).values(
            id_curve=id_curve,
            date=date,
            value=value,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            index_elements=['id_curve', 'date'],
            set_={'value': insert_statement.excluded.value,
                  'update_time': insert_statement.excluded.update_time},
        )
        session.execute(update_statement)
        session.commit()


class DeliveryPoint:
    def __init__(self):
        # delivery_point
        self._dp_table = base.classes.delivery_point_dict
        # self._countries = {'Chech_base': 'Czechia'}

    def insert_new_data(self, id_type: int, id_country: int, id_source: int, point_name: str):
        insert_statement = insert(self._dp_table, bind=engine).values(
            id_type=id_type,
            # id_country=id_country,
            id_source=id_source,
            point_name=point_name,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            index_elements=['id'],
            set_={'update_time': insert_statement.excluded.update_time},
        )
        inserted_row = session.execute(update_statement)
        session.commit()
        return inserted_row.inserted_primary_key[0]
