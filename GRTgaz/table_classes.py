from connection import engine, base, session
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
import numpy as np


class FlowCurves:
    def __init__(self):
        # flow_curves
        self._flow_curves = base.classes.flow_curves

    def search_data(self, id_source: int, id_point: int, id_unit: int, from_country: int,
                        to_country: int, from_company: int, to_company: int, id_type: int, curve_name: str):
        search_record = session.query(
            self._flow_curves.id).filter(
            self._flow_curves.id_source == id_source,
            self._flow_curves.id_point == id_point,
            self._flow_curves.id_unit == id_unit,
            self._flow_curves.from_country == from_country,
            self._flow_curves.to_country == to_country,
            self._flow_curves.from_company == from_company,
            self._flow_curves.to_company == to_company,
            self._flow_curves.id_type == id_type,
            self._flow_curves.curve_name == curve_name).all()

        match len(search_record):
            case 0:
                return self.insert_new_data(id_source=id_source,
                                            id_point=id_point,
                                            id_unit=id_unit,
                                            from_country=from_country,
                                            to_country=to_country,
                                            from_company=from_company,
                                            to_company=to_company,
                                            id_type=id_type,
                                            curve_name=curve_name)
            case 1:
                return search_record[0][0]
            case _:
                raise IndexError(f'there are more than one record in flow_curves with fields:'
                                 f'\nid_source = {id_source}'
                                 f'\nid_point = {id_point}'
                                 f'\nid_unit = {id_unit}'
                                 f'\nfrom_country = {from_country}'
                                 f'\nto_country = {to_country}'
                                 f'\nfrom_company = {from_company}'
                                 f'\nto_company = {to_company}'
                                 f'\nid_type = {id_type}'
                                 f'\ncurve_name = {curve_name}')

    def insert_new_data(self, id_source: int, id_point: int, id_unit: int, from_country: int,
                        to_country: int, from_company: int, to_company: int, id_type: int, curve_name: str):
        insert_statement = insert(self._flow_curves, bind=engine).values(
            id_source=id_source,
            id_point=id_point,
            id_unit=id_unit,
            from_country=from_country,
            to_country=to_country,
            from_company=from_company,
            to_company=to_company,
            id_type=id_type,
            curve_name=curve_name,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            index_elements=['id'],
            set_={'update_time': insert_statement.excluded.update_time},
        )
        inserted_row = session.execute(update_statement)
        session.commit()
        return inserted_row.inserted_primary_key[0]


class CurvesDict:
    def __init__(self):
        # curves_dict
        self._curves_dict_table = base.classes.curves_dict

    def search_data(self, id_sector: int, id_flow_curves: int):
        search_record = session.query(
            self._curves_dict_table.id).filter(
            self._curves_dict_table.id_sector == id_sector,
            self._curves_dict_table.id_flow_curves == id_flow_curves).all()

        match len(search_record):
            case 0:
                return self.insert_new_data(id_sector=id_sector,
                                            id_flow_curves=id_flow_curves)
            case 1:
                return search_record[0][0]
            case _:
                raise IndexError(f'there are more than one record in flow_curves with fields:'
                                 f'\nid_sector = {id_sector}'
                                 f'\nid_flow_curves = {id_flow_curves}')

    def insert_new_data(self, id_sector, id_flow_curves: int):
        insert_statement = insert(self._curves_dict_table, bind=engine).values(
            id_sector=id_sector,
            id_flow_curves=id_flow_curves,
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

