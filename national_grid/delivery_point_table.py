import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../GRTgaz'))
from connection import engine, base, session
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime


class DeliveryPointDict:
    def __init__(self):
        # delivery_point_dict
        self._delivery_point_dict = base.classes.delivery_point_dict

    def search_data(self, id_type: int, id_country: int, id_source: int, point_name: str):
        search_record = session.query(
            self._delivery_point_dict.id).filter(
            self._delivery_point_dict.id_type == id_type,
            self._delivery_point_dict.id_country == id_country,
            self._delivery_point_dict.id_source == id_source,
            self._delivery_point_dict.point_name == point_name).all()

        match len(search_record):
            case 0:
                return self.insert_new_data(id_type=id_type,
                                            id_country=id_country,
                                            id_source=id_source,
                                            point_name=point_name)
            case 1:
                return search_record[0][0]
            case _:
                raise IndexError(f'there are more than one record in delivery_point_dict with fields:'
                                 f'\nid_type = {id_type}'
                                 f'\nid_country = {id_country}'
                                 f'\nid_source = {id_source}'
                                 f'\npoint_name = {point_name}')

    def insert_new_data(self, id_type: int, id_country: int, id_source: int, point_name: str):
        insert_statement = insert(self._delivery_point_dict, bind=engine).values(
            id_type=id_type,
            id_country=id_country,
            id_source=id_source,
            point_name=point_name,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            index_elements=['point_name', 'id_type'],
            set_={'update_time': insert_statement.excluded.update_time},
        )
        inserted_row = session.execute(update_statement)
        session.commit()
        return inserted_row.inserted_primary_key[0]
