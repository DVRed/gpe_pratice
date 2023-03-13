from db_initial_connection import engine, base, session
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
import numpy as np


class Curves:

    def __init__(self):

        # curves_dict
        self._curves_table = base.classes.curves

    @staticmethod
    def next_table_id():
        return session.execute('SELECT nextval('"'curves_id_seq'"');').scalar()

    def insert_new_data(self, id_curve, date: datetime, value: np.float64):
        insert_statement = insert(self._curves_table, bind=engine).values(
            id_curve=id_curve,
            date=date,
            value=value,
            update_time=datetime.today()
        )

        update_statement = insert_statement.on_conflict_do_update(
            # constraint=self._curves_table.primary_key,
            index_elements=['id_curve','date'],
            set_={'value': insert_statement.excluded.value,
                  'update_time': insert_statement.excluded.update_time},
        )
        session.execute(update_statement)
        session.commit()


