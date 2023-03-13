from db_initial_connection import engine, base, session
from sqlalchemy import insert, and_
from datetime import datetime


class CurvesDict:

    def __init__(self):

        # curves_dict
        self.table = base.classes.curves_dict

    def next_table_id(self):

        result = session.query(self.table.id)
        id_list = []
        for row in result:
            id_list.append(row.id)
        if len(id_list) == 0:
            return 1
        else:
            return max(id_list) + 1

    def validate_curves(self, sector: str, time_period: str, id_curve: int):

        search_sector = session.query(
            base.classes.sector_dict.id
        ).filter(
            base.classes.sector_dict.sector_name == sector
        ).all()[0]
        if len(search_sector) == 1:
            id_sector = search_sector[0]
        else:
            print('{} not found'.format(sector))
            raise ValueError(sector)

        search_period = session.query(base.classes.sector_dict.id).filter(
            base.classes.time_periods_dict.period_code == time_period
        ).all()[0]
        if len(search_period) == 1:
            id_time_period = search_period[0]
        else:
            print('{} not found'.format(time_period))
            raise ValueError(time_period)

        if sector == 'forward prices':
            check_exists = session.query(
                self.table.id
            ).filter(
                and_(
                    self.table.id_sector == id_sector,
                    self.table.time_period == id_time_period,
                    self.table.id_prices_curves == id_curve
                )
            ).all()
        elif sector == 'lng':
            check_exists = session.query(
                self.table.id
            ).filter(
                and_(
                    self.table.id_sector == id_sector,
                    self.table.time_period == id_time_period,
                    self.table.id_lng_curves == id_curve
                )
            ).all()
        elif sector == 'ugs':
            check_exists = session.query(
                self.table.id
            ).filter(
                and_(
                    self.table.id_sector == id_sector,
                    self.table.time_period == id_time_period,
                    self.table.id_ugs_curves == id_curve
                )
            ).all()
        elif sector == 'flows':
            check_exists = session.query(
                self.table.id
            ).filter(
                and_(
                    self.table.id_sector == id_sector,
                    self.table.time_period == id_time_period,
                    self.table.id_flow_curves == id_curve
                )
            ).all()
        elif sector == 'macro values':
            check_exists = session.query(
                self.table.id
            ).filter(
                and_(
                    self.table.id_sector == id_sector,
                    self.table.time_period == id_time_period,
                    self.table.id_macro_curves == id_curve
                )
            ).all()
        elif sector == 'currency prices':
            check_exists = session.query(
                self.table.id
            ).filter(
                and_(
                    self.table.id_sector == id_sector,
                    self.table.time_period == id_time_period,
                    self.table.id_currency == id_curve
                )
            ).all()
        elif sector == 'volatility':
            check_exists = session.query(
                self.table.id
            ).filter(
                and_(
                    self.table.id_sector == id_sector,
                    self.table.time_period == id_time_period,
                    self.table.id_volatility_curves == id_curve
                )
            ).all()
        elif sector == 'weather':
            check_exists = session.query(
                self.table.id
            ).filter(
                and_(
                    self.table.id_sector == id_sector,
                    self.table.time_period == id_time_period,
                    self.table.id_weather_curves == id_curve
                )
            ).all()
        else:
            raise ValueError('unknown sector')

        if len(check_exists) == 0:
            return [id_sector, id_time_period]
        else:
            return 'curve already exists'

    def add_new_curve(self, sector: str, time_period: str, id_curve: int):

        validation = self.validate_curves(sector, time_period, id_curve)

        id_sector = validation[0]
        id_time_period = validation[1]
        if validation == 'curve already exists':
            pass
        else:
            if sector == 'forward prices':
                insert_command = insert(
                    self.table
                ).values(
                    id=self.next_table_id(),
                    id_sector=id_sector,
                    time_period=id_time_period,
                    id_prices_curves=id_curve,
                    update_time=datetime.today()
                )
                engine.connect().execute(insert_command)
            elif sector == 'lng':
                insert_command = insert(
                    self.table
                ).values(
                    id=self.next_table_id(),
                    id_sector=id_sector,
                    time_period=id_time_period,
                    id_lng_curves=id_curve,
                    update_time=datetime.today()
                )
                engine.connect().execute(insert_command)
            elif sector == 'ugs':
                insert_command = insert(
                    self.table
                ).values(
                    id=self.next_table_id(),
                    id_sector=id_sector,
                    time_period=id_time_period,
                    id_ugs_curves=id_curve,
                    update_time=datetime.today()
                )
                engine.connect().execute(insert_command)
            elif sector == 'flows':
                insert_command = insert(
                    self.table
                ).values(
                    id=self.next_table_id(),
                    id_sector=id_sector,
                    time_period=id_time_period,
                    id_flow_curves=id_curve,
                    update_time=datetime.today()
                )
                engine.connect().execute(insert_command)
            elif sector == 'macro values':
                insert_command = insert(
                    self.table
                ).values(
                    id=self.next_table_id(),
                    id_sector=id_sector,
                    time_period=id_time_period,
                    id_macro_curves=id_curve,
                    update_time=datetime.today()
                )
                engine.connect().execute(insert_command)
            elif sector == 'volatility':
                insert_command = insert(
                    self.table
                ).values(
                    id=self.next_table_id(),
                    id_sector=id_sector,
                    time_period=id_time_period,
                    id_volatility=id_curve,
                    update_time=datetime.today()
                )
                engine.connect().execute(insert_command)
            elif sector == 'currency prices':
                insert_command = insert(
                    self.table
                ).values(
                    id=self.next_table_id(),
                    id_sector=id_sector,
                    time_period=id_time_period,
                    id_currency=id_curve,
                    update_time=datetime.today()
                )
                engine.connect().execute(insert_command)
            elif sector == 'weather':
                insert_command = insert(
                    self.table
                ).values(
                    id=self.next_table_id(),
                    id_sector=id_sector,
                    time_period=id_time_period,
                    id_weather_curves=id_curve,
                    update_time=datetime.today()
                )
                engine.connect().execute(insert_command)
            else:
                raise ValueError('re-check sector')
        return 'successfull'

