from __future__ import annotations

import numpy as np
from pandas import NaT, DataFrame
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.dialects.postgresql import insert
from exxeta_settings import CURRENCIES, DELIVERY_POINT_TYPES, UNITS
from psycopg2.extensions import register_adapter, AsIs
from config import DBConfigInstance, ANALYTICS_BASE_DB_CONFIG


# действия по согласованию числовых форматов
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)


class DBConnector:
    """
    Класс подключения к БД аналитической информации через SQLAlchemy
    """

    def __init__(self, config: DBConfigInstance = ANALYTICS_BASE_DB_CONFIG):
        self._config = config.DB_URI
        self.engine = None
        self.session = None
        self.base = None

    def create_engine(self):
        """Создает подключение к БД

        Returns:
            Engine: объект подключения engine
        """
        self.engine = create_engine(
            self._config, echo=False)
        return self.engine

    def create_session(self):
        """Создает сессию

        Returns:
            Session: объект сессии
        """
        self.session = Session(bind=self.engine)
        return self.session

    def connect_to_base(self):
        """Отображает схему БД

        Returns:
            AutomapBase: БД
        """
        self.base = automap_base()
        self.create_engine()
        self.base.prepare(self.engine, reflect=True)
        return self.base


class DBLoader:
    """Базовый класс для загрузки данных в БД

    Attributes:
        in_base (AutomapBase): БД
        in_session (Session): объект сессии в БД
    """

    def __init__(self, in_base, in_session):
        self.base = in_base
        self.session = in_session

    def check_item(self, in_value: dict[str, str], in_table: str, in_check_column_name: list | None) -> int | None:
        """Проверяет наличие записи в таблице.

        Если находит запись value в таблице table, то возвращает id.

        Args:
            in_value: dict - запись вида {'название_столбца': 'значение'}
            in_table: str - наименование таблицы, в которой производится поиск
            in_check_column_name: list | None
        Returns:
            id: int - идентификатор записи или None (если такой записи нет)
        """
        result_id = None
        try:
            text_string = ' AND '.join([f"{key} = '{in_value[key]}'" for key in
                                        (in_value if in_check_column_name is None else in_check_column_name)])
            text_string = text_string.replace("= 'None'", "is Null") \
                .replace("= 'NaT'", "is Null") \
                .replace("= 'nan'", "is Null") \
                .replace("= 'NaN'", "is Null")
            result = self.session.query(in_table).filter(text(text_string)).one()
            result_id = result.id
        except NoResultFound:
            # print(f"{datetime.now()}| Value {in_value} is not '{in_table}'")
            pass
        return result_id

    def get_id_for_new_item(self, in_table):
        """Генерирует id для новой записи в таблице

        Args:
            in_table: str - наименование таблицы, в которую будет записываться значение
        Returns:
            id: int - идентификатор новой записи
        """
        table_name = getattr(self, 'table_name', None)
        if table_name is None:
            table_name = in_table.__name__
        # используем последовательности для непрерывного нумерования

        # text_string = (f"CREATE SEQUENCE IF NOT EXISTS \"{table_name}_id_seq\";\n"
        #                f"SELECT setval('{table_name}_id_seq', "
        #                f"COALESCE((SELECT MAX(id)+1 FROM {table_name}), 1), FALSE);")
        # self.session.execute(text_string)

        value_id = self.session.execute(f"SELECT nextval('{table_name}_id_seq');").scalar()
        self.session.commit()
        return value_id

    @staticmethod
    def get_current_datetime():
        """Получаем текущие дату и время для записи в `update_time`"""
        return datetime.now()

    def set_base_columns_value(self, in_value: dict, in_value_id: int):
        """Задает значения колонкам id и update_time"""
        return in_value.update({'id': in_value_id, 'update_time': self.get_current_datetime()})

    def insert_item(self, in_value: dict, in_table, in_check_column_name: list | None = None) -> int:
        """Записывает значения в таблицу

        Args:
            in_value: строка (словарь) с данными для записи
            in_table: таблица для записи
            in_check_column_name
        Returns:
            value_id (int) - id добавленной записи
        """
        # если записи еще нет в таблице, то check_item вернет None
        value_id = self.check_item(in_value, in_table, in_check_column_name)
        if value_id is None:
            # value_id = self.get_id_for_new_item(in_table) # проверим, будут ли конфликты
            # new_item - теперь экземпляр класса, соответствующего таблице для записи
            new_item = in_table()
            # in_value.update({'id': value_id, 'update_time': self.get_current_datetime()}) # проверим, будут ли конфликты
            in_value.update({'update_time': self.get_current_datetime()})  # проверим, будут ли конфликты
            # [setattr(new_item, key, in_value[key]) for key in in_value]
            # self.session.add(new_item) # старая версия с полной блокировкой таблицы
            v_inserted_values = insert(in_table).values(in_value)
            try:  # проверим, будут ли конфликты
                self.session.execute(v_inserted_values)
                self.session.commit()  # проверим, будут ли конфликты
            except:  # проверим, будут ли конфликты
                self.session.rollback()  # проверим, будут ли конфликты
            value_id = self.check_item(in_value, in_table, in_check_column_name)  # проверим, будут ли конфликты
        return value_id


class DBLoaderDeliveryPointType(DBLoader):
    """Класс для загрузки данных в таблицу `delivery_point_types_dict`

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (str): колонка, по которой будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'delivery_point_types_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = 'point_type'

    def _get_table(self):
        """Позволяет достать объект-таблицу из схемы БД"""
        return getattr(self.base.classes, self.table_name)

    def insert_item(self, in_value: str, in_table=None) -> int:
        """Записывает значения в таблицу

        Args:
            in_value: строка с данными для записи
            in_table: таблица для записи
        Returns:
            value_id (int) - id добавленной записи
        """
        in_table = self.table
        in_value = {self.check_column_name: in_value}
        value_id = DBLoader(self.base, self.session).insert_item(in_value, in_table, [self.check_column_name])
        return value_id


class DBLoaderDeliveryPoint(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `delivery_point_dict`

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (str): колонка, по которой будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'delivery_point_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = 'point_name'

    def insert_item(self, in_value, in_table=None) -> int:
        """Записывает значения в таблицу

        Args:
            in_value: строка с данными для записи
            in_table: таблица для записи
        Returns:
            value_id (int) - id добавленной записи
        """
        in_table = self.table
        point_type = DELIVERY_POINT_TYPES[in_value['point_type']]
        in_value = {
            self.check_column_name: in_value['point_name'],
            'id_type': DBLoaderDeliveryPointType(self.base, self.session).insert_item(point_type)
        }
        value_id = DBLoader(self.base, self.session).insert_item(in_value, in_table, [self.check_column_name])
        return value_id


class DBLoaderCurrency(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `currencies_dict`.

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (str): колонка, по которой будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'currencies_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = 'currency_code'


class DBLoaderUnit(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `units_dict`.

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (str): колонка, по которой будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'units_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = 'unit_name'


class DBLoaderMarket(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `markets_dict`.

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (str): колонка, по которой будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'markets_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = 'market_name'


class DBLoaderProductType(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `product_types_dict`.

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (str): колонка, по которой будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'product_types_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = 'product_type'


class DBLoaderProducts(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `products_dict`.

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (list[str]): колонки, по которым будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'products_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = [
            'id_delivery_point', 'id_currency', 'id_unit', 'id_market',
            'id_product_type', 'beg_date', 'end_date', 'code'
        ]

    def insert_item(
            self, in_delivery_point: str, in_currency: str, in_unit: str, in_market: str, in_product_type: str,
            in_beg_date: datetime, in_end_date: datetime, in_product_name: str, in_comment: str, in_table=None
    ) -> int:
        """Записывает значения в таблицу

        Args:
            in_delivery_point: наименование пункта поставки
            in_currency: наименование валюты платежа по сделке
            in_unit: наименование единицы измерения коммодити
            in_market: наименование коммодити ('Natural Gas', 'Coal' и тд)
            in_product_type: наименование типа продукта ('Month', 'Quarter' и тд)
            in_beg_date: дата и время начала поставки
            in_end_date: дата и время окончания поставки
            in_product_name: код продукта ('DA', 'DEC21', '2024', 'Q1/27' и тд)
            in_comment: комментарий, включающий часы поставки и тип периода поставки. Например: TODO
                        'Delivery for 2.0 hours (Spot)'. Для всех коммодити, которые не относятся
                        к 'Natural Gas', комментарий определяется в теле функции и выглядит так:
                        'Incorrect delivery period'. Такоя формулировка обусловлена тем, что
                        при указании дат начала и конца поставки для негазовых коммодити (и СПГ)
                        не учитывается специфика их поставки по причине отсутствия необходимости
                        в этом на текущий момент (Q2 2022).
            in_table: таблица для записи
        Returns:
            value_id (int) - id добавленной записи
        """
        in_table = self.table
        params = (self.base, self.session)
        value = {
            'id_delivery_point': DBLoaderDeliveryPoint(*params).insert_item(in_delivery_point),
            'id_currency': DBLoaderCurrency(*params).insert_item(in_currency),
            'id_unit': DBLoaderUnit(*params).insert_item(in_unit),
            'id_market': DBLoaderMarket(*params).insert_item(in_market),
            'id_product_type': DBLoaderProductType(*params).insert_item(in_product_type),
            'code': in_product_name,
            'comment': in_comment if in_market == 'Natural Gas' else 'Incorrect delivery period'  # TODO
        }
        if in_beg_date is not NaT:
            value.update({'beg_date': in_beg_date, 'end_date': in_end_date})
        value_id = DBLoader(*params).insert_item(value, in_table, self.check_column_name)
        return value_id


class DBLoaderInstrumentType(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `instrument_types_dict`

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (str): колонка, по которой будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'instrument_types_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = 'instrument_type'


class DBLoaderInstrument(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `instruments_dict`

    Attributes:
        table: объект-таблица из базы данных
        check_column_name (list[str]): колонки, по которым будет вестись поиск похожих записей
    """
    # соответствует названию таблицы в БД
    table_name = 'instruments_dict'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()
        self.check_column_name = ['id_product_1', 'id_product_2', 'id_instrument_type']

    def insert_item(self, in_product_1: dict, in_product_2: dict | None, in_instrument_type: str, in_table=None) -> int:
        """Записывает значения в таблицу

        Args:
            in_product_1: словарь со значениями для вставки (проверки) в таблицу(-е) с продуктами
            in_product_2: словарь со значениями для вставки (проверки) в таблицу(-е) с продуктами или None,
                          если в сделке фигурировал только один продукт (in_instrument_type == 'Single').
            in_instrument_type: наименование типа инструмента ('Single' или 'Spread')
            in_table: таблица для записи
        Returns:
            value_id (int) - id добавленной записи
        """
        in_table = self.table
        params = (self.base, self.session)
        value = {
            'id_product_1': DBLoaderProducts(*params).insert_item(**in_product_1),
            'id_product_2': DBLoaderProducts(*params).insert_item(**in_product_2) if in_product_2 is not None else None,
            'id_instrument_type': DBLoaderInstrumentType(*params).insert_item(in_instrument_type)
        }
        value_id = DBLoader(*params).insert_item(value, in_table, self.check_column_name)
        return value_id


class Deal:
    """Класс для преобразования словаря с информацией по конкретной сделке.

    Экземпляр класса формирует необходимые для загрузки в БД поля, получив на вход информацию о сделке.

    Attributes:
        in_data_row: словарь со всей информацией по конкретной сделке
    """

    def __init__(self, in_data_row: dict):
        # для того, чтобы свободно пользоваться отдельными значениями (например, чтобы узнать цену,
        # обратившись к соответствующему атрибуту по имени: self.price), описывающими сделки
        self.__dict__.update(in_data_row)
        # для более удобной загрузки продуктов и инструмента впоследствии
        self.product_1 = self._get_product()
        self.product_2 = self._get_product(order=2)

    @staticmethod
    def _get_specific_instrument_name(in_instrument_name: str, in_specific: str) -> str:
        """Добавляет преффиксы к исходному коду инструмента.

        У некоторых продуктов в названии присутствуют различные дополнения - в особенности у "электрических".
        Например, 'Base DEC21'. Здесь основной код продукта - 'DEC21', а специфичное дополнение - 'Base'.
        Это дополнение подразумевает определенную особенность. Поэтому такие продукты лучше хранить отдельно
        от 'DEC21', чтобы избежать возможных ошибок.

        То есть, если у продукта присутствует специфичное значение, оно будет объединено с кодом продукта
        в теле функции. В противном случае функция вернет код продукта.

        Args:
            in_instrument_name: "чистое" наименование продукта ('DEC21', 'Q4/25', 'Sum 2025' и тд)
            in_specific: специфичная часть продукта ('Base', 'Peak', 'Offpeak', 'WD 5+6' и тд)
        Returns:
            str - конечный вид кода продукта для загрузки в базу
        """
        return in_specific + in_instrument_name if in_specific != '' else in_instrument_name

    def _get_product(self, order: int = 1) -> dict:
        """Собирает данные о продукте в единый словарь

        Args:
            order: порядковый номер продукта. Может принимать значения 1 (по умолчанию) и 2. Если
                   order=2, то у сделки instrument_type = 'Spread' и в теле функции меняется
                   наполнение словаря с данными.
        Returns:
            product (dict) - словарь с данными по продукту
        """
        assert order in (1, 2), "Аргумент 'order' должен быть равен 1 или 2"
        comment = f"Delivery for {int(self.delivery_hours_1)} hours ({self.delivery_period_type})"
        product = {
            'in_delivery_point': {'point_name': self.delivery_point_1, 'point_type': self.commodity_type},
            'in_currency': self.currency, 'in_unit': self.unit,
            'in_market': self.commodity_type, 'in_product_type': self.product_type,
            'in_beg_date': self.delivery_start_1,
            'in_end_date': self.delivery_end_1,
            'in_product_name': self._get_specific_instrument_name(self.instrument_1, self.specific),
            'in_comment': comment
        }
        # TODO: изменить маркировку пропущенных пунктов поставки и инструментов при парсинге
        if order == 2:
            if (isinstance(self.delivery_point_2, float) or self.delivery_point_2 == 'nan') and (
                    isinstance(self.instrument_2, float) or (self.instrument_2 == 'nan')):
                product = None
            # если в спецификации сделки указан второй пункт поставки и не указан второй инструмент
            # то необходимо в словаре изменить подтягиваемый пункт поставки на второй
            # RO CEGH VTP PEGAS/TTF Hi Cal 51.6 PEGAS Month JUN20 - примерная спецификация такой сделки
            # (эти спецификации хранятся в market_deals.deal_contract)
            elif not (isinstance(self.delivery_point_2, float) or self.delivery_point_2 == 'nan') and (
                    isinstance(self.instrument_2, float) or (self.instrument_2 == 'nan')):
                product.update(
                    {'in_delivery_point': {'point_name': self.delivery_point_2, 'point_type': self.commodity_type},
                     'in_currency': CURRENCIES[self.delivery_point_2],
                     'in_unit': UNITS[self.delivery_point_2]})
            # если в спецификации сделки не указан второй пункт поставки, но указан второй инструмент
            # то необходимо в словаре изменить информацию о поставках (указать для вторго инструмента - суффикс "_2")
            # RO TTF Hi Cal 51.6 PEGAS Quarter1 Q3/20/Q4/20 - примерная спецификация такой сделки
            # (эти спецификации хранятся в market_deals.deal_contract)
            elif (isinstance(self.delivery_point_2, float) or self.delivery_point_2 == 'nan') and not (
                    isinstance(self.instrument_2, float) or (self.instrument_2 == 'nan')):
                comment = f"Delivery for {np.round(self.delivery_hours_2)} hours ({self.delivery_period_type})"
                product.update({'in_beg_date': self.delivery_start_2, 'in_end_date': self.delivery_end_2,
                                'in_product_name': self._get_specific_instrument_name(self.instrument_2, self.specific),
                                'in_comment': comment})
            else:
                raise TypeError(f'unknown type of dp {type(self.delivery_point_2)} or instr {type(self.instrument_2)}')

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


class DBLoaderDeals(DBLoaderDeliveryPointType):
    """Класс для загрузки данных в таблицу `market_deals`

    Attributes:
        table: объект-таблица из базы данных
    """
    # соответствует названию таблицы в БД
    table_name = 'market_deals'

    def __init__(self, in_base, in_session):
        super().__init__(in_base, in_session)
        self.table = self._get_table()

    def _get_deals_value(self, deal: Deal) -> dict:
        params = (self.base, self.session)
        if deal.delivery_point_1 in ['PEG', 'Peg Nord', 'AOC']:
            deal.volume /= 24
        elif deal.delivery_point_1 in ['NBP', 'IBP', 'ZEE']:
            deal.volume = deal.volume * 1000 / 24
        value = {
            'id_instrument': DBLoaderInstrument(*params).insert_item(deal.product_1, deal.product_2,
                                                                     deal.instrument_type),
            'id_market': DBLoaderMarket(*params).insert_item(deal.commodity_type),
            'deal_datetime': deal.date,
            'deal_contract': deal.contract,
            'volume': deal.volume,
            'price': deal.price,
            'venue': deal.venue
        }
        return value

    def insert_item(self, in_deal: Deal, in_table=None) -> int:
        """Записывает значения в таблицу

        Args:
            in_deal: экземпляр класса Deal - отдельная сделка
            in_table: таблица для записи
        Returns:
            value_id (int) - id добавленной записи
        """
        in_table = self.table
        params = (self.base, self.session)
        value = self._get_deals_value(in_deal)
        value_id = DBLoader(*params).insert_item(value, in_table)
        return value_id

    def bulk_insert_items(self, in_deals_df: DataFrame, n_rows=None):
        """Записывает значения в таблицу

        Отличается от `insert_item` тем, что загружает все сделки, хранимые в датафрейме `in_deals_df` разом
        либо партиями по `n_rows` строк.

        Args:
            in_deals_df: датафрейм со сделками для загрузки
            n_rows: количество загружаемых за раз сделок в БД
        """
        deal_list = []
        for deal_index, row in in_deals_df.iterrows():
            deal = Deal(row.to_dict())
            value = self._get_deals_value(deal)
            value.update({'update_time': self.get_current_datetime()})
            deal_list.append(value)
        # так как pandas не особо запотится о сохранности последовательности id, то
        # перед каждой загрузкой партии сделок сиквенции задается верное значение
        text_string = (f"CREATE SEQUENCE IF NOT EXISTS \"{self.table_name}_id_seq\";\n"
                       f"SELECT setval('{self.table_name}_id_seq', "
                       f"COALESCE((SELECT MAX(id)+1 FROM {self.table_name}), 1), FALSE);")
        self.session.execute(text_string)

        data_frame = DataFrame(deal_list)
        data_frame.to_sql(self.table_name, DBConnector().create_engine(), chunksize=n_rows, if_exists='append',
                          index=False)

