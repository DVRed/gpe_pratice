from __future__ import annotations

import pandas as pd
from loguru import logger
from pandas import DataFrame
from sqlalchemy import create_engine, text
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from db_config import DBConfigInstance, ANALYTICS_BASE_DB_CONFIG


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
        logger.debug(f'creating engine with {self._config} config')
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
        base (AutomapBase): БД
        session (Session): объект сессии в БД
    """

    def __init__(self, in_base, in_session):
        self.base = in_base
        self.session = in_session


def execute_query_to_dataframe(query_text: str) -> DataFrame:
    """
    Connects to DB, executes query_text into pandas DataFrame
    """
    connector = DBConnector()
    base = connector.connect_to_base()
    session = connector.create_session()

    df = pd.read_sql_query(query_text, session.connection().connection)

    session.close()
    connector.engine.dispose()
    return df


if __name__ == '__main__':
    connector1 = DBConnector()
    base1 = connector1.connect_to_base()
    session1 = connector1.create_session()

    # for price_index, row in tqdm(a.iterrows(), ncols=100, total=a.shape[0],
    #                              desc='EEX prices loader', disable=False):
    #     curve = Price(row.to_dict())
    #     DBLoaderCurves(Base, session).insert_item(curve)
    res1 = session1.execute(text("SELECT * FROM unit_types"))
    print(res1)
    for row1 in res1:
        print(row1)

    dfr1 = pd.read_sql_query("SELECT * FROM unit_types", session1.connection().connection)

    i = 0

    session1.close()
    connector1.engine.dispose()
