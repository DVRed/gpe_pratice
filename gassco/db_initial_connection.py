from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from db_config import ANALYTICS_BASE_DB_CONFIG


class InitialToConnection:
    """
    Класс подключения к БД по сделкам через SQLАlchemy
    """

    def __init__(self):
        self.__DB_URI = ANALYTICS_BASE_DB_CONFIG.DB_URI
        self.engine = create_engine(self.__DB_URI)

    def get_connect(self) -> classmethod:
        """
        Получение соединения с БД
        :return: объект соединения
        """
        out_connect = self.engine.connect()
        return out_connect

    def get_initial_base(self) -> classmethod:
        """
        Получение схемы БД через автомапинг
        :return: объект схемы БД
        """
        out_base = automap_base()
        out_base.prepare(self.engine,
                         reflect=True)
        return out_base

    def get_initial_meta_data(self) -> classmethod:
        """
        Получение метаданных БД
        :return: объект метаданных
        """
        out_meta_data = MetaData(bind=self.engine)
        return out_meta_data

    def get_session(self) -> Session:
        """
        создание сессии БД
        :return: объект сессии
        """
        out_session = Session(bind=self.engine)
        return out_session


# осуществляем подключение к БД
initial_to_connect = InitialToConnection()
engine = initial_to_connect.engine
meta_data = initial_to_connect.get_initial_meta_data()
base = initial_to_connect.get_initial_base()
session = initial_to_connect.get_session()

