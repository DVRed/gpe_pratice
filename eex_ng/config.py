"""
Разработчик: А.Гладыщев, А.Афанасьев, Д.Девяткин
Описание:
Этот файл содержит переменные, необходимые для поключения к БД analytics_base.
Входные данные:
Нет
Выходные данные:
Нет
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv('/srv/sstd/.env')


@dataclass(slots=True, frozen=True)
class DBConfig:
    DBMS: str
    DRIVER: str
    HOSTNAME: str
    DATABASE: str
    USERNAME: str
    PASSWORD: str
    config_name: str


class DBConfigInstance:

    def __init__(self, in_db_config: DBConfig):
        self._DBMS = in_db_config.DBMS
        self._DRIVER = in_db_config.DRIVER
        self._HOSTNAME = in_db_config.HOSTNAME
        self._DATABASE = in_db_config.DATABASE
        self._USERNAME = in_db_config.USERNAME
        self._PASSWORD = in_db_config.PASSWORD
        self.config_name = in_db_config.config_name
        self.DB_URI = self._set_db_url()

    def _set_db_url(self):
        db_url = '{}+{}://{}:{}@{}/{}'.format(
            self._DBMS, self._DRIVER, self._USERNAME,
            self._PASSWORD, self._HOSTNAME, self._DATABASE
        )
        return db_url

    def __repr__(self):
        print_result = (f"{self.config_name}("
                        f"DBMS='{self._DBMS}', "
                        f"DRIVER='{self._DRIVER}', "
                        f"USERNAME='{self._USERNAME}', "
                        f"PASSWORD='{self._PASSWORD}', "
                        f"HOSTNAME='{self._HOSTNAME}', "
                        f"DATABASE='{self._DATABASE}'"
                        ")")
        return print_result


# конфиг для подключения к аналитической базе рыночной информации analytics_base
_ANALYTICS_BASE_DB_CONFIG = DBConfig(DBMS=os.environ.get('DATABASE_DIALECT'),
                                     DRIVER=os.environ.get('DATABASE_DRIVER'),
                                     HOSTNAME=os.environ.get('SERVER_HOST_NAME'),
                                     DATABASE=os.environ.get('ANALYTICS_DATABASE_NAME'),
                                     USERNAME=os.environ.get('ANALYTICS_DATABASE_USERNAME'),
                                     PASSWORD=os.environ.get('ANALYTICS_DATABASE_PASSWORD'),
                                     config_name='ANALYTICS_BASE_DB_CONFIG')
# для непосредственного подключения к базе необходимо использовать
# эту переменную, потому что в ней есть атрибут DB_URL, подаваемый в
# SQLAlchemy при создании движка
ANALYTICS_BASE_DB_CONFIG = DBConfigInstance(_ANALYTICS_BASE_DB_CONFIG)
