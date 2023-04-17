import sys
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from loguru import logger

from config import DatabaseName, DatabaseConfig, _catch_bad_config


logger.remove(0)
logger.add(
    sys.stderr,
    format="{time:MMM D, YYYY HH:mm:ss} | {level} | {message}",
    colorize=True,
    diagnose=True
)


class DatabaseConnection:
    """Creates database connection by database name `db_name`.

    Database connection is `Engine` instance of SQLAlchemy package.
    Connection is used when reading, inserting, deleting and updating
    data in database.
    """

    def __init__(self, *, db_name: DatabaseName) -> None:
        """Build new `DatabaseConnection` instance.

        Args:
            db_name: name of database to create database connection
        """
        self._db_name = db_name
        self._config: str = self.create_db_connection_config()
        logger.info("Created confing for {database!r}".format(database=self._db_name))
        self._connection: Optional[Engine] = None

    def create_db_connection_config(self) -> str:
        """Creates database configuration by database name `db_name`"""
        config = DatabaseConfig(db_name=self._db_name)
        return config.get_config()

    @_catch_bad_config
    def get_connection(self) -> Engine:
        """Returns database connection"""
        if not self._check_if_connection_exists():
            connection = create_engine(url=self._config)
            self._save_connection(connection)
            logger.info("Created connection to {database!r}".format(database=self._db_name))
        return self._connection

    def _check_if_connection_exists(self) -> bool:
        """Returns True if database connection exists"""
        return False if self._connection is None else True

    def _save_connection(self, connection: Engine) -> None:
        """Set database engine instance"""
        self._connection = connection

    def close(self) -> None:
        """Closes connection if exists"""
        if self._check_if_connection_exists():
            self._connection.dispose()
            logger.info("Closed connection to {database!r}".format(database=self._db_name))
            self._reset_connection()

    def _reset_connection(self) -> None:
        """Resets connection if it was disposed"""
        self._connection = None

    def get_session(self) -> Session:
        """create db session"""
        out_session = Session(bind=self._connection)
        return out_session

    def get_initial_base(self) -> classmethod:
        """create db schema via automap"""
        out_base = automap_base()
        out_base.prepare(self._connection,
                         reflect=True)
        return out_base


# connection
connect = DatabaseConnection(db_name='analytics_base')
engine = connect.get_connection()
session = connect.get_session()
base = connect.get_initial_base()
