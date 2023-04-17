import os
from typing import Literal, Mapping, Callable
from dotenv import load_dotenv

from sqlalchemy.exc import ArgumentError

load_dotenv('.env')

ENV_VAR_NAME_PREFIXES: Mapping['DatabaseName', str] = {
    'analytics_base': 'ANALYTICS',
    'analytics_base_test': 'ANALYTICS_TEST'
}
DatabaseName = Literal["analytics_base", "analytics_base_test"]


class NoSuchDatabase(Exception):
    """Raises if database does not exist"""
    ...


class BadConnectionConfig(Exception):
    """Raises if database configuration is not correct"""
    ...


class DatabaseConfig:
    """Creates database configuration by database name `db_name`.

    Database configuration is string that looks like 'dialect+driver://user:password@host/database'.
    Config is used when creating a database engine with SQLAlchemy.
    """

    def __init__(self, db_name: DatabaseName) -> None:
        """Build new `DatabaseConfig` instance.

        Args:
            db_name: name of database to create database configuration
        """
        self._db_name = db_name
        self._env_var_name_prefix: str = self._get_env_var_name_prefix()
        self._password = os.environ.get(f'{self._env_var_name_prefix}_DATABASE_PASSWORD')

    @property
    def source_name(self) -> str:
        """Returns database dialect from environmental variables"""
        return os.environ.get(f'DATABASE_DIALECT')

    @property
    def driver(self) -> str:
        """Returns database driver name from environmental variables"""
        return os.environ.get(f'DATABASE_DRIVER')

    @property
    def host(self) -> str:
        """Returns host name of the database from environmental variables"""
        return os.environ.get(f'SERVER_HOST_NAME')

    @property
    def db_name(self) -> str:
        """Returns name of database from environmental variables"""
        return self._db_name

    @property
    def user(self) -> str:
        """Returns database username from environmental variables"""
        return os.environ.get(f'{self._env_var_name_prefix}_DATABASE_USERNAME')

    def get_config(self) -> str:
        """Returns string with database configuration"""
        return (f"{self.source_name}+{self.driver}://"
                f"{self.user}:{self._password}@"
                f"{self.host}/{self.db_name}")

    def _get_env_var_name_prefix(self) -> str:
        """Returns prefix for environment variables.

        This prefix could be found in constant `ENV_VAR_NAME_PREFIXES`. It uses to associate
        full name of environment variable with the correct database value.

        For example:
        If db_name == 'deals' then prefix == 'DEALS'. In this case, we need to get
        all database configuration environment variables starting with 'DEALS' prefix,
        such as `DEALS_DATABASE_USERNAME`, `DEALS_DATABASE_PASSWORD` and others.

        Raises:
            NoSuchDatabase if `db_name` is not in `ENV_VAR_NAME_PREFIXES`
        """
        if self._check_if_database_exists():
            return ENV_VAR_NAME_PREFIXES[self._db_name]
        raise NoSuchDatabase("Database with such name had not been found in 'databases.config.ENV_VAR_NAME_PREFIXES'."
                             "If database exists, please add its name and prefix to "
                             "'databases.config.ENV_VAR_NAME_PREFIXES' and rewrite '.env' file with new values.")

    def _check_if_database_exists(self) -> bool:
        """Checks if `db_name` is in `ENV_VAR_NAME_PREFIXES`"""
        return True if self._db_name in ENV_VAR_NAME_PREFIXES else False


def _catch_bad_config(get_connection_func: Callable) -> Callable:
    """Catch `BadConnectionConfig` when creation the database configuration"""

    def try_get_connection(*args):
        try:
            return get_connection_func(*args)
        except ArgumentError as error:
            raise BadConnectionConfig(f'It is impossible to create connection with '
                                      f'incorrect connection config --> {error!r}')

    return try_get_connection
