from abc import ABC, abstractmethod

import pandas as pd


class Period(ABC):

    @staticmethod
    @abstractmethod
    def get_products(date) -> str:
        """
        converts date into products
        """
        pass

    def print(self):
        return self.__class__.__name__


class Day(Period):
    symbol_egsi = '_DAILY'

    @staticmethod
    def get_products(date) -> str:
        dayli = date.strftime("%d/%m/%Y")
        return f"{dayli}".upper()


class Weekend(Period):
    symbol_egsi = 'W_WEEK'

    @staticmethod
    def get_products(date) -> str:
        weekend = date.strftime("%d/%m")
        return f"WkEnd {weekend}".upper()


class Week(Period):
    symbol_egsi = 'F_WEEK'

    @staticmethod
    def get_products(date) -> str:
        week_str = date.strftime("%W")
        return f"Week {week_str}/{str(date.year)[2:]}".upper()

        # 4 digit year
        # week = week.strftime("%W/%y")
        # del_start = f"Week {week}"


class Month(Period):
    symbol = 'BM'
    symbol_egsi = 'FM'

    @staticmethod
    def get_products(date) -> str:
        month_str = date.strftime("%b")
        return f"{month_str}/{str(date.year)[2:]}".upper()

        # 4 digit year
        # month = month.strftime("%b/%Y")
        # del_start= f"{month}"


class Quarter(Period):
    symbol = 'BQ'
    symbol_egsi = 'FQ'

    @staticmethod
    def get_products(date) -> str:
        year = date.strftime("%Y")
        quarter = pd.Timestamp(date).quarter
        return f"{quarter}/{str(year)[2:]}".upper()

        # 4 digit year
        # del_start = f"{quarter}/{year}"


class Season(Period):
    symbol = 'BS'
    symbol_egsi = 'FS'

    @staticmethod
    def get_products(date) -> str:
        season = 'SUM' if date.month in range(3, 9) else 'WIN'
        return f"{season}-{str(date.year)[2:]}".upper()

        # 4 digit year
        # del_start = f"{season}-{year.year}"


class Year(Period):
    symbol = 'BY'
    symbol_egsi = 'FY'

    @staticmethod
    def get_products(date) -> str:
        year = date.strftime("%Y")
        return f"Cal-{str(year)[2:]}".upper()

        # 4 digit year
        # return f"Cal-{year}"
