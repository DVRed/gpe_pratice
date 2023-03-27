"""
Разработчик: А.Гладыщев, А.Афанасьев, Д.Девяткин
Описание:
Этот файл производит загрузку данных, полученных в результате парсинга
в БД analytics_base.
Входные данные:
source: источник данных
delivery_point: пункт сдачи-приемки газа
unit: единица измерения данных
from_company: id компании со одной стороны компрессорной станции
to_company: id компании со другой стороны компрессорной станции
from_country: id страны со одной стороны компрессорной станции
to_country: id страны со другой стороны компрессорной станции
flow_type: id типа потока (nomination, physical flow)
curve_name: название кривой
Выходные данные:
Нет
"""

import sys
from db_initial_connection import base, session
from curves import Curves
import pandas as pd
import numpy as np
from snam_parser import SnamScrapping
import datetime
import tqdm
import os
from dotenv import load_dotenv

load_dotenv('/srv/sstd/.env')


class SnamLoader:

    def __init__(self):

        # curves_dict
        self.table = base.classes.flow_curves
        self.flows_unit_dict = {'entry': 'mln m3',
                                'exit': 'mln m3',
                                'LDZ_demand': 'mln m3',
                                'domestic_production': 'mln m3',
                                'industrial_demand': 'mln m3',
                                'withdrawal': 'mln m3'}
        self.curves_class = Curves()

    def snam_validate(self, source: str, point_name: str, curve_name: str, unit: str = None):

        search_source = session.query(
            base.classes.source_dict.id).filter(
            base.classes.source_dict.source_name == source).all()

        if len(search_source) != 1:
            raise IndexError(source)
        else:
            id_source = search_source[0]

        search_dp = session.query(
            base.classes.delivery_point_dict.id).filter(
            base.classes.delivery_point_dict.point_name == point_name).all()

        if len(search_dp) != 1:
            raise IndexError(point_name)
        else:
            id_dp = search_dp[0]

        if unit is None:
            unit = self.flows_unit_dict[curve_name]
        else:
            pass

        search_unit = session.query(
            base.classes.units_dict.id).filter(
            base.classes.units_dict.unit_name == unit).all()

        if len(search_unit) != 1:
            raise IndexError(unit)
        else:
            id_unit = search_unit[0]

        check_exists = session.query(self.table.id).filter(
            self.table.id_source == id_source[0],
            self.table.id_point == id_dp[0],
            self.table.id_unit == id_unit[0],
            self.table.curve_name == curve_name).all()

        if len(check_exists) != 1:
            raise IndexError('no such curve in flow_curves')
        else:
            id_flow_curve = check_exists[0][0]

        id_curve = session.query(
            base.classes.curves_dict.id).filter(
            base.classes.curves_dict.id_flow_curves == id_flow_curve).all()

        if len(id_curve) != 1:
            raise IndexError('no such curve in curves_dict')
        else:
            return id_curve[0][0]

    def insert_snam(self, source: str = 'Snam', df_snam: pd.DataFrame = None, unit: str = None):
        data = df_snam
        data['id_curve'] = data.apply(
            lambda x: self.snam_validate(
                source, x.point_name, x.type, unit), axis=1)
        data = data[['date', 'id_curve', 'value']].to_numpy()
        for row in tqdm.tqdm(data):
            try:
                if row[2] == '-':
                    self.curves_class.insert_new_data(
                        row[1], row[0], np.float64(0))
                else:
                    self.curves_class.insert_new_data(
                        row[1], row[0], np.float64(row[2]))
            except:
                sys.exit(1)
        return 'ok'


if __name__ == '__main__':
    # initial set-up
    _proxy = {
        'http': os.environ.get('INTERNAL_HTTP_PROXY'),
        'https': os.environ.get('INTERNAL_HTTPS_PROXY')
    }
    # download headers template
    _headers = {
        "GET": "pubblicazioni-smart-public/pubblicazioni-smart/document/get?d=JPUBB007/YYYY/datiOperativi_YEARMMDD_EN.xlsx&fileName=datiOperativi_YEARMMDD_EN&user_key=b09ce7de6fbc08ffddd91c401122c7fd HTTP/1.1",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Host": "jarvis-1389-public.api-corp.c01p.snam.it",
        "Origin": "https://jarvis.snam.it",
        "Pragma": "no-cache",
        "Referer": "https://jarvis.snam.it/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
        "X-jarvis-multiCompany": "SNM",
        "X-jarvis-transactionId": "7b393f35-7961-499b-ace7-3faa86de3699",
        "accept": "application/octet-stream",
        "sec-ch-ua": '"Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
    }
    # main page url
    _start_url = "https://jarvis.snam.it/public-data"
    _confirm_url = ""
    # download url template
    _download_url = \
        "https://jarvis-1389-public.api-corp.c01p.snam.it/" \
        "pubblicazioni-smart-public/pubblicazioni-smart/document/" \
        "get?d=JPUBB007/YYYY/datiOperativi_YEARMMDD_EN.xlsx&" \
        "fileName=datiOperativi_YEARMMDD_EN&" \
        "user_key=ff546003775b66f4ff2c52b2702b5382"

    snam_scrapper = SnamScrapping(_proxy,
                                  _headers,
                                  _start_url,
                                  _download_url)
    snam_scrapper.test_connection()
    num_updated_days: int = 5
    start_parse = datetime.datetime.today() - \
                  datetime.timedelta(num_updated_days)
    end_parse = datetime.datetime.today()
    result = snam_scrapper.parse(start_parse, end_parse)
    result["value"] = result["value"] * 0.0000971817
    SnamLoader().insert_snam(df_snam=result)
