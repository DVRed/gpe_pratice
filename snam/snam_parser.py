"""
Разработчик: Д.Девяткин, А.Афанасьев, А.Гладыщев, Д.Хомяков, М.Петров
Файл забирает данные с https://jarvis.snam.it/public-data
для последующей загрузки в БД analytics_base
Входные данные:
Нет
Выходные данные:
pandas DataFrame
"""

import pandas as pd
import requests
import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv('/srv/sstd/.env')


class SnamScrapping:

    def __init__(self,
                 proxy: dict,
                 headers: dict,
                 start_url: str,
                 download_url: str):
        """
        initialize and test connection to the given web-site
        :url: Snam web-site general URL without requests
        """
        # the following variables were simply copied from source_dict_uploading.py
        self._proxy = proxy
        self._headers = headers
        self._start_url = start_url
        self._download_url = download_url
        self._session = requests.Session()

    def test_connection(self):
        """

        """
        # TODO check connection, if response != 200, sys.exit
        _req = self._session.get(self._start_url,
                                 verify=False,
                                 proxies=self._proxy,
                                 allow_redirects=True)
        print(f"test connection {_req}")

    def transform_data(self, cur_date: datetime) -> tuple[str, dict]:
        """
        Need to transform the initial request
        :url: URL address (request) which initiates downloading of the CSV file
        """
        year = cur_date.year
        month = cur_date.strftime("%m")
        day = cur_date.strftime("%d")
        download_url = self._download_url
        download_headers = self._headers
        # a cycle of replacement of dates
        # URL transform depending on request type:
        download_url = download_url.replace('YYYY', str(year))
        download_url = download_url.replace("YEARMMDD", f"{year}{month}{day}")
        download_headers['GET'] = download_headers['GET'].replace(
            'YYYY', str(year)
        )
        download_headers['GET'] = download_headers['GET'].replace(
            "YEARMMDD", f"{year}{month}{day}"
        )
        return download_url, download_headers

    def parse(self,
              start_date: datetime.datetime,
              end_date: datetime.datetime) -> pd.DataFrame:
        """

        """
        _result_df = pd.DataFrame({
            "point_name": [],
            "date": [],
            "value": [],
            "type": []
        })
        _parsing_date = start_date
        while _parsing_date <= end_date:
            """
            
            """
            _success_request = False
            _attempts_count = 0
            url, headers = self.transform_data(_parsing_date)
            _raw_date = None
            _df_data = None
            while not (_success_request and _attempts_count < 5):
                try:
                    _response = requests.get(url,
                                             headers=headers,
                                             verify=False,
                                             proxies=self._proxy,
                                             allow_redirects=True)
                    _df_data = pd.read_excel(_response.content,
                                             sheet_name=4,
                                             header=10,
                                             index_col=0)
                    _raw_date = pd.read_excel(_response.content,
                                              sheet_name=4,
                                              header=7,
                                              index_col=0)
                    _success_request = True
                except:
                    _attempts_count += 1
                    time.sleep(100)
                if _attempts_count == 5:
                    print('Max retries exceeded')
            _raw_date = _raw_date.columns[0]
            _raw_date = _raw_date.split(':')[1][1:]
            _report_date = datetime.datetime.strptime(_raw_date, "%m/%d/%Y")

            _df_data = _df_data[["A.1", "B.1"]]
            _df_data = _df_data.reset_index()
            _df_data.columns = ["point_name", "value1", "value2"]
            _df_data.iloc()[_df_data[_df_data['point_name'] == "- Thermoelectric"].index[0], 1] = \
                _df_data.iloc()[_df_data[_df_data['point_name'] == "- Thermoelectric"].index[0], 2]
            _df_data["value"] = _df_data["value1"]
            _df_data = _df_data.drop("value1", axis=1).drop("value2", axis=1)
            _df_data = _df_data.dropna(subset=["point_name"])
            _df_data = _df_data.drop(_df_data.tail(18).index)
            _df_data = _df_data.reset_index(drop=True)
            _df_data = _df_data.drop(_df_data.index[[30]])
            _df_data = _df_data.drop(_df_data[_df_data["point_name"] == "INTAKE"].index)
            _df_data = _df_data.drop(_df_data[_df_data["point_name"] == "EXPORT"].index)
            _df_data = _df_data.drop(_df_data[_df_data["point_name"] == "DEMAND"].index)
            _df_data = _df_data.drop(_df_data[_df_data["point_name"] == "DEMAND (1)"].index)
            _df_data = _df_data.drop(
                _df_data[_df_data["point_name"] == "Other storages (Edison Stoccaggio, IGS) (4)"].index)
            _df_data = _df_data.drop(_df_data[_df_data["point_name"] == "Shippers imbalance from nominations(3)"].index)

            _df_data = _df_data.replace({"point_name": {
                "- Tarvisio": "Arnoldstein",
                "- LNG Cavarzere": "Porto Levante LNG Terminal",
                "- LNG Livorno": "FSRU OLT Offshore LNG Toscana",
                "- LNG Panigaglia": "Panigaglia LNG Terminal",
                "- Passo Gries": "Passo Gries",
                "- Mazara del Vallo": "Mazara del Vallo",
                "- Gela": "Gela",
                "- Gorizia": "Gorizia",
                "- Melendugno": "Melendugno",
                "- Bizzarone": "Bizzarone",
                "- San Marino": "San Marino",
                "- Final customers directly interconnected with the Transmission System ": "Final customers directly interconnected with the Transmission System",
                "- Deliveries to distribution systems and to other national TSOs interconnections": "Deliveries to distribution systems and to other national TSOs interconnections",
                "- Losses, Fuel consumption and unaccounted for gas (6)": "Losses, Fuel consumption and unaccounted for gas",
                "- National production": "National production",
                "- Stogit": "Stogit",
                "- Other storages (Edison Stoccaggio, IGS)": "Other storages (Edison Stoccaggio, IGS)",
                "- Thermoelectric": "Thermoelectric",
                "STORAGE SYSTEMS (-Injection, +Withdrawal)": "Storage systems (Injection, Withdrawal)",
                "STORAGE SYSTEMS (Injection, +Withdrawal)": "Storage systems (Injection, Withdrawal)",
                "Δ Line Pack (5)(6)": "Line Pack", }})
            _df_data["date"] = _report_date
            _df_data = _df_data.reset_index(drop=True)
            _df_data.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8], 'type'] = 'entry'
            _df_data.loc[[14, 15, 16, 17, 18, 19], 'type'] = 'exit'
            _df_data.loc[[9], 'type'] = 'domestic_production'
            _df_data.loc[[10, 23], 'type'] = 'industrial_demand'
            _df_data.loc[[11, 12, 13], 'type'] = 'LDZ_demand'
            _df_data.loc[[20, 21, 22], 'type'] = 'withdrawal'
            _result_df = pd.concat([_result_df, _df_data], ignore_index=True)
            _parsing_date += datetime.timedelta(1)
            time.sleep(5)

        return _result_df


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
    start_parse = datetime.datetime.today() - datetime.timedelta(1)
    end_parse = datetime.datetime.today()
    snam_scrapper.parse(start_parse, end_parse)
