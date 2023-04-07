import re
import pandas as pd
import requests

from datetime import date, timedelta
from eex_ng.pandas_configurator import PandasConfigurator
from eex_loader.exxeta_settings import UNITS, CURRENCIES


class EexNaturalGasIndicesParser:
    """
    Class to parse EEX Natural Gas Indices
    https://www.eex.com/en/market-data/natural-gas/indices
    """

    url = 'https://webservice-eex.gvsi.com/query/json/getDaily/symbol/ontradeprice/offtradeprice/close/' \
          'onexchsingletradevolume/onexchtradevolumeeex/offexchtradevolumeeex/openinterest/tradedatetimegmt/'

    headers = {
        'Origin': 'https://www.eex.com',
        'Referer': 'https://www.eex.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
    }

    # '<symbol for request>': <hub_name>
    symbols_daily_hard = {
        # EEX Within-Day reference price
        '"#E.CEGH_WDRP"': 'Austria VTP',  # CEGH VTP
        '"#E.ETF_WDRP"': 'Denmark ETF',   # ETF
        '"#E.PEG_WDRP"': 'Peg Nord',      # PEG
        '"#E.PVB_WDRP"': 'PVB',
        '"#E.THE_WDRP"': 'THE VTP',       # THE
        '"#E.TTF_WDRP"': 'TTF',
        '"#E.ZTP_WDRP"': 'ZTP',
    }

    # '<symbol for request>': <hub_name>
    symbols_monthly_hard = {
        # EEX Monthly index
        '"$E.G8BM"': 'Austria VTP',  # CEGH VTP
        '"$E.G1BM"': 'Czech VTP',    # CZ VTP
        '"$E.GDBM"': 'Denmark ETF',  # ETF
        '"$E.G9BM"': 'NBP',
        '"$E.G5BM"': 'Peg Nord',     # PEG
        '"$E.GCBM"': 'PSV',
        '"$E.GEBM"': 'PVB',
        '"$E.G0BM"': 'THE VTP',      # THE
        '"$E.G3BM"': 'TTF',
        '"$E.GABM"': 'ZEE',
        '"$E.GBBM"': 'ZTP',
    }

    pc = PandasConfigurator()

    def __init__(self, end_date: date, start_date: date = None):
        if start_date is None:
            self.start_date = end_date - timedelta(days=10)
        else:
            self.start_date = start_date
        self.end_date = end_date
        if self.end_date - self.start_date < timedelta(0):
            raise Exception("End_date can't be before Start_date")

    def parse(self):

        """
        Make requests and form Pandas.DataFrame
        :return: DataFrame
        """

        self.make_requests(symbols=self.symbols_daily_hard,
                           products='DA',
                           product_type='Daily')

        self.make_requests(symbols=self.symbols_monthly_hard,
                           products='MA',
                           product_type='Monthly')

        self.pc.df = self.pc.df.dropna(subset=['price'])

        self.pc.df['date'] = pd.to_datetime(self.pc.df['date'])
        self.pc.df['date'] = self.pc.df['date'] - pd.to_timedelta(self.pc.df['date'].dt.hour, unit='h')

        return self.pc.df

    def make_requests(self, symbols, products, product_type):
        for symbol, hub_name in symbols.items():
            params = {
                'priceSymbol': symbol,
                # if end_date == start_date response will contain 1 item
                'daysback': str((self.end_date - self.start_date).days + 1),
                'chartstopdate': self.end_date.strftime('%Y/%m/%d'),
                'dailybarinterval': 'Days',
                'aggregatepriceselection': 'First'
            }
            r = requests.get(self.url, params=params, headers=self.headers)

            df = pd.DataFrame(r.json()['results']['items'])
            if len(r.json()['results']['items']) != 0:
                self.pc.append(
                    date=df['tradedatetimegmt'],
                    price=df['close'],
                    hub=hub_name,
                    currency=CURRENCIES[hub_name],
                    unit=UNITS[hub_name],
                    prices_name='EEX Natural Gas Spot Index ' + hub_name,
                    price_type='PX_SETTLE',
                    products=products,  # WD or MA
                    product_type=product_type,  # Daily or Month
                    id_source=9
                )

    def get_df(self):
        return self.pc.df

    # unused, get list of symbols for requests from <script> html page response
    def get_symbols_monthly(self):
        return self.get_symbols([r'(?<=let baseSymbols4 = \[).+(?=];)', r'"\$E\.[A-Z0-9_]+"'])

    def get_symbols_daily(self):
        return self.get_symbols([r'(?<=let baseSymbols = \[).+(?=];)', r'"#E\.[A-Z0-9_]+"'])

    def get_symbols(self, patterns):
        r = requests.get('https://www.eex.com/en/market-data/natural-gas/indices', headers=self.headers)
        symbol_list = re.findall(patterns[0], r.text)
        symbols = re.findall(patterns[1], symbol_list[0])
        return symbols


if __name__ == '__main__':
    parser = EexNaturalGasIndicesParser(
        end_date=date.today()
    )
    parser.parse()
    result = parser.get_df()
    # result.to_excel('./output/out_indices.xlsx')
