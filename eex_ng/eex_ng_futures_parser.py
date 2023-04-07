import pandas as pd
import requests

from datetime import date, timedelta, datetime
from eex_ng.eex_periods import Day, Weekend, Week, Month, Quarter, Season, Year, Period
from eex_ng.pandas_configurator import PandasConfigurator
from eex_ng.utils import daterange
from eex_loader.exxeta_settings import CURRENCIES, UNITS


class EexNaturalGasFuturesParser:
    """
    Class to parse EEX Natural Gas Futures
    https://www.eex.com/en/market-data/natural-gas/futures
    """

    url = 'https://webservice-eex.gvsi.com/query/json/getChain/gv.pricesymbol/gv.displaydate/gv.expirationdate' \
          '/tradedatetimegmt/gv.eexdeliverystart/ontradeprice/close/onexchsingletradevolume/onexchtradevolumeeex' \
          '/offexchtradevolumeeex/openinterest/'

    headers = {
        'Origin': 'https://www.eex.com',
        'Referer': 'https://www.eex.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
    }

    symbols = {
        '/E.G8': 'Austria VTP',  # CEGH VTP
        '/E.G1': 'Czech VTP',  # CZ VTP
        '/E.GD': 'Denmark ETF',  # ETF
        # '/E.GLJM': 'JKM',  # JKM has only Month period => JKM request is done separately
        '/E.G9': 'NBP',
        '/E.G5': 'Peg Nord',  # PEG
        '/E.GC': 'PSV',
        '/E.GE': 'PVB',
        '/E.G0': 'THE VTP',  # THE
        '/E.G3': 'TTF',
        '/E.GA': 'ZEE',
        '/E.GB': 'ZTP'
    }

    symbols_EGSI = {
        '/E.G8': 'Austria VTP EGSI',  # code is the same as for the non EGSI
        '/E.GG': 'THE VTP EGSI',
        '/E.G3': 'TTF EGSI'  # code is the same as for the non EGSI
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

        for on_date in daterange(self.start_date, self.end_date):
            if on_date.weekday() >= 5:
                continue
            for period in [Month(), Quarter(), Season(), Year()]:
                self.make_requests(
                    # dictionary that contains {'<symbol_for_request>': '<hub_name>'}
                    symbols={'"' + symbol + period.symbol + '"': self.symbols[symbol] for symbol in self.symbols},
                    on_date=on_date,
                    period=period
                )

            # EGSI requests
            for period in [Day(), Weekend(), Week(), Month(), Quarter(), Season(), Year()]:
                self.make_requests(
                    # dictionary that contains {'<symbol_for_request>': '<hub_name>'}
                    symbols={'"' + symbol + period.symbol_egsi + '"': self.symbols_EGSI[symbol] for symbol in
                             self.symbols_EGSI},
                    on_date=on_date,
                    period=period
                )

            self.make_requests(
                # dictionary that contains {'<symbol_for_request>': '<hub_name>'}
                symbols={'"/E.GLJM"': 'JKM'},
                on_date=on_date,
                period=Month()
            )

        self.pc.df = self.pc.df.dropna(subset=['price'])
        self.pc.df = self.pc.df.drop(self.pc.df[self.pc.df.price <= 0].index)
        self.pc.df['date'] = pd.to_datetime(self.pc.df['date'])
        self.pc.df['date'] = self.pc.df['date'] - pd.to_timedelta(self.pc.df['date'].dt.hour, unit='h')

        return self.pc.df

    def make_requests(self, symbols, on_date, period: Period):
        """
        makes requests and appends response to DataFrame using PandasConfigurator
        """
        for symbol, hub_name in symbols.items():
            params = {
                'optionroot': symbol,
                'onDate': on_date.strftime('%Y/%m/%d')
            }
            r = requests.get(self.url, params=params, headers=self.headers)

            df = pd.DataFrame(r.json()['results']['items'])
            if len(r.json()['results']['items']) != 0:
                df['gv.displaydate'] = df['gv.displaydate'].map(lambda d: datetime.strptime(d, "%m/%d/%Y"))
                for price, price_type in {'ontradeprice': 'PX_LAST', 'close': 'PX_SETTLE'}.items():
                    self.pc.append(
                        date=df['tradedatetimegmt'],
                        price=df[price],
                        hub=hub_name,
                        currency=CURRENCIES[hub_name],
                        unit=UNITS[hub_name],
                        prices_name='EEX ' + hub_name + ' Natural Gas Futures',
                        price_type=price_type,
                        products=df['gv.displaydate'].map(period.get_products),
                        product_type=period.print(),
                        beg_date=df['gv.displaydate'],
                        id_source=9
                    )

    def get_df(self):
        return self.pc.df


if __name__ == '__main__':
    parser = EexNaturalGasFuturesParser(
        end_date=date.today(),
        start_date=date.today()
    )
    parser.parse()
    result = parser.get_df()
    result.to_excel('./output/out_futures.xlsx')
