import json

import pandas as pd
import requests

from datetime import date, timedelta
from eex_ng.pandas_configurator import PandasConfigurator
from eex_loader.exxeta_settings import UNITS, CURRENCIES


class EexNaturalGasSpotParser:
    """
    Class to parse Natural Gas Spot from eex
    https://www.eex.com/en/market-data/natural-gas/spot
    """

    url = 'https://webservice-eex.gvsi.com/query/json/getDaily/ontradeprice/onexchsingletradevolume/close' \
          '/onexchtradevolumeeex/tradedatetimegmt/'

    headers = {
        'Origin': 'https://www.eex.com',
        'Referer': 'https://www.eex.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
    }

    # '<symbol for request>': '<hub_name_for_db>'
    symbols_weekend = {
        '"#E.CEGH_GWE1"': 'Austria VTP',  # CEGH VTP
        '"#E.OTE_GSWE"':  'Czech VTP',    # CZ VTP
        '"#E.ETF_GWE1"':  'Denmark ETF',  # ETF
        '"#E.NBP_GPWE"':  'NBP',
        '"#E.PEG_GWE1"':  'Peg Nord',     # PEG
        '"#E.PVB_GSWE"':  'PSV',
        '"#E.THE_GWE1"':  'PVB',
        '"#E.TTF_GWE1"':  'THE VTP',      # THE
        '"#E.ZEE_GWWE"':  'TTF',
        '"#E.ZTP_GTWE"':  'ZEE',
    }

    # '<symbol for request>': '<hub_name_for_db>'
    symbols_day_ahead = {
        '"#E.CEGH_GND1"': 'Austria VTP',  # CEGH VTP
        '"#E.OTE_GSND"':  'Czech VTP',    # CZ VTP
        '"#E.ETF_GND1"':  'Denmark ETF',  # ETF
        '"#E.NBP_GPND"':  'NBP',
        '"#E.PEG_GND1"':  'Peg Nord',     # PEG
        '"#E.PVB_GSND"':  'PSV',
        '"#E.THE_GND1"':  'PVB',
        '"#E.TTF_GND1"':  'THE VTP',      # THE
        '"#E.ZEE_GWND"':  'TTF',
        '"#E.ZTP_GTND"':  'ZEE',
    }

    pc = PandasConfigurator()

    def __init__(self, end_date: date, start_date: date):
        if start_date is None:
            self.start_date = end_date - timedelta(days=10)
        else:
            self.start_date = start_date
        self.end_date = end_date
        if self.end_date - self.start_date < timedelta(0):
            raise Exception("End_date can't be before Start_date")

    def parse(self):

        self.make_requests(symbols=self.symbols_day_ahead,
                           products='DA',
                           product_type='Day')

        self.make_requests(symbols=self.symbols_weekend,
                           products='WKND',
                           product_type='Day')

        self.pc.df = self.pc.df.dropna(subset=['price'])
        self.pc.df = self.pc.df.drop(self.pc.df[self.pc.df.price <= 0].index)
        self.pc.df['date'] = pd.to_datetime(self.pc.df['date'])
        self.pc.df['date'] = self.pc.df['date'] - pd.to_timedelta(self.pc.df['date'].dt.hour, unit='h')

    def make_requests(self, symbols, products, product_type):
        """
        make requests and append response to DataFrame using PandasConfigurator
        """
        for symbol, hub_name in symbols.items():
            params = {
                'priceSymbol': symbol,
                # if end_date == start_date response will contain 1 item
                'chartstartdate': self.start_date.strftime('%Y/%m/%d'),
                'chartstopdate': self.end_date.strftime('%Y/%m/%d'),
                'dailybarinterval': 'Days',
                'aggregatepriceselection': 'First'
            }
            r = requests.get(self.url, params=params, headers=self.headers)
            df = pd.DataFrame(r.json()['results']['items'])
            if len(r.json()['results']['items']) != 0:
                self.pc.append(
                    date=df['tradedatetimegmt'],
                    price=df['ontradeprice'],
                    hub=hub_name,
                    currency=CURRENCIES[hub_name],
                    unit=UNITS[hub_name],
                    prices_name='EEX Natural Gas Spot ' + hub_name + ' Weekend' if products == 'WD' else ' Day Ahead',
                    price_type='PX_LAST',
                    products=products,  # DA or WKND
                    product_type=product_type,
                    id_source=9
                )

    def get_df(self):
        return self.pc.df


if __name__ == '__main__':
    parser = EexNaturalGasSpotParser(
        end_date=date.today(),
        start_date=date.today()
    )
    parser.parse()
    result = parser.get_df()
    result.to_excel('./output/out_spot.xlsx')
