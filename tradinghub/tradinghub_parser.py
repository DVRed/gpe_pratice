import urllib.parse
from datetime import date, timedelta
import requests
import pandas as pd


class TradingHubParser:

    # Dict to convert value from json to value to print to result

    delivery_points = {
        'slPsyn_H_Gas': 'SLPsyn H-Gas',
        'slPana_H_Gas': 'SLPana H-Gas',
        'slPsyn_L_Gas': 'SLPsyn L-Gas',
        'slPana_L_Gas': 'SLPana L-Gas',
        'rlMmT_H_Gas': 'RLMmT H-Gas',
        'rlMmT_L_Gas': 'RLMmT L-Gas',
        'rlMoT_H_Gas': 'RLMoT H-Gas',
        'rlMoT_L_Gas': 'RLMoT L-Gas'
    }

    def __init__(self, end_date: date, start_date: date = None):
        if start_date is None:
            self.start_date = end_date - timedelta(days=10)
        else:
            self.start_date = start_date
        self.end_date = end_date

    def get_url(self):
        params = {
            'DatumStart': self.start_date.strftime('%m-%d-%Y'),
            'DatumEnde': self.end_date.strftime('%m-%d-%Y'),
            'GasXType_Id': 'all'
        }
        url = 'https://datenservice-api.tradinghub.eu/api/evoq/GetAggregierteVerbrauchsdatenTabelle?'
        return url + urllib.parse.urlencode(params)

    def parse(self):
        url = self.get_url()
        r = requests.get(url)
        json_data = r.json()

        # The JSON has the next structure
        # [{...},
        # {
        #     gasXType:       "allocation"
        #     statusDE:       "vorläufig"
        #     statusEN:       "preliminary"
        #     gastag:         "2023-03-16T06:00:00"
        #     slPsyn_H_Gas:   1088888976
        #     slPana_H_Gas:   211178664
        #     slPsyn_L_Gas:   201686808
        #     slPana_L_Gas:   70836288
        #     rlMmT_H_Gas:    null
        #     rlMmT_L_Gas:    null
        #     rlMoT_H_Gas:    null
        #     rlMoT_L_Gas:    null
        # },
        # {...},
        # ...]

        df = pd.DataFrame(json_data)
        df = df.rename(columns=(TradingHubParser.delivery_points | {'gastag': 'date', 'statusEN': 'flow_type'}))
        df = pd.melt(df, id_vars=['date', 'flow_type'],
                     value_vars=TradingHubParser.delivery_points.values(),
                     var_name='delivery_point', value_name='value')
        df = df.dropna()
        df['from_country'] = 'DE'
        df['to_country'] = 'DE'
        df['curve_type'] = 'Physical_flow'

        # '2023-03-17T06:00:00'
        # '%Y-%m-%dT%H:%M:%S'
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S')

        df = df[['date', 'delivery_point', 'from_country', 'to_country', 'curve_type', 'flow_type', 'value']]
        return df


if __name__ == '__main__':
    parser = TradingHubParser(end_date=date.today())
    data_frame = parser.parse()
    print(data_frame)
