import urllib.parse
from datetime import date
import requests
from dateutil.relativedelta import relativedelta
import pandas as pd


def get_url():
    end_date = date.today()
    start_date = (end_date - relativedelta(months=4)).replace(day=1)
    params = {
        'DatumStart': start_date.strftime('%m-%d-%Y'),
        'DatumEnde': end_date.strftime('%m-%d-%Y'),
        'GasXType_Id': 'all'
    }
    url = 'https://datenservice-api.tradinghub.eu/api/evoq/GetAggregierteVerbrauchsdatenTabelle?'
    return url + urllib.parse.urlencode(params)


def parse(output_path):
    url = get_url()
    r = requests.get(url)
    json_data = r.json()

    df = pd.DataFrame({
        'date': [],
        'delivery_point': [],
        'from_country': [],
        'to_country': [],
        'curve_type': [],
        'flow_type': [],
        'value': []
    })

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

    # TODO JSON can contain null

    for i in range(len(json_data)):

        # json_data[i]['gastag'] = 2023-03-13T06:00:00
        # [0:10] = 2023-03-13
        # split('-') = ['2023', '03', '13']
        date_split = json_data[i]['gastag'][0:10].split('-')
        # reverse = ['13', '03', '2023']
        date_split.reverse()
        # join = '13/03/2023'
        parsed_date = '/'.join(date_split)

        for p in TradingHubParser.delivery_points.keys():
            if json_data[i][p] is None:
                continue
            df = pd.concat([
                df,
                pd.DataFrame({
                    'date': parsed_date,
                    'delivery_point': TradingHubParser.delivery_points[p],
                    'from_country': 'DE',
                    'to_country': 'DE',
                    'curve_type': 'Physical_flow',
                    'flow_type': json_data[i]['statusEN'],
                    'value': json_data[i][p]
                }, index=[0])
            ])

    df.to_excel(output_path, index=False)


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
