"""
parser from https://umm.gassco.no/ and https://umm.gassco.no/ch/
use request to catch xhr data and js data
Business goal: download aggregated future event data
and history data for each params
"""
import requests
import pandas as pd
import numpy as np
from db_initial_connection import base, session
from curves import Curves
from datetime import datetime

class GasscoParser:

    def __init__(self):
        """
        configure cookies for passing agreements
        choice parsing's url
        """
        self.cookies = {
            '_ga': 'GA1.2.807198347.1658330305',
            '_gid': 'GA1.2.84288676.1658330305',
            'Igng4tfXGSb8T3XfxU': '1658352269306:1',
            'JSESSIONID': '0369E31C6FF288A70A000563B18CFB4E'
        }
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image'
                      '/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit'
                          '/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
        }
        self.proxy = {'http': "http://192.168.114.230:8080",
                       'https': "http://192.168.114.230:8080"}

        self.urls = ['https://umm.gassco.no/ch/2Y/1',
                    'https://umm.gassco.no/ch/2Y/4',
                    'https://umm.gassco.no/ch/2Y/5',
                    'https://umm.gassco.no/ch/2Y/6',
                    'https://umm.gassco.no/ch/2Y/7',
                    'https://umm.gassco.no/ch/2Y/8',
                    'https://umm.gassco.no/ch/2Y/9',
                    'https://umm.gassco.no/ch/2Y/43',
                    'https://umm.gassco.no/ch/2Y/44'
                    ]

    def gassco_parsing(self):
        """
        """
        result_df = pd.DataFrame({
            'date': [],
            'source_name': [],
            'point_name': [],
            'value': [],
            'unit': [],
            'flow_type': [],
            'curve': [],
            'from_country': [],
            'to_country': [],
            'from_company': [],
            'to_company': []
        })
        for url in self.urls:
            response = requests.get(url, cookies=self.cookies, headers=self.headers, proxies=self.proxy)
            print(response.status_code)
            data_json = response.json()
            df = pd.DataFrame(data_json["data"])
            df['x'] = df['x'].apply(str)
            df['y'] = df['y'].apply(float)
            df['x'] = df['x'].str.slice(0, 10)
            df['x'] = pd.to_datetime(df['x'], unit='s')
            df['date'] = df['x']
            df['value'] = df['y']
            df['unit'] = 'MSm3'
            df['source_name'] = 'Gassco'
            df['flow_type'] = 'nomination'
            df['curve'] = 'exit'
            if f"{data_json['label']}" == 'Dornum':
                df['point_name'] = 'Dornum (EPT1 & EPT2)'
                df['from_country'] = 'NO'
                df['to_country'] = 'DE'
                df['from_company']= 'Gassco'
                df['to_company'] = None
            elif f"{data_json['label']}" == 'Dunkerque':
                df['point_name'] = 'Dunkerque'
                df['from_country'] = 'NO'
                df['to_country'] = 'FR'
                df['from_company']= 'Gassco'
                df['to_company'] = 'GRTgaz'
            elif f"{data_json['label']}" == 'Easington':
                df['point_name'] = 'Easington'
                df['from_country'] = 'NO'
                df['to_country'] = 'UK'
                df['from_company']= 'Gassco'
                df['to_company'] = 'National Grid'
            elif f"{data_json['label']}" == 'Emden':
                df['point_name'] = 'Emden'
                df['from_country'] = 'NO'
                df['to_country'] = None
                df['from_company']= 'Gassco'
                df['to_company'] = None
            elif f"{data_json['label']}" == 'Zeebrugge':
                df['point_name'] = 'Zeebrugge ZPT'
                df['from_country'] = 'NO'
                df['to_country'] = 'BE'
                df['from_company']= 'Gassco'
                df['to_company'] = 'Fluxys Belgium'
            elif f"{data_json['label']}" == 'St.Fergus':
                df['point_name'] = 'St. Fergus'
                df['from_country'] = 'NO'
                df['to_country'] = 'UK'
                df['from_company']= 'Gassco'
                df['to_company'] = 'National Grid'
            elif f"{data_json['label']}" == 'Fields Delivering into SEGAL':
                df['point_name'] = 'Fields Delivering into SEGAL'
                df['from_country'] = 'NO'
                df['to_country'] = 'UK'
                df['from_company']= 'Gassco'
                df['to_company'] = 'National Grid'
            elif f"{data_json['label']}" == 'Other Exit Nominations':
                df['point_name'] = 'Other Exit Nominations'
                df['from_country'] = 'NO'
                df['to_country'] = None
                df['from_company']= 'Gassco'
                df['to_company'] = None
            elif f"{data_json['label']}" == 'Sum Exit Nominations NCS':
                df['point_name'] = 'Sum Exit Nominations NCS'
                df['from_country'] = 'NO'
                df['to_country'] = None
                df['from_company']= 'Gassco'
                df['to_company'] = None
            df = df.drop('x', axis=1).drop('y', axis=1)
            result_df = pd.concat([result_df, df], ignore_index=True)
        return result_df

class GasscoLoader:

    def __init__(self):

        # curves_dict
        self.table = base.classes.flow_curves
        self.flows_unit_dict = {'exit': 'MSm3',
                                'exit': 'MSm3'}
        self.curves_class = Curves()

    def validate(self, source_name: str, point_name: str, curve_name: str, flow_type: str, unit: str = None):

        search_source_name = session.query(base.classes.source_dict.id).filter(
            base.classes.source_dict.source_name == source_name).all()

        if len(search_source_name) != 1:
            raise IndexError(source_name)
        else:
            id_source = search_source_name[0]

        search_delivery_point = session.query(base.classes.delivery_point_dict.id).filter(
            base.classes.delivery_point_dict.point_name == point_name).all()

        if len(search_delivery_point) != 1:
            raise IndexError(point_name)
        else:
            id_dp = search_delivery_point[0]

        search_flow_type = session.query(base.classes.flow_types.id).filter(
            base.classes.flow_types.flow_type == flow_type).all()

        if len(search_flow_type) != 1:
            raise IndexError(flow_type)
        else:
            id_flow_type = search_flow_type[0]

        if unit == None:
            unit = self.flows_unit_dict[curve_name]
        else:
            pass

        search_unit = session.query(base.classes.units_dict.id).filter(base.classes.units_dict.unit_name == unit).all()

        if len(search_unit) != 1:
            raise IndexError(unit)
        else:
            id_unit = search_unit[0]

        check_exists = session.query(self.table.id).filter(
            self.table.id_source == id_source[0],
            self.table.id_point == id_dp[0],
            self.table.id_type == id_flow_type[0],
            self.table.id_unit == id_unit[0],
            self.table.curve_name == curve_name).all()

        if len(check_exists) != 1:
            raise IndexError('no such curve in flow_curves')
        else:
            id_flow_curve = check_exists[0][0]

        id_curve = session.query(base.classes.curves_dict.id).filter(
            base.classes.curves_dict.id_flow_curves == id_flow_curve).all()

        if len(id_curve) != 1:
            raise IndexError('no such curve in curves_dict')
        else:
            return id_curve[0][0]

    def insert(self, df: pd.DataFrame = None, unit: str = None):
        data = df
        data['id_curve'] = data.apply(lambda x: self.validate(x.source_name, x.point_name, x.curve, x.flow_type, unit), axis=1)
        data = data[['date', 'id_curve', 'value']].to_numpy()
        for row in data:
            print(row)
            if row[2] == '-':
                self.curves_class.insert_new_data(row[1], row[0], 0)
            else:
                self.curves_class.insert_new_data(row[1], row[0], np.float64(row[2]))
        return 'ok'

if __name__ == '__main__':
    final_df = GasscoParser().gassco_parsing()
    GasscoLoader().insert(df=final_df)

