"""
Data parcer for The European Energy Exchange (EEX) - is the leading energy exchange.
https://www.eex.com/en/market-data/power/futures#%7B%22snippetpicker%22%3A%22EEX%20Belgian%20Power%20Futures%22%7D
TODO:perfect-
Business goal:
Scrapy tables from url and format them to expected view
"""
import requests
from datetime import timedelta, datetime
import pandas as pd
import re
import numpy as np
from monthdelta import monthdelta

class Eex:

    def __init__(self):
        """
        choice
        params for scraping
        and dictionaries for swapping

        """
        # get company name
        #EEX German/Austrian Power Future
        self.replace_list = {'EEX Austrian Power Future':'Austria','EEX Belgian Power Futures':'Belgium',
                             'EEX-PXE Bulgarian Power Futures':'Bulgaria','EEX-PXE Czech Power Futures':'Czechia',
                             'EEX Dutch Power Futures':'Netherlands','EEX French Power Futures':'France',
                             'EEX GB Power Futures':'United Kingdom','EEX German Power Future':'Germany',
                             'EEX Greek Power Futures':'Greece','EEX Nordic Power Futures':'Nordic',
                             'EEX-PXE Serbian Power Futures':'Republic of Serbia','EEX Spanish Power Futures':'Spain',
                             'EEX Swiss Power Futures':'Switzerland','EEX-PXE Hungarian Power Futures':'Hungary',
                             'EEX Italian Power Futures':'Italy','EEX Japanese Power Futures - Kansai':'JP_Kansai',
                             'EEX Japanese Power Futures - Tokyo':'JP_Tokyo','EEX-PXE Polish Power Futures':'Poland',
                             'EEX-PXE Romanian Power Futures':'Romania','EEX-PXE Slovakian Financial Power Future':'Slovakian',
                             'EEX-PXE Slovenian Power Futures':'Slovenian'}
        #self.proxy = {'http': 'http://192.168.114.230:8080', 'https': 'http://192.168.114.230:8080'}
        self.country_marker = {'E.AT': "EEX Austrian Power Future",
                               'E.AB': "EEX Austrian Power Future",
                               'E.AP': "EEX Austrian Power Future",
                               'E.AW': "EEX Austrian Power Future",
                               'E.Q1': "EEX Belgian Power Futures",
                               'E.FK': "EEX-PXE Bulgarian Power Futures",
                               'E.FX': "EEX-PXE Czech Power Futures",
                               'E.WX': "EEX-PXE Czech Power Futures",
                               'E.PX': "EEX-PXE Czech Power Futures",
                               'E.Q0': "EEX Dutch Power Futures",
                               'E.QW': "EEX Dutch Power Futures",
                               'E.QP': "EEX Dutch Power Futures",
                               'E.QB': "EEX Dutch Power Futures",
                               'E.F7': "EEX French Power Futures",
                               'E.P7': "EEX French Power Futures",
                               'E.FU': "EEX GB Power Futures",
                               'E.DE': "EEX German Power Future",
                               'E.DB': "EEX German Power Future",
                               'E.DW': "EEX German Power Future",
                               'E.DP': "EEX German Power Future",
                               'E.F1': "EEX German/Austrian Power Future",
                               'E.FF': "EEX Greek Power Futures",
                               'E.F9': "EEX-PXE Hungarian Power Futures",
                               'E.P9': "EEX-PXE Hungarian Power Futures",
                               'E.FD': "EEX Italian Power Futures",
                               'E.PD': "EEX Italian Power Futures",
                               'E.FQ': "EEX Japanese Power Futures - Kansai",
                               'E.FO': "EEX Japanese Power Futures - Tokyo",
                               'E.FB': "EEX Nordic Power Futures",
                               'E.FP': "EEX-PXE Polish Power Futures",
                               'E.FH': "EEX-PXE Romanian Power Futures",
                               'E.FR': "EEX-PXE Romanian Power Futures",
                               'E.FZ': "EEX-PXE Serbian Power Futures",
                               'E.FY': "EEX-PXE Slovakian Financial Power Future",
                               'E.FV': "EEX-PXE Slovenian Power Futures",
                               'E.FE': "EEX Spanish Power Futures",
                               'E.FC': "EEX Swiss Power Futures"
                               }
        # params for request
        self.list_baseload = ['/E.AB_DAILY', '/E.AWB_WEEK', '/E.ATB_WEEK', '/E.ATBM', '/E.ATBQ', '/E.ATBY', '/E.Q1BM',
                              '/E.Q1BQ', '/E.Q1BY', '/E.FKB_WEEK', '/E.FKBM', '/E.FKBQ', '/E.FKBY', '/E.FX_DAILY',
                              '/E.WXB_WEEK', '/E.FXB_WEEK', '/E.FXBM', '/E.FXBQ', '/E.FXBY', '/E.QB_DAILY',
                              '/E.QWB_WEEK', '/E.Q0B_WEEK', '/E.Q0BM', '/E.Q0BQ', '/E.Q0BY', '/E.F7_DAILY',
                              '/E.F7W_WEEK', '/E.F7B_WEEK', '/E.F7BM', '/E.F7BQ', '/E.F7BY', '/E.FU_DAILY',
                              '/E.FUW_WEEK', '/E.FUB_WEEK', '/E.FUBM', '/E.FUBQ', '/E.FUBS', '/E.FUBY', '/E.DB_DAILY',
                              '/E.DWB_WEEK', '/E.DEB_WEEK', '/E.DEBM', '/E.DEBQ', '/E.DEBY', '/E.F1BM', '/E.F1BQ',
                              '/E.F1BY', '/E.FFBM', '/E.FFBQ', '/E.FFBY', '/E.FBB_WEEK', '/E.FBBM', '/E.FBBQ',
                              '/E.FBBY', '/E.FZB_WEEK', '/E.FZBM', '/E.FZBQ', '/E.FZBY', '/E.FE_DAILY', '/E.FEW_WEEK',
                              '/E.FEB_WEEK', '/E.FEBM', '/E.FEBQ', '/E.FEBY', '/E.FC_DAILY', '/E.FCW_WEEK',
                              '/E.FCB_WEEK', '/E.FCBM', '/E.FCBQ', '/E.FCBY', '/E.F9_DAILY', '/E.W9B_WEEK',
                              '/E.F9B_WEEK', '/E.F9BM', '/E.F9BQ', '/E.F9BY', '/E.FD_DAILY', '/E.FDW_WEEK',
                              '/E.FDB_WEEK', '/E.FDBM', '/E.FDBQ', '/E.FDBY', '/E.FQB_WEEK', '/E.FQBM', '/E.FQBQ',
                              '/E.FQBS', '/E.FQBY', '/E.FOB_WEEK', '/E.FOBM', '/E.FOBQ', '/E.FOBS', '/E.FOBY',
                              '/E.FPBM', '/E.FPBQ', '/E.FPBY', '/E.FHB_WEEK', '/E.FHBM', '/E.FHBQ', '/E.FHBY',
                              '/E.FYBM', '/E.FYBQ', '/E.FYBY', '/E.FVB_WEEK', '/E.FVBM', '/E.FVBQ', '/E.FVBY']
        # params for request
        self.list_peakload = ['/E.AP_DAILY', '/E.AWP_WEEK', '/E.ATP_WEEK', '/E.ATPM', '/E.ATPQ', '/E.ATPY',
                              '/E.PX_DAILY', '/E.WXP_WEEK', '/E.FXP_WEEK', '/E.FXPM', '/E.FXPQ', '/E.FXPY',
                              '/E.QP_DAILY', '/E.QWP_WEEK', '/E.Q0P_WEEK', '/E.Q0PM', '/E.Q0PQ', '/E.Q0PY',
                              '/E.P7_DAILY', '/E.P7W_WEEK', '/E.F7P_WEEK', '/E.F7PM', '/E.F7PQ', '/E.F7PY',
                              '/E.FUP_WEEK', '/E.FUPM', '/E.FUPQ', '/E.FUPS', '/E.FUPY', '/E.DP_DAILY', '/E.DWP_WEEK',
                              '/E.DEP_WEEK', '/E.DEPM', '/E.DEPQ', '/E.DEPY', '/E.P9_DAILY', '/E.W9P_WEEK',
                              '/E.F9P_WEEK', '/E.F9PM', '/E.F9PQ', '/E.F9PY', '/E.PD_DAILY', '/E.PDW_WEEK',
                              '/E.FDP_WEEK', '/E.FDPM', '/E.FDPQ', '/E.FDPY', '/E.FQP_WEEK', '/E.FQPM', '/E.FQPQ',
                              '/E.FQPS', '/E.FQPY', '/E.FOP_WEEK', '/E.FOPM', '/E.FOPQ', '/E.FOPS', '/E.FOPY',
                              '/E.FPPM', '/E.FPPQ', '/E.FPPY', '/E.FRP_WEEK', '/E.FRPM', '/E.FRPQ', '/E.FRPY',
                              '/E.FYPM', '/E.FYPQ', '/E.FYPY', '/E.FVP_WEEK', '/E.FVPM', '/E.FVPQ', '/E.FVPY']
        self.eex_list_create()

    def eex_list_create(self):
        """
        create list with date and company marker for pull in request
        """
        end_date = (datetime.today() - timedelta(days=0))
        start_date = (datetime.today() - timedelta(days=1))
        res = pd.date_range(start_date, end_date, freq="1D")
        self.headers = {
            'Origin': 'https://www.eex.com',
            'Referer': 'https://www.eex.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/104.0.5112.102 Safari/537.36 Edg/104.0.1293.70',
        }
        self.params_baseload = [{'optionroot': f'"{i}"', 'onDate': f'{j.strftime("%Y-%m-%d")}', } for i in
                                self.list_baseload for j in res]
        self.params_peakload = [{'optionroot': f'"{i}"', 'onDate': f'{j.strftime("%Y-%m-%d")}', } for i in
                                self.list_peakload for j in res]

    def eex_borderland_parser(self):
        """
        create database:
        response -> choice json list, choice date params
        """
        table_list = []
        for i in range(len(self.params_baseload)):
            # list = [1, 2, 3, 4, 5]
            # if params[i]["onDate"].weekday() in list:
            response = requests.get(
                'https://webservice-eex.gvsi.com/query/json/getChain/gv.pricesymbol/gv.displaydate/gv.expirationdate'
                '/tradedatetimegmt/gv.eexdeliverystart/ontradeprice/close/onexchsingletradevolume'
                '/onexchtradevolumeeex/offexchtradevolumeeex/openinterest/',
                params=self.params_baseload[i], headers=self.headers)
            data = response.json()
            if re.match(r'(.*)_DAILY', self.params_baseload[i]["optionroot"]):  # Day
                for j in range(len(data["results"]["items"])):
                    dayli = data["results"]["items"][j]["gv.displaydate"]
                    dayli = datetime.strptime(dayli, "%m/%d/%Y").date()
                    dayli = dayli.strftime("%d/%m/%Y")
                    table = {
                        'date': self.params_baseload[i]["onDate"],
                        'period': 'Day',
                        'name': self.params_baseload[i]["optionroot"][2:6],
                        'del_start': f"{dayli}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)W{1}(.*)(.W)(.*)', self.params_baseload[i]["optionroot"]):  # Weekend
                for j in range(len(data["results"]["items"])):
                    weekend = data["results"]["items"][j]["gv.displaydate"]
                    weekend = datetime.strptime(weekend, "%m/%d/%Y").date()
                    weekend = weekend.strftime("%d/%m")
                    table = {
                        'date': self.params_baseload[i]["onDate"],
                        'period': 'Weekend',
                        'name': self.params_baseload[i]["optionroot"][2:6],
                        'del_start': f"WkEnd {weekend}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)[^W][^W][^W][^W][_](.*)', self.params_baseload[i]["optionroot"]):  # Week
                for j in range(len(data["results"]["items"])):
                    week = data["results"]["items"][j]["gv.displaydate"]
                    week = datetime.strptime(week, "%m/%d/%Y").date()
                    week = week.strftime("%W/%y")
                    table = {
                        'date': self.params_baseload[i]["onDate"],
                        'period': 'Week',
                        'name': self.params_baseload[i]["optionroot"][2:6],
                        'del_start': f"Week {week}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)[M]', self.params_baseload[i]["optionroot"]):  # Month
                for j in range(len(data["results"]["items"])):
                    month = data["results"]["items"][j]["gv.displaydate"]
                    month = datetime.strptime(month, "%m/%d/%Y").date()
                    month = month.strftime("%b/%Y")
                    table = {
                        'date': self.params_baseload[i]["onDate"],
                        'period': 'Month',
                        'name': self.params_baseload[i]["optionroot"][2:6],
                        'del_start': f"{month}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)Q', self.params_baseload[i]["optionroot"]):  # Quarter
                for j in range(len(data["results"]["items"])):
                    quarter = data["results"]["items"][j]["gv.displaydate"]
                    quarter = datetime.strptime(quarter, "%m/%d/%Y").date()
                    year = quarter.strftime("%Y")
                    quarter = pd.Timestamp(quarter).quarter
                    table = {
                        'date': self.params_baseload[i]["onDate"],
                        'period': 'Quarter',
                        'name': self.params_baseload[i]["optionroot"][2:6],
                        'del_start': f"{quarter}/{year}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)[Y]', self.params_baseload[i]["optionroot"]):  # Year
                for j in range(len(data["results"]["items"])):
                    year = data["results"]["items"][j]["gv.displaydate"]
                    year = datetime.strptime(year, "%m/%d/%Y").date()
                    year = year.strftime("%Y")
                    table = {
                        'date': self.params_baseload[i]["onDate"],
                        'period': 'Year',
                        'name': self.params_baseload[i]["optionroot"][2:6],
                        'del_start': f"Cal-{year}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
        df = pd.DataFrame(table_list)
        df.insert(4, 'type', 'Baseload')
        self.df_1 = df

    def eex_peakload_parser(self):
        """
        create database:
        response -> choice json list, choice date params
        """
        table_list = []
        for i in range(len(self.params_peakload)):
            # list = [1, 2, 3, 4, 5]
            # if params[i]["onDate"].weekday() in list:
            #     print(params[i])
            response = requests.get(
                'https://webservice-eex.gvsi.com/query/json/getChain/gv.pricesymbol/gv.displaydate/gv.expirationdate'
                '/tradedatetimegmt/gv.eexdeliverystart/ontradeprice/close/onexchsingletradevolume'
                '/onexchtradevolumeeex/offexchtradevolumeeex/openinterest/',
                params=self.params_peakload[i], headers=self.headers, verify=False)
            data = response.json()
            if re.match(r'(.*)_DAILY', self.params_peakload[i]["optionroot"]):  # Day
                for j in range(len(data["results"]["items"])):
                    dayli = data["results"]["items"][j]["gv.displaydate"]
                    dayli = datetime.strptime(dayli, "%m/%d/%Y").date()
                    dayli = dayli.strftime("%d/%m/%Y")
                    table = {
                        'date': self.params_peakload[i]["onDate"],
                        'period': 'Day',
                        'name': self.params_peakload[i]["optionroot"][2:6],
                        'del_start': f"{dayli}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)W{1}(.*)(.W)(.*)', self.params_peakload[i]["optionroot"]):  # Weekend
                for j in range(len(data["results"]["items"])):
                    weekend = data["results"]["items"][j]["gv.displaydate"]
                    weekend = datetime.strptime(weekend, "%m/%d/%Y").date()
                    weekend = weekend.strftime("%d/%m")
                    table = {
                        'date': self.params_peakload[i]["onDate"],
                        'period': 'Weekend',
                        'name': self.params_peakload[i]["optionroot"][2:6],
                        'del_start': f"WkEnd {weekend}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)[^W][^W][^W][^W][_](.*)', self.params_peakload[i]["optionroot"]):  # Week
                for j in range(len(data["results"]["items"])):
                    week = data["results"]["items"][j]["gv.displaydate"]
                    week = datetime.strptime(week, "%m/%d/%Y").date()
                    week = week.strftime("%W/%y")
                    table = {
                        'date': self.params_peakload[i]["onDate"],
                        'period': 'Week',
                        'name': self.params_peakload[i]["optionroot"][2:6],
                        'del_start': f"Week {week}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)[M]', self.params_peakload[i]["optionroot"]):  # Month
                for j in range(len(data["results"]["items"])):
                    month = data["results"]["items"][j]["gv.displaydate"]
                    month = datetime.strptime(month, "%m/%d/%Y").date()
                    month = month.strftime("%b/%Y")
                    table = {
                        'date': self.params_peakload[i]["onDate"],
                        'period': 'Month',
                        'name': self.params_peakload[i]["optionroot"][2:6],
                        'del_start': f"{month}",
                        'price': data["results"]["items"][j]["close"]
                    }
                    table_list.append(table)
            elif re.match(r'(.*)Q', self.params_peakload[i]["optionroot"]):  # Quarter
                for j in range(len(data["results"]["items"])):
                    quarter = data["results"]["items"][j]["gv.displaydate"]
                quarter = datetime.strptime(quarter, "%m/%d/%Y").date()
                year = quarter.strftime("%Y")
                quarter = pd.Timestamp(quarter).quarter
                table = {
                    'date': self.params_peakload[i]["onDate"],
                    'period': 'Quarter',
                    'name': self.params_peakload[i]["optionroot"][2:6],
                    'del_start': f"{quarter}/{year}",
                    'price': data["results"]["items"][j]["close"]
                }
                table_list.append(table)
            elif re.match(r'(.*)[Y]', self.params_peakload[i]["optionroot"]):  # Year
                for j in range(len(data["results"]["items"])):
                    year = data["results"]["items"][j]["gv.displaydate"]
                year = datetime.strptime(year, "%m/%d/%Y").date()
                year = year.strftime("%Y")
                table = {
                    'date': self.params_peakload[i]["onDate"],
                    'period': 'Year',
                    'name': self.params_peakload[i]["optionroot"][2:6],
                    'del_start': f"Cal-{year}",
                    'price': data["results"]["items"][j]["close"]
                }
                table_list.append(table)
                df = pd.DataFrame(table_list)
                df.insert(4, 'type', 'Peak')
                self.df_2 = df

    def _da_transformer(self,product_period:str):
        return 'DA'

    def _month_transformer(self,product_period:str):
        return product_period[:3].upper()+product_period[-2:]

    def _quarter_transformer(self,product_period:str):
        return 'Q'+product_period[:2]+product_period[-2:]

    def _year_transformer(self,product_period:str):
        return product_period[-4:]

    def _instrument_to_beg_dates(self, instrument: str,trading_date):

        month = re.findall(r'(\w\w\w)(\d\d)', instrument)
        quarter = re.findall(r'(Q\d)\/(\d\d)', instrument)
        season = re.findall(r'(Sum|Win)\s(\d\d\d\d)', instrument)
        year = re.findall(r'(\d\d\d\d)', instrument)

        quarters = {'Q1': ['01', '02', '03'], 'Q2': ['04', '05', '06'], 'Q3': ['07', '08', '09'],
                    'Q4': ['10', '11', '12']}
        seasons = {'Sum': ['04', '05', '06', '07', '08', '09'], 'Win': ['10', '11', '12', '01', '02', '03']}
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

        if instrument!='DA':
            if len(season) != 0:
                list_of_products = np.array(seasons[season[0][0]])
                beg_date = datetime(int(season[0][1]), int(list_of_products[0]), 1, 6)
                end_date = beg_date+monthdelta(6)
            elif len(quarter) != 0:
                list_of_products = np.array(quarters[quarter[0][0]])
                beg_date = datetime(int('20'+quarter[0][1]), int(list_of_products[0]), 1, 6)
                end_date = beg_date + monthdelta(3)
            elif len(year) != 0:
                beg_date = datetime(int(year[0]), 1, 1, 6)
                end_date = beg_date + monthdelta(12)
            elif len(month) != 0:
                beg_date = datetime.strptime(month[0][0] + month[0][1], '%b%y')+timedelta(hours=6)
                end_date = beg_date + monthdelta(1)
            else:
                print('UNKNOWN INSTRUMENT: {}'.format(instrument))
                raise ValueError(instrument)
        else:
            beg_date = datetime.strptime(trading_date,'%d/%m/%Y')+timedelta(hours=6)
            end_date = beg_date+timedelta(days=1)
            instrument = 'DA'

        return {'beg_date': beg_date, 'end_date': end_date,'instrument':instrument}

    def _data_transforming(self,df:pd.DataFrame):

        temp_df = df[(df['period']!='Week')&(df['period']!='Weekend')]
        instr_types_dict = {'Day':['DA',self._da_transformer],
                            'Month':['Month',self._month_transformer],
                            'Quarter':['Quarter',self._quarter_transformer],
                            'Year':['Year',self._year_transformer]}
        temp_df['products']=temp_df.apply(lambda x: instr_types_dict[x.period][1](x.del_start), axis=1)
        temp_df['beg_date'] = temp_df.apply(lambda x: self._instrument_to_beg_dates(x.products,x.del_start)['beg_date'], axis=1)
        temp_df['end_date'] = temp_df.apply(lambda x: self._instrument_to_beg_dates(x.products, x.del_start)['end_date'], axis=1)
        temp_df['products'] = temp_df.apply(lambda x: self._instrument_to_beg_dates(x.products, x.del_start)['instrument'], axis=1)
        temp_df['currency'] = 'EUR'
        temp_df['unit'] = 'MWh'
        temp_df['id_source'] = 9
        temp_df['price_type'] = 'PX_LAST'
        temp_df = temp_df.rename(columns={'marker':'hub'})
        return temp_df

    def _da_spliter(self,da_df:pd.DataFrame):

        da_days ={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

        temp_df = da_df.sort_values(by=['date','beg_date']).reset_index(drop=True)
        true_da = temp_df.drop_duplicates(subset=['date','beg_date'],keep='first')
        false_da =temp_df[~temp_df.index.isin(true_da.index.values)]
        false_da['products'] = false_da.apply(lambda x: da_days[x.beg_date.weekday()],axis=1)
        final_temp = pd.concat([true_da,false_da])
        return final_temp

    def table_coasele(self):
        """
        merge tables
        """
        dfs = pd.concat([self.df_1, self.df_2], axis=0)
        dfs['marker'] = dfs['name'].map(self.country_marker)
        dfs = dfs.drop(columns='name')
        dfs = self._data_transforming(dfs)
        dfs_da = self._da_spliter(dfs[dfs['products']=='DA'])
        dfs_non_da = dfs[dfs['products'] != 'DA']
        final_dfs = pd.concat([dfs_non_da,dfs_da])
        final_dfs = final_dfs[final_dfs['hub']!='EEX German/Austrian Power Future']
        final_dfs = final_dfs.dropna(subset=['price'])
        final_dfs['products'] = final_dfs['type']+' '+final_dfs['products']
        final_dfs['products'] = final_dfs['products'].str.replace('Baseload ','')
        final_dfs = final_dfs.replace(list(self.replace_list.keys()),list(self.replace_list.values()))
        final_dfs = final_dfs.rename(columns={'period':'product_type'})
        return final_dfs

if __name__=='__main__':
    a = Eex()
    a.eex_borderland_parser()
    a.eex_borderland_parser()
    b = a.table_coasele()
    print(b)
    a