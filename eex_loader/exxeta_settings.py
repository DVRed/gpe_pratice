from datetime import datetime
from pathlib import Path

# наименования хабов в форме, пригодной для их удобного парсинга из экзитовских
# строк - спецификаций сделок [газ]
HUBS = [
    'CEGH VTP', 'GASPOOL', 'NCG', 'TTF', 'AOC', 'PSV', 'THE', 'CZ GAS', 'Czech Virtual Point',
    r'DK\(GTF\)', 'Denmark ETF', r'\bPEG\b', 'TRS', 'Hungarian Virtual Point', 'IBP', 'NBP',
    'ZTP', 'ZEEBRUGGE', 'SLOVAK VTP', 'Slovak Virtual Point'
]

# наименования стран в форме, пригодной для их удобного парсинга из экзитовских
# строк - спецификаций сделок [электричество]
COUNTRIES = [
    'Austria', 'Belgium', 'France', 'French', 'Bulgarian', 'Germany', 'German', 'Czech',
    'Swiss', 'Holland', 'Hungary', 'Italy', 'Nordic', 'Poland', 'Romania', 'Serbian',
    'Slovenian', 'Spain', r'UK\b'
]

# наименования тикеров в форме, пригодной для их удобного парсинга из экзитовских
# строк - спецификаций сделок [уголь]
COAL_DELIVERY_POINTS = ['API 2', 'API 4', 'API 5', 'API 8', 'NEWC', 'ICI 4']

# наименования тикеров в форме, пригодной для их удобного парсинга из экзитовских
# строк - спецификаций сделок [эмиссионные квоты]
EMISSIONS = ['CER', 'EUA', 'ERU', 'UKA']

# наименования тикров в форме, пригодной для их удобного парсинга из экзитовских
# строк - спецификаций сделок [СПГ]
LNG_DELIVERY_POINTS = ['JKM']

# наименования товаров в соответствие от пунктов-регионов поставки
DELIVERY_POINTS_ASSOCIATIONS = [
    (HUBS, 'Natural Gas'), (COUNTRIES, 'Electricity'), (COAL_DELIVERY_POINTS, 'Coal'),
    (EMISSIONS, 'CO2'), (LNG_DELIVERY_POINTS, 'LNG')
]

# словарь для преобразования названий пунктов поставки к виду, соответствующему базе данных
DELIVERY_POINTS_CORRECT = {'CEGH VTP': 'Austria VTP', 'CZ GAS': 'Czech VTP', 'Czech Virtual Point': 'Czech VTP',
                           'Hungarian Virtual Point': 'MGP', 'SLOVAK VTP': 'Slovak VTP', 'PEG': 'Peg Nord',
                           'Slovak Virtual Point': 'Slovak VTP', 'Zeebrugge': 'ZEE', 'THE': 'THE VTP',
                           'GASPOOL': 'Gaspool', 'French': 'France', 'German': 'Germany', 'DK(GTF)': 'Denmark GTF'}

# соответствие типов пунктов поставки различным "коммодити"
DELIVERY_POINT_TYPES = {'Natural Gas': 'hub', 'Electricity': 'country', 'Coal': 'coal_region',
                        'CO2': 'emissions_region', 'LNG': 'lng_terminal'}

# список бирж, встречающихся в спецификациях экзитовских сделок.
# Используется для отнесения сделки к одной из платформ: OTC или Exchange
EXCHANGES = [
    'CME', 'ICE', 'EEX', 'LCH', 'SGX', 'ICE ENDEX', 'PEGAS', 'ECX', 'EFP', 'PXE', 'NDAQ', 'MEFF'
]

# соответствие единиц измерения пунктам поставки
UNITS = {'TTF': 'MWh', 'NCG': 'MWh', 'Gaspool': 'MWh', 'THE VTP': 'MWh',
         'Austria VTP': 'MWh', 'Czech VTP': 'MWh', 'Slovak VTP': 'MWh',
         'Peg Nord': 'MWh', 'AOC': 'MWh', 'Denmark ETF': 'MWh', 'Denmark GTF': 'MWh',
         'NBP': 'therm', 'IBP': 'therm', 'ZEE': 'therm',
         'ZTP': 'MWh', 'MGP': 'MWh', 'PSV': 'MWh', 'NCG (Lo-Cal)': 'MWh',
         'Gaspool (Lo-Cal)': 'MWh', 'THE VTP (Lo-Cal)': 'MWh', 'ZTP (Lo-Cal)': 'MWh',
         'Austria': 'specific', 'Belgium': 'specific', 'France': 'specific', 'French': 'specific',
         'Bulgarian': 'specific', 'Germany': 'specific', 'German': 'specific', 'Czech': 'specific',
         'Swiss': 'specific', 'Holland': 'specific', 'Hungary': 'specific', 'Italy': 'specific',
         'Nordic': 'specific', 'Poland': 'specific', 'Romania': 'specific', 'Serbian': 'specific',
         'Slovenian': 'specific', 'Spain': 'specific', 'UK': 'specific',
         'API 2': 'MT', 'API 4': 'MT', 'API 5': 'MT', 'API 8': 'MT', 'NEWC': 'MT', 'ICI 4': 'MT',
         'CER': 'MT', 'EUA': 'MT', 'ERU': 'MT', 'UKA': 'MT', 'JKM': 'mmbtu'
         }

# соответствие валют пунктам поставки
CURRENCIES = {'TTF': 'EUR', 'NCG': 'EUR', 'Gaspool': 'EUR', 'THE VTP': 'EUR',
              'Austria VTP': 'EUR', 'Czech VTP': 'EUR', 'Slovak VTP': 'EUR',
              'Peg Nord': 'EUR', 'AOC': 'EUR', 'Denmark ETF': 'EUR', 'Denmark GTF': 'EUR',
              'NBP': 'GBP', 'IBP': 'GBP', 'ZEE': 'GBP', 'ZTP': 'EUR',
              'MGP': 'EUR', 'PSV': 'EUR', 'NCG (Lo-Cal)': 'EUR',
              'Gaspool (Lo-Cal)': 'EUR', 'THE VTP (Lo-Cal)': 'EUR', 'ZTP (Lo-Cal)': 'EUR',
              'Austria': 'EUR', 'Belgium': 'EUR', 'France': 'EUR', 'French': 'EUR',
              'Bulgarian': 'EUR', 'Germany': 'EUR', 'German': 'EUR', 'Czech': 'EUR',
              'Swiss': 'EUR', 'Holland': 'EUR', 'Hungary': 'EUR', 'Italy': 'EUR',
              'Nordic': 'EUR', 'Poland': 'EUR', 'Romania': 'EUR', 'Serbian': 'EUR',
              'Slovenian': 'EUR', 'Spain': 'EUR', 'UK': 'GBP', 'API 2': 'USD', 'API 4': 'USD', 'API 5': 'USD',
              'API 8': 'USD', 'NEWC': 'USD', 'ICI 4': 'USD',
              'CER': 'EUR', 'EUA': 'EUR', 'ERU': 'EUR', 'UKA': 'GBP', 'JKM': 'USD'}

# коллекция для парсинга продуктов из спецификации сделки
PARSER_PATTERNS = (
    {'pattern': 'Prompt', 'instrument': r'prompt\s(.+)'},
    {'pattern': r'BOM\s(\bBOM\b)|\bBOW\b|Saturday|Sunday|W\WEND'},
    {'pattern': r'WD\d{2}\s\d{4}'},
    {'pattern': r'Week\b', 'instrument': r'Week\s(W\d{2}\s\d{4})'},
    {'pattern': r'Day D|Days D', 'instrument': r'\s(D\d{2}\W\d{2}\W\d{4})'},
    {'pattern':
         'Weekend', 'instrument': r'WE\d{2}\W\d{2}\W\d{2}\W\d{4}'},
    {'pattern': r'WE\d{2}\W\d{2}\W\d{2}\W\d{4}'},
    {'pattern': r'\w{5,6}\s\w{2}\W\w{3}\d{2}'},
    {'pattern': r'[A-Z]{3}\d{2}'},
    {'pattern': r'\b[Qq]\d\D\d{2}\b'},
    {'pattern': r'[Win]{3}\s\d{4}\W\d{4}'},
    {'pattern': r'[Sum]{3}\s\d{4}'},
    {'pattern': r'[Year]{4,5}\s(\d{4})'},
    {'pattern': r'[Cal]{3}\s[Years]{4,5}\s(\d{4})'},
    {'pattern': r'[Yyears0-9]{4,6}\s(\d{4})\W(\d{4})'},
    {'pattern': r'[Gas]+\s[Years]+\s\d{2}\W\d{2}'},
    {'pattern': r'SPOT ph3'},
)

# коллекция для парсинга специфичных дополнений к продуктам
PARSER_PATTERNS_SPECIFIC = (
    {'pattern': 'base|peak|offpeak|off-peak|spark',
     'instrument': r'base|peak|offpeak|off-peak|spark\sspread|clean\sspark\sspread'},
    {'pattern': r'\(*WD\s\d.*\)*', 'instrument': r'(WD\s[0-9\+]+)\s?'},
    {'pattern': r'Block\s[0-9\+]+', 'instrument': r'Block\s[0-9\+]+'},
    {'pattern': r'Anon\sBlock|Anon'}
)

# выходные дни, в которые отсутствуют торги. Используется для корректировки периодов поставки
HOLIDAY = [
    datetime(2020, 1, 1), datetime(2020, 4, 10), datetime(2020, 5, 8),
    datetime(2020, 5, 25), datetime(2020, 8, 31), datetime(2020, 12, 28),
    datetime(2021, 1, 1), datetime(2021, 4, 2), datetime(2021, 4, 5),
    datetime(2021, 5, 3), datetime(2021, 5, 31), datetime(2021, 8, 30),
    datetime(2021, 12, 27), datetime(2021, 12, 28), datetime(2022, 1, 3),
    datetime(2022, 4, 15), datetime(2022, 4, 18), datetime(2022, 5, 2),
    datetime(2022, 6, 2), datetime(2022, 7, 3), datetime(2022, 8, 29),
    datetime(2022, 12, 26), datetime(2022, 12, 27), datetime(2023, 1, 2)
]

BASE_DIR = Path.cwd()

# путь к папке с данными экзиты, которые необходимо загрузить в базу
# DATA_DIR = Path.home() / 'Desktop' / 'exxeta'
DATA_DIR = Path('/', 'mnt', 'teamdocs_ns', 'TRD_Exchange', '')

# наименование колонок с основной информацией
MAIN_COLUMNS = ['Date/Time', 'Contract', 'Qty', 'Price']

# наименование колонок с дополнительной информацией.
# Заполняются в процессе предобработки файла с сырыми данными
EXTRA_COLUMNS = ['delivery_point_1', 'delivery_point_2', 'instrument_1', 'instrument_2',
                 'delivery_start_1', 'delivery_start_2', 'delivery_end_1', 'delivery_end_2',
                 'delivery_hours_1', 'delivery_hours_2']

# различные сборки конфигураций для подключения к базе данных
# home computer
# DB_CONFIG = {
#     'drivername': 'postgresql',
#     'host': 'localhost',
#     'port': '5432',
#     'username': 'postgres',
#     'password': '123',
#     'database': 'postgres'
# }

# psdev08
# DB_CONFIG = {
#     'drivername': 'postgresql',
#     'host': '127.0.0.1',
#     'port': '5432',
#     'username': 'psdevAdmin',
#     'password': 'q12345Q!',
#     'database': 'analytics_base'
# }

# psdev10
DB_CONFIG = {
    'drivername': 'postgresql',
    'host': '10.18.22.217',
    'port': '5432',
    'username': 'postgres',
    'password': '!QAZ2wsx',
    'database': 'analytics_base'
}

# psdev10 test
DB_CONFIG_TEST = {
    'drivername': 'postgresql',
    'host': '10.18.22.217',
    'port': '5432',
    'username': 'postgres',
    'password': '!QAZ2wsx',
    'database': 'analytics_base_test'
}
