import sys
import os
import pandas as pd
sys.path.insert(1, os.path.join(sys.path[0], '../GRTgaz'))
from grtgaz_loader import *


additional_columns['source'] = 'Fluxys'


if __name__ == '__main__':
    data_types = ('domestic', 'interconnection')
    data_type = data_types[0]
    folder = 'parsed_data'
    
    
    result = pd.read_csv(f'{folder}/all_Fluxys_{data_type}.csv', index_col=False)
    GRTgazLoader().insert_grtgaz(df_fs=result)
    
