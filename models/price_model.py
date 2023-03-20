"""
Разработчик: Д.Девяткин, А.Мартынов, И.Зимин, А.Воротов
Описание:
Этот файл необходим для получения ценовых симуляций.
Входные данные:
model_type: тип модели, используемый при моделировании
n_simulations: количество симуляций
delivery_start: дата, с которой считается DA
delivery_end: дата, до которой считается DA
date_of_pricing: дата, выступающая первым днем симуляций
hub: хаб
date_of_volat: дата на которую берётся волатильность
Выходные данные:
Словарь:
{'futures': симуляции форвардов,
'spot': DA симуляции,
'da': DA на дату date_of_volat,
'atm': стоимость продукта с поставкой от delivery_start до delivery_end,
рассчитанная как среднее значение месячных фьючерсов}
"""

import numpy as np
import pandas as pd
from Models.get_data_from_DB.get_vol_surface import IceVolat
from Models.get_data_from_DB.get_frd_curve import get_frd_advanced
from sqlalchemy import create_engine
import math
from datetime import datetime
import matplotlib.pyplot as plt
import pdb

class PriceModel:

    def __init__(self, model_type: str, n_simulations: int, delivery_start: np.datetime64, delivery_end: np.datetime64,
                 date_of_pricing: np.datetime64 = np.datetime64('today'),
                 hub: str = 'TTF', date_of_volat: np.datetime64 = np.datetime64('2022-05-16'), seed: datetime = None,
                 mult_for_delta: float = 0, mult_for_volat: float = 0, mult_for_theta: int = 0, bom: bool = False):
        """
        Это класс для создания ценовых симуляций по заданным инструментам.

        Methods
        ----------
        simple_model()
            Базовая модель симуляций, написана была Зиминым, переведена в класс и в ССТД Девяткиным.
        gbm_model()
            Geometric Brownian motion model by Petrov.

        Attributes
        ----------
        model_type: str
            тип модели для ценовых симуляций. Пример simple_model, risk_model, two_factor_model ....
            на 23.05.2022 реализована simple_model
        n_simulations: int
            Количество ценовых симуляций (>1)
        delivery_start: np.datetime64
            Дата начала поставки по контракту
        delivery_end: np.datetime64
            Дата окончания поставки по контракту
        date_of_pricing: np.datetime64 = np.datetime64('today')
            Дата симуляций
        hub: str = 'TTF'
            Пункт поставки по контракту ('TTF','THE VTP', 'Austria VTP')
        date_of_volat:np.datetime64=np.datetime64('2022-05-16')
            Дата на которую берётся волатильность
        """

        self.model_type = model_type
        self.n_simulations = n_simulations
        self.date_of_pricing = date_of_pricing
        self.sim_date = date_of_pricing
        self.from_date_m = str(self.date_of_pricing - np.timedelta64(300, 'D'))[:10]
        self.to_date_m = str(self.date_of_pricing)[:10]
        self.hub = hub
        self.delivery_start = delivery_start
        self.delivery_end = delivery_end
        self.date_of_volat = date_of_volat
        self.mult_for_delta = mult_for_delta
        self.seed = seed
        self.mult_for_volat = mult_for_volat
        self.mult_for_theta = mult_for_theta
        self.bom = bom

    def _get_all_forwards(self):

        # Подключение к базе данных
        engine = create_engine('postgresql://postgres:/analytics_base', echo=False)

        # Получение данных по фьючам из базы
        futures = pd.read_sql(
            f"SELECT * FROM f_get_all_forwards('{self.from_date_m}','{self.to_date_m}','{self.hub}','{self.delivery_start}')",
            con=engine).pivot(
            index='date',
            columns='beg_date',
            values='value')
        engine.dispose()

        return futures

    def _get_forward_curve(self):

        # Подключение к базе данных
        engine = create_engine('postgresql://postgres:/analytics_base', echo=False)

        # Получение данных по фьючам из базы
        forward_curve = pd.read_sql(
            f"SELECT * FROM f_get_forward_curve('{self.hub}','{self.date_of_volat}')",
            con=engine)
        engine.dispose()

        return forward_curve

    def _get_bom(self):

        # Подключение к базе данных
        engine = create_engine('postgresql://postgres:/analytics_base', echo=False)

        # Получение данных по фьючам из базы
        bom_data = pd.read_sql(
            f"select * from f_get_any_price_curve('{self.hub}','{'Argus'}','{'BOM'}','{'midpoint'}','{self.from_date_m}','{self.to_date_m}')",
            con=engine)
        engine.dispose()

        return bom_data

    def simple_model(self) -> dict[float: np.ndarray, float: np.ndarray, float: np.ndarray]:
        """
        Базовая модель симуляций, написана была Зиминым, переведена в класс и в ССТД Девяткиным.
        """

        # месяца для расчетов
        month = list(map(lambda x: int(str(np.datetime64(str(self.delivery_start)[:7]) + np.timedelta64(x, 'M'))[5:7]),
                         range(int((np.datetime64(str(self.delivery_end)[:7]) - np.datetime64(
                             str(self.delivery_start)[:7])) / np.timedelta64(1, 'M')))))

        # Получение данных по фьючам из базы
        futures = self._get_all_forwards()

        forward_curve = self._get_forward_curve()

        day_ahead_price = forward_curve[(forward_curve['code'] == 'DA') & (forward_curve['price_type'] == 'midpoint')][
            'value'].to_numpy()[0]

        if self.bom and int(str(self.delivery_start)[:10][-2]):
            futures = futures.to_numpy()[:, :len(month) - 1]
        else:
            futures = futures.to_numpy()[:, :len(month)]

        # get lognormal error
        data = np.log(np.float64(np.transpose(futures[-180:]) / np.transpose(futures[-181:-1])))

        # volatilty parse
        ice_class_data = IceVolat(self.date_of_volat, 'all', 'atm')

        month_range = (np.datetime64(self.delivery_start, 'M') -
                       np.datetime64(self.sim_date, 'M')) / np.timedelta64(1, 'M')

        # get vols
        # vols_0 = ice_class_data.get_table().to_numpy()[:, int(date_range) - 1:]
        vols_0 = ice_class_data.get_table().to_numpy()
        vols_0 = vols_0 + vols_0 * self.mult_for_volat
        # extend last known volatility for 50 days
        vols_0 = np.concatenate((vols_0[0], np.repeat(vols_0[0][-1], 50)), axis=0)

        if int(str(self.delivery_start)[:10][-2:]) == 1:
            vols_0 = vols_0[int(month_range) - 1:]
        else:
            vols_0 = vols_0[int(month_range):]
        vols_0 = vols_0[:futures.shape[1]]
        '''
        if len(vols_0) < futures.shape[1]:
            vols_0 = vols_0[:futures.shape[1]]
        else:
            vols_0 = vols_0[:futures.shape[1]]
        '''
        # if vols_0.shape
        vols = vols_0 / (365 ** 0.5)

        if self.bom and int(str(self.delivery_start)[:10][-2]):
            bom_data = self._get_bom()['value'].to_numpy()
            bom_np = np.log(np.float64(bom_data[-180:] / bom_data[-181:-1]))
            bom_vol = np.std(bom_np, axis=0)
            vols = np.append(bom_vol, vols)
            data = np.append(data, np.array([bom_np]), axis=0)
        else:
            pass

        cov_matrix = np.float64(np.dot(np.diag(vols), np.dot(np.corrcoef(data), np.diag(vols))))

        # range for sims
        dates_range = (self.delivery_end - self.sim_date) / np.timedelta64(1, 'D') - self.mult_for_theta

        if self.seed is None:
            multiv_data = np.random.multivariate_normal(mean=np.float64((vols ** 2) * (-1 / 2)), cov=cov_matrix,
                                                        size=[self.n_simulations, int(dates_range)])
        else:
            rng = np.random.default_rng(int(self.seed.strftime('%Y%m%d')))
            multiv_data = rng.multivariate_normal(mean=np.float64((vols ** 2) * (-1 / 2)), cov=cov_matrix,
                                                  size=[self.n_simulations, int(dates_range)])

        # cumulative sum
        cumulat = np.cumsum(multiv_data, axis=1)
        # цены по форвардам к использованию
        # print(np.mean(np.float64(futures[-1])))

        if self.bom and int(str(self.delivery_start)[:10][-2]):
            prices = (np.append(bom_data[-1], np.float64(futures[-1])) + self.mult_for_delta) * np.exp(cumulat)
            atm_price = np.mean(np.append(bom_data[-1], np.float64(futures[-1])))
        else:
            prices = (np.float64(futures[-1]) + self.mult_for_delta) * np.exp(cumulat)
            atm_price = np.mean(np.float64(futures[-1]))

        # получаем Day-Ahead котировки
        if self.bom and int(str(self.delivery_start)[:10][-2]):
            for_da = prices[:, :int((self.delivery_end - self.sim_date - self.mult_for_theta) / np.timedelta64(1, 'D'))]
            days_of_index = np.append(np.array([self.delivery_start]), np.array(list(map(
                lambda x: np.datetime64(self.delivery_start, 'M') + np.timedelta64(1 * x, 'M') + np.timedelta64(0, 'D'),
                range(1, math.ceil(len(month) / 1) + 1)))))
        else:
            for_da = prices[:,
                     int((self.delivery_start - self.sim_date - self.mult_for_theta) / np.timedelta64(1, 'D')):int(
                         (self.delivery_end - self.sim_date - self.mult_for_theta) / np.timedelta64(1, 'D'))]
            days_of_index = list(map(
                lambda x: np.datetime64(self.delivery_start, 'M') + np.timedelta64(1 * x, 'M') + np.timedelta64(0, 'D'),
                range(math.ceil(len(month) / 1) + 1)))

        days_of_index_repeated = np.cumsum(
            list(map(lambda x: (days_of_index[x + 1] - days_of_index[x]) / np.timedelta64(1, 'D'),
                     range(len(days_of_index) - 1))))

        days_of_index_repeated = np.sort(np.append(days_of_index_repeated, np.array([0]), axis=0))

        spot = list(
            map(lambda x: for_da[:, int(days_of_index_repeated[x]):int(days_of_index_repeated[x + 1]), x],
                range(len(month))))

        spot_prices_sim = []
        for i in range(len(spot[0])):
            temp = []
            for j in range(len(spot)):
                temp.append(spot[j][i])
            spot_prices_sim.append([item for sublist in temp for item in sublist])

        # np.savez('/mnt/teamdocs_ns/TRD_RSK/sims_2022-07-27_no_seed.npz', prices)
        # np.savez('/mnt/teamdocs_ns/TRD_RSK/sims_2022-07-27_no_seed_da.npz', np.array(spot_prices_sim))

        if self.bom and int(str(self.delivery_start)[:10][-2]):
            prices = prices[:, :, 1:]
        else:
            prices = prices

        return {'futures': prices, 'spot': np.array(spot_prices_sim), 'da': day_ahead_price, 'atm': atm_price}


    def gbm_model(self, strike: float, show_plot: bool= False) -> np.ndarray:
        """
        Calculation of contract cost by Geometric Brownian motion model.
        GBM model numerical equation:
        S_i = S_{i-1} * exp(mu - volatility ** 2 / 2) * dt +
        volatility * sqrt(dt) * z[i]
        where:
        S_i = price of the contract for time moment t = i
        mu = price drift value, mu = 0
        volatility = implied volatility
        sqrt(dt) * z[i] - Wiener process approximation
        z - normal distributed random variables array
        """
        # get futures data from the DB
        forward_curve = get_frd_advanced(self.date_of_pricing,
                                         beg_date= self.delivery_start.item().
                                         strftime('%Y-%m-%d'))
        # define model params
        # price_0 - last known forward value, GBM simulations start with that value
        # extract single float value from pd.DataFrame object
        price_0 = forward_curve['forward']['value'].iloc[0]
        # we consider mu = 0, see full documentation
        mu = 0
        # --- get volatility ----
        # take into account that if date range between delivery start and
        # delivery and is bigger than 1 month,
        # we need to calculate another volatility for following months.
        volatility = IceVolat(self.date_of_pricing,'all','atm').interpolation()
        print(f"volatility = {volatility}")
        cur_day = self.date_of_pricing
        months = {}
        while self.delivery_end > cur_day:
            cur_month = cur_day.item().strftime('%Y-%m-01')
            if cur_month in months.keys():
                months[cur_month] += 1
            else:
                months[cur_month] = 1
            cur_day += np.timedelta64(1, 'D')

        volatilities = []
        for key in months.keys():
            if key in volatility.keys():
                volatilities.append(volatility[key](strike))

        months[list(months.keys())[1]] += months[list(months.keys())[0]]
        months.pop(list(months.keys())[0])
        #print("volatilities = {0}".format(volatilities))
        volatilities = np.repeat(volatilities, list(months.values()))
        # number of days between day of pricing and
        date_range = int((self.delivery_end -
                          self.date_of_pricing) / np.timedelta64(1, 'D'))
        #print('date_range = {0}'.format(date_range))
        # 1 simulation value stands for one day
        t: int = 1
        # total days in year
        m: int = 365
        dt = t / m
        # list of random variables
        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(self.n_simulations)))
        z = np.random.standard_normal(size= (date_range + 1, self.n_simulations))
        # make mean = 0 and std = 1 of the generated distribution
        z -= z.mean()
        z /= z.std()
        # initial points of simulations
        s = np.zeros((date_range, self.n_simulations))
        s[0] = price_0
        # GBM equation
        for i in range(1, date_range):
            s[i] = s[i - 1] * np.exp((mu - volatilities[i-1] ** 2 / 2) * dt +
                                     volatilities[i-1] * math.sqrt(dt) * z[i])

        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.title(f'{self.n_simulations} Simulated Stocks Paths', fontweight="bold", pad=20)
            plt.xlabel('Steps')
            plt.ylabel('Price')
            plt.plot(s[:, :50])
            plt.show()
            plt.figure(figsize=(10, 6))
            plt.hist(s[-1], bins=10)
            plt.title('Histogram of Final Value of the Stock', fontweight="bold", pad=20)
            plt.axvline(s[-1].mean(), c= 'b')
            plt.show()

        # filter simulation data
        days_till_start = int((self.delivery_start - self.date_of_pricing) / np.timedelta64(1, 'D'))
        s = s[days_till_start:]

        return  s


if __name__ == '__main__':
    import time
    t_0 = time.time()
    a = PriceModel(model_type='simple',
                   n_simulations= 10000,
                   delivery_start= np.datetime64('2023-03-01'),
                   delivery_end= np.datetime64('2024-01-01'),
                   date_of_pricing= np.datetime64('2023-02-22'),
                   hub= 'TTF',
                   date_of_volat= np.datetime64('2023-02-21'))
    a1 = a.simple_model()
    pdb.set_trace()
    # print(a['spot'][:31].max())
    # from Models.collar import Collar
    #sims = a.gbm_model(strike= 220)
    #put = Collar(simulations= sims, opt_type= 'put', strike= 220)
    #put.calculate_price()
    #sims = a.gbm_model(strike= 270)
    #call = Collar(simulations= sims, opt_type= 'call', strike= 270)
    #call.calculate_price()
    #print("total time of running  = {0}".format(time.time()-t_0))

