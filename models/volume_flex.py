"""
Разработчик: Д.Девяткин, А.Мартынов
Описание:
Класс для расчета объёмной гибкости, страйки по которой можно
записать в формате (x,y,z) по хабам: TTF, THE VTP, Austria VTP.
Поставка может начинаться и заканчиваться только в начале и
в конце месяца, соответственно.
Входные данные:
delivery_start: np.datetime64 Дата начала поставки по контракту
delivery_end: np.datetime64 Дата окончания поставки по контракту
delivery_point: str Пункт поставки. Возможные варианты - TTF, THE VTP, Austria VTP
date_of_pricing: np.datetime64 Дата оценки
(влияет на симуляции и подтягивание реальных данных из базы)
date_of_volat: np.datetime64 Дата волатильности (влияет на симуляции)
indexes: list
    Принимает список только целых значений
    Индексы в формате:
    1 - сколько инструментов берем
    2 - лаг
    3 - сколько месяцев усреднения
    4 - сколько месяцев действует индекс (можно скипнуть)
    Пример:indexes = [1,0,1,1] - классический Month-Ahead Index

Выходные данные:
-------
'err': ошибка,
'premium': премия к MWh,
'fee_0': стоимость контракта,
'xes': ряд иксов,
'yes': ряд игреков,
'expected_withdrawal': мартица ожидаемых отборов,
'calc_time': время вычислений
"""

import pdb
import matplotlib.pyplot as plt
import numpy as np
from Models.price_model import PriceModel
from Models.index_counter.index_count_class import IndexCounter
import time
from Models.lsmc.lsmc_model import lsmc
from datetime import datetime
from Models.post_data_to_DB.calc_results_loader import CalcResultLoader


class VolumeFlex:

    def __init__(self, delivery_start: np.datetime64, delivery_end: np.datetime64, delivery_point: str,
                 date_of_pricing: np.datetime64, date_of_volat: np.datetime64, max_acq: int, min_aqc: int,
                 max_dcq: int, min_dcq: int, n_sims: int = 2500, fix_price: float = None,
                 indexes: list = [1, 0, 1], price_model_type: str = 'simple',comment='simple_calc'):

        """
        Класс для расчета объёмной гибкости, страйки по которой можно записать в формате (x,y,z) по хабам: TTF, THE VTP, Austria VTP.

        Поставка может начинаться и заканчиваться только в начале и в конце месяца, соответственно.

        Attributes:
        ----------
        delivery_start: np.datetime64
            Дата начала поставки по контракту
        delivery_end: np.datetime64
            Дата окончания поставки по контракту
        delivery_point: str
            Пункт поставки. Возможные варианты - TTF, THE VTP, Austria VTP
        date_of_pricing: np.datetime64
            Дата оценки (влияет на симуляции и подтягивание реальных данных из базы)
        date_of_volat: np.datetime64
            Дата волатильности (влияет на симуляции)
        indexes: list
            Принимает список только целых значений
            Индексы в формате:
            1 - сколько инструментов берем
            2 - лаг
            3 - сколько месяцев усреднения
            4 - сколько месяцев действует индекс (можно скипнуть)
            Пример:indexes = [1,0,1,1] - классический Month-Ahead Index

        Methods:
        -------
        _price_model(n_sims)
            Вызывает ценовую модель
        _index_counter(price_sims)
            Вызывает расчет индекса
        volume_flex_counter(max_acq, min_aqc, max_dcq, min_dcq, n_sims: int = 5000)
            Вызывает 2 предыдущих метода и LSMC. Считает гибкость и подгоняет премию.

        P.S. Потенциально можно и интрументы с кастомным сроком поставки, но я не тестил.
        """

        self._delivery_start = delivery_start
        self._delivery_end = delivery_end
        self._delivery_point = delivery_point
        self._date_of_pricing = date_of_pricing
        self._date_of_volat = date_of_volat
        self._max_acq = max_acq
        self._min_acq = min_aqc
        self._max_dcq = max_dcq
        self._min_dcq = min_dcq
        self._n_sims = n_sims
        self._fix_price = fix_price
        self._price_model_type = price_model_type
        self._time_start = datetime.now()
        self.comment = comment
        if 3 <= len(indexes) <= 4:
            self._indexes = indexes
        else:
            raise ValueError(indexes)

    def _price_model(self, mult_for_delta: float = 0, date_of_pricing: np.datetime64 = None, mult_for_volat: float = 0,
                     mult_for_theta: int = 0):
        """
        Симулирует цены на газ.

        Attributes:
        ----------
        n_sims: int
            Количество симуляций
            БОЛЬШЕ 7500 НЕ СТАВИТЬ. РАСЧЕТЫ ПАДАЮТ.
        Массив на выходе имеет размерность к-во симуляций Х к-во месячных инструментов Х к-во дней.
        """

        if date_of_pricing is None:
            date_of_pricing = self._date_of_pricing
        else:
            pass

        simulations = PriceModel(model_type=self._price_model_type,
                                 n_simulations=self._n_sims,
                                 delivery_start=self._delivery_start,
                                 delivery_end=self._delivery_end,
                                 date_of_pricing=date_of_pricing,
                                 hub=self._delivery_point,
                                 date_of_volat=self._date_of_volat,
                                 mult_for_delta=mult_for_delta,
                                 mult_for_volat=mult_for_volat,
                                 mult_for_theta=mult_for_theta,bom=False).simple_model()
        return simulations

    def _index_counter(self, price_sims: np.array, sim_date: np.datetime64 = None, mult_for_theta: int = 0):
        """
        Расчитывает индекс/страйки по оцениваемому контракту.

        Attributes:
        ----------
        price_sims: np.array
            Ценовые симуляции (с выходными).
            Массив должен иметь на входе размерность к-во симуляций Х к-во месячных инструментов Х к-во дней
        На выходе получаем размерность размерность к-во симуляций Х к-во дней поставки
        """

        if sim_date is None:
            sim_date = self._date_of_pricing
        else:
            sim_date = sim_date

        strikes = IndexCounter().index_count(contract_start=self._delivery_start,
                                             contract_end=self._delivery_end,
                                             simulation_date=sim_date,
                                             indexes=self._indexes,
                                             data=price_sims, mult_for_theta=mult_for_theta)

        return strikes

    def _lsmc_counter(self, delivery_days: int, strikes: np.array, price_simulations: np.array, fee: float = 0,
                      mult_for_greek: float = None):

        if mult_for_greek is None:
            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._min_acq,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations,
                     strike_paths=np.array(strikes),
                     premium=fee)
        else:
            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._max_acq * mult_for_greek,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations,
                     strike_paths=np.array(strikes),
                     premium=fee)

        return {'option_cost': np.mean(a[0]), 'premium': a[0], 'decision_matrix': a[1]}

    def _derivative_function(self, atm_diff: np.array, premium_fee_0: np.array, x):
        '''
        4th power polynomial equation
        '''
        plft = np.polyfit(atm_diff, premium_fee_0, 4)
        return plft[0] * 4 * (x ** 3) + plft[1] * 3 * (x ** 2) + plft[2] * 2 * (x ** 1) + plft[3]

    def count_premium(self, n_sims: int = 2500, custom_sims: np.array = None, custom_index: np.array = None,
                      fix_price: float = None):

        """
        Считает гибкость и подгоняет премию.

        Attributes:
        ----------
        max_acq
            Максимальный суммарный отбор по контракту
        min_aqc
            Минимальный суммарный отбор по контракту
        max_dcq
            Максимальный суточный отбор по контракту
        min_dcq
            Минимальный суточный отбор по контракту
        n_sims: int = 5000, optional
            Количество симуляций
            БОЛЬШЕ 7500 НЕ СТАВИТЬ!!! РАСЧЕТЫ ПАДАЮТ.
        custom_sims: np.array = None

        custom_index: np.array = None

        fix_price:float=None
        """

        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        if custom_sims is None:
            price_simulations = self._price_model()
        else:
            price_simulations = custom_sims

        if fix_price is None:
            if custom_index is None:
                strikes = self._index_counter(price_simulations['futures'])
            else:
                strikes = custom_index
        else:
            strikes = np.resize(np.array([fix_price]), (n_sims, delivery_days))

        xes = []
        yes = []
        prems = []

        for i in range(10):

            prem = price_simulations['da'] * 0.1 * i
            lsmc_result = self._lsmc_counter(delivery_days=delivery_days,
                                             price_simulations=price_simulations['spot'],
                                             strikes=np.array(strikes), fee=prem)

            if prem == 0:
                prems.append(lsmc_result['premium'])
            else:
                pass
            if lsmc_result['option_cost'] < 0:

                xes.append(prem)
                yes.append(lsmc_result['option_cost'])

                break
            else:
                xes.append(prem)
                yes.append(lsmc_result['option_cost'])
            del (lsmc_result)

        print(yes)
        print(xes)
        premia = -(yes[-1] * (xes[-2] - xes[-1]) / (yes[-2] - yes[-1])) + xes[-1]

        lsmc_result_0 = self._lsmc_counter(delivery_days=delivery_days, price_simulations=price_simulations['spot'],
                                           strikes=np.array(strikes), fee=premia)
        # print(lsmc_result_0['option_cost'])

        # calculating calculation time
        self._calc_time = (datetime.now() - self._time_start).seconds

        # Student test by Martynov - 1.645
        avg_error = np.std(prems) / (np.mean(prems) * (n_sims ** 0.5)) * 1.645

        # price_spec setup
        if self._fix_price is None:
            price_spec = str(self._indexes)
        else:
            price_spec = str(self._fix_price)

        # loading data to bd
        self._load_to_bd(status=True, type_of_error=None, contract_value_npv=yes[0], premium=premia,
                         avg_error=avg_error,price_spec=price_spec)
        return {'err': np.std(prems) / (np.mean(prems) * (n_sims ** 0.5)) * 1.645,
                'premium': premia,
                'fee_0': yes[0],
                'expected_withdrawal': lsmc_result_0['decision_matrix'],
                'calc_time': self._calc_time, 'strike_prices':np.array(strikes)}

    def count_delta(self, show_plot: bool = False, n_path: int = 9):
        """
        Func for delta calculation
        :param show_plot:bool=False: is need when
        :return: if 'curve' is selected output will be pd.DataFrame else np.float64 value
        """

        # simulate forward curve
        price_simulations = self._price_model()
        # atm price for next valuations
        atm_price = price_simulations['atm']
        # counting withdrawal prices via IndexCounter or fix
        if self._fix_price is None:
            # via IndexCounter
            strikes = self._index_counter(price_simulations['futures'])
        else:
            # creating array of arrays that contains withdrawal price (fix_price)
            delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))
            strikes = np.resize(np.array([self._fix_price]), (self._n_sims, delivery_days))

        # counting premium for contract
        prem_count = self.count_premium(custom_sims=price_simulations, custom_index=strikes)

        # no comments
        fee_0 = prem_count['fee_0']
        # working with decision matrix to create withdrawal paths
        withdrawal_paths = prem_count['expected_withdrawal'] * 1
        # transform bool matrix to matrix of daily withdrawals
        withdrawal_paths[withdrawal_paths == 1] = self._max_dcq
        withdrawal_paths[withdrawal_paths == 0] = self._min_dcq
        expected_withdrawal = np.mean(np.mean(withdrawal_paths, axis=1))
        # cumulative withdrawals
        withdrawals = np.cumsum(withdrawal_paths, axis=1)
        # counting average strike price of withdrawals
        for_w = ((prem_count['expected_withdrawal'] * 1) * strikes)
        for_w[for_w == 0.] = np.nan
        average_strike_price = np.nanmean(np.nanmean(for_w, axis=1))
        # cumulative withdrawals
        premia_with_fee0 = prem_count['premium']

        """
        Taken from price_model gbm_model
        Respect to M.Petrov
        (C) DTO&AT, Quantitative division
        """
        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.title(f'{n_path} Simulated Withdrawal Paths', fontweight="bold", pad=20)
            # plt.title(f'2500 Simulated Withdrawal Paths', fontweight="bold", pad=20)
            plt.xlabel('Day')
            plt.ylabel('Volume')
            # plt.plot(delta['withdrawal_paths'].T,linewidth=1)
            plt.plot(withdrawals.T[:, n_path], linewidth=1)
            plt.show()

        return {'tcq': self._max_acq, 'mtcq': self._min_acq, 'dcq': self._max_dcq,
                'mdcq': self._min_dcq, 'idx': str(self._indexes),
                'beg_date': self._delivery_start, 'end_date': self._delivery_end, 'premium': premia_with_fee0,
                'delta_hedge_size': (expected_withdrawal * average_strike_price) / self._max_dcq / atm_price,
                'withdrawal_paths': withdrawal_paths, 'avg_withdrawals': expected_withdrawal,
                'avg_strike_price': average_strike_price, 'fee_0': fee_0, 'strike_prices':for_w}

    def count_theta(self):

        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        diff_with_atm = []
        time_before_exp = (self._delivery_start - self._date_of_pricing) / np.timedelta64(1, 'D')
        step = np.floor(time_before_exp / 30)
        premia_with_fee0 = []
        date_and_time = []
        for i in range(0, int(time_before_exp) - 30, int(step)):

            print('i:', i)
            date = self._date_of_pricing + np.timedelta64(i, 'D')

            price_simulations = self._price_model(mult_for_theta=i)

            if self._fix_price is None:
                strikes = self._index_counter(price_simulations['futures'], date, mult_for_theta=i)
            else:
                strikes = np.resize(np.array([self._fix_price]), (self._n_sims, delivery_days))

            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._min_acq,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations['spot'],
                     strike_paths=np.array(strikes),
                     premium=0)

            diff_with_atm.append(i)
            premia_with_fee0.append(np.mean(a[0]))
            date_and_time.append(self._date_of_pricing + np.timedelta64(i, 'D'))

        theta_der_values = []
        for i in diff_with_atm:
            theta_der_values.append(self._derivative_function(diff_with_atm, premia_with_fee0, i))

        return {'tcq': np.repeat(self._max_acq, len(theta_der_values)),
                'mtcq': np.repeat(self._min_acq, len(theta_der_values)),
                'dcq': np.repeat(self._max_dcq, len(theta_der_values)),
                'mdcq': np.repeat(self._min_dcq, len(theta_der_values)),
                'idx': np.repeat(str(self._indexes), len(theta_der_values)),
                'beg_date': np.repeat(self._delivery_start, len(theta_der_values)),
                'end_date': np.repeat(self._delivery_end, len(theta_der_values)),
                'time_from_sim_date': date_and_time, 'premium': premia_with_fee0, 'theta_values': theta_der_values}

    def count_vega(self):

        vol_range = [-0.5, -0.4, -0.3, -0.25, -0.2, -0.15, -0.1, -0.08, -0.07, -0.06, -0.05,
                     -0.04, -0.03, -0.02, -0.01,
                     0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        diff_with_atm = []
        premia_with_fee0 = []
        for i in vol_range:
            # print('i:', i)
            price_simulations = self._price_model(mult_for_volat=i)

            if self._fix_price is None:
                strikes = self._index_counter(price_simulations['futures'], self._date_of_pricing)
            else:
                strikes = np.resize(np.array([self._fix_price]), (self._n_sims, delivery_days))

            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._min_acq,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations['spot'],
                     strike_paths=np.array(strikes),
                     premium=0)
            diff_with_atm.append(i)
            premia_with_fee0.append(np.mean(a[0]))

        vega_der_values = []
        for i in diff_with_atm:
            vega_der_values.append(self._derivative_function(diff_with_atm, premia_with_fee0, i))

        return {'tcq': np.repeat(self._max_acq, len(vega_der_values)),
                'mtcq': np.repeat(self._min_acq, len(vega_der_values)),
                'dcq': np.repeat(self._max_dcq, len(vega_der_values)),
                'mdcq': np.repeat(self._min_dcq, len(vega_der_values)),
                'idx': np.repeat(str(self._indexes), len(vega_der_values)),
                'beg_date': np.repeat(self._delivery_start, len(vega_der_values)),
                'end_date': np.repeat(self._delivery_end, len(vega_der_values)),
                'vol_delta': diff_with_atm, 'premium': premia_with_fee0, 'vega_values': vega_der_values}

    def count_maq_greek(self, n_sims: int = 2500, custom_sims: np.array = None, custom_index: np.array = None,
                        fix_price: float = None):

        """
        Считает гибкость и подгоняет премию.

        Attributes:
        ----------
        max_acq
            Максимальный суммарный отбор по контракту
        min_aqc
            Минимальный суммарный отбор по контракту
        max_dcq
            Максимальный суточный отбор по контракту
        min_dcq
            Минимальный суточный отбор по контракту
        n_sims: int = 5000, optional
            Количество симуляций
            БОЛЬШЕ 7500 НЕ СТАВИТЬ!!! РАСЧЕТЫ ПАДАЮТ.
        custom_sims: np.array = None

        custom_index: np.array = None

        fix_price:float=None
        """

        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        if custom_sims is None:
            price_simulations = self._price_model()
        else:
            price_simulations = custom_sims

        if fix_price is None:
            if custom_index is None:
                strikes = self._index_counter(price_simulations['futures'])
            else:
                strikes = custom_index
        else:
            strikes = np.resize(np.array([fix_price]), (n_sims, delivery_days))

        fee_0 = []
        maq_pc = []
        premium = []
        for j in range(20):
            xes = []
            yes = []
            prems = []
            print(j)
            for i in range(10):

                prem = price_simulations['da'] * 0.1 * i
                lsmc_result = self._lsmc_counter(delivery_days=delivery_days,
                                                 price_simulations=price_simulations['spot'],
                                                 strikes=np.array(strikes), fee=prem, mult_for_greek=j * 0.05)

                if prem == 0:
                    prems.append(lsmc_result['option_cost'])
                else:
                    pass
                if lsmc_result['option_cost'] < 0:

                    xes.append(prem)
                    yes.append(lsmc_result['option_cost'])

                    break
                else:
                    xes.append(prem)
                    yes.append(lsmc_result['option_cost'])
                del (lsmc_result)

            premia = -(yes[-1] * (xes[-2] - xes[-1]) / (yes[-2] - yes[-1])) + xes[-1]

            # lsmc_result_0 = self._lsmc_counter(delivery_days=delivery_days, price_simulations=price_simulations['spot'],
            #                                   strikes=np.array(strikes), fee=premia)
            # print(lsmc_result_0['option_cost'])
            fee_0.append(yes[0])
            maq_pc.append(j)
            premium.append(premia)

        return {'tcq': np.repeat(self._max_acq, len(fee_0)),
                'mtcq': np.repeat(self._min_acq, len(fee_0)),
                'dcq': np.repeat(self._max_dcq, len(fee_0)),
                'mdcq': np.repeat(self._min_dcq, len(fee_0)),
                'idx': np.repeat(str(self._indexes), len(fee_0)),
                'beg_date': np.repeat(self._delivery_start, len(fee_0)),
                'end_date': np.repeat(self._delivery_end, len(fee_0)),
                'premium': premium, 'maq_pc': maq_pc, 'fee_0': fee_0}

    def _load_to_bd(self, status: bool, premium: float = None, type_of_error: str = None,
                    contract_value_npv: float = None, avg_error: float = None, price_spec: str = None):

        output_var = CalcResultLoader().insert(
            point_name=self._delivery_point,
            model_type='volume_flex_' + self._price_model_type,
            is_successfull=status,
            type_of_error=type_of_error,
            calc_time=self._calc_time,
            n_sims=self._n_sims,
            beg_date=datetime.strptime(str(self._delivery_start)[:10], '%Y-%m-%d'),
            end_date=datetime.strptime(str(self._delivery_end)[:10], '%Y-%m-%d'),
            max_inv=None,
            min_inv=None,
            max_end_inv_tcq=self._max_acq,
            min_end_inv_mtcq=self._min_acq,
            max_inj_cap_dcq=self._max_dcq,
            min_inj_cap_mdcq=self._min_dcq,
            max_with_cap=None,
            min_with_cap=None,
            calc_date=datetime.strptime(str(self._date_of_pricing)[:10], '%Y-%m-%d'),
            volat_date=datetime.strptime(str(self._date_of_volat)[:10], '%Y-%m-%d'),
            contract_value_npv=contract_value_npv,
            avg_error=avg_error,
            premium=premium,
            price_spec=price_spec,
            comment=self.comment)

        return output_var


if __name__ == '__main__':
    t_0 = time.time()
    print(np.datetime64('2023-06-01')-np.datetime64('2023-03-18'))
    contract = VolumeFlex(delivery_start=np.datetime64('2023-03-18'), delivery_end=np.datetime64('2023-06-01'),
                          delivery_point='TTF',
                          date_of_pricing=np.datetime64('2023-03-15'), date_of_volat=np.datetime64('2023-03-14'),
                          max_acq=100 * 75, min_aqc=100 * 75 * 0.5, max_dcq=100,
                          min_dcq=50,
                          n_sims=2500,indexes=[1,0,1])
    premium = contract.count_premium()
    print(premium)
    print(time.time() - t_0)
    import pdb
    pdb.set_trace()
    # print(time.time() - t_0)
    """
    #theta = contract.count_theta()
    #print(theta)
    #pd.DataFrame.from_dict(theta).to_excel('/mnt/teamdocs_ns/TRD_Exchange/Valuations/Greeks_test/ctheta.xlsx',index=False)

    print(time.time() - t_0)
    vega = contract.count_vega()
    print(vega)
    pd.DataFrame.from_dict(vega).to_excel('/mnt/teamdocs_ns/TRD_Exchange/Valuations/Greeks_test/cvega_zeroflex.xlsx',index=False)
    print(time.time() - t_0)
    sg = contract.count_maq_greek()
    print(sg)
    pd.DataFrame.from_dict(sg).to_excel('/mnt/teamdocs_ns/TRD_Exchange/Valuations/Greeks_test/maq_greek_zeroflex.xlsx',index=False)
    print(time.time() - t_0)
    """
