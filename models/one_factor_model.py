"""
Business goal:
- calculate 1F (E. Schwartz) model of commodity price.
- plot graphics (if needed)
- return simulations 3D-array
"""
import datetime
import numpy as np
from volume_flex import VolumeFlex
from Models.get_data_from_DB.get_frd_curve import get_frd_advanced
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
from Models.get_data_from_DB.get_vol_surface import IceVolat


class OneFactorModel:
    """
    One-factor model based on Schmidt-Schwartz papers
    """
    def __init__(self, pricing_date: np.datetime64, n_months: int= 36):
        """
        Simulate prices using 1-factor model by Schwartz & Schmidt
        Methods
        --------
        :get_forward: get the real forwards values
        :calc_spot: calculate 1D-array spot prices of the asset.
        :map_months_season_coeff: create a dictionary of seasonal coefficients,
        where every month has its own coefficient.
        :calc_forward: calculate 1D-array forward.
        :stochastic_spot: calculate spot prices N simulations (for every day)
        :stochastic_forward: calculate forward prices N simulations
        :error_function: return value of error (MSE) function (for optimization)
        :fit_model: estimate 1F model params using scipy.optimize
        :show_plot: plot simulated prices

        Attributes
        ---------
        self._forward: list - real values of forwards
        self._contr_start: datetime - start of the contract exposition
        self._contr_end: datetime - end of contract exposition
        self._months_range: int - number of months between start and end
        self._f_range: int - number of seasonal coefficients
        self._days_range: int - number of days between start and end
        self._days_dict: dict - map exact date and its number
        self._wiener: np.array - Wiener process approximation
        self.res: OptimizeResult - results of scipy.optimize.minimize
        self.season_coefs: dict - map months and its coefficients
        """
        # if number of months less than 12, raise Error
        # TODO Add assertion description
        assert n_months >= 12
        self._months_range = n_months
        self._delivery_days_range = None
        self._total_days = None
        self._pricing_date = np.datetime64(pricing_date, 'D')
        self._full_days = int((np.datetime64(
            np.datetime64(self._pricing_date, 'M') +
            np.timedelta64(n_months, 'M'), 'D') -
            np.datetime64(np.datetime64(self._pricing_date, 'M'), 'D')) / np.timedelta64(1, 'D'))
        indexes = self.get_forward(pricing_date, self._months_range)
        self._forward = indexes['forwards']
        self._day_ahead = indexes['day_ahead'][0]
        # seasonal coefficients cannot be bigger than 12 because there are only
        # 12 months in a year. If we want somehow calculate coeffs with less than
        # 12 months, we need to take that number of months
        self._f_range = 12
        # OptimizeResult
        self.res = None
        self.custom_sigma = None
        # Seasonal coefficients {months: np.datetime64: coefficient: float}
        self.season_coefs = {}

    @staticmethod
    def get_forward(pricing_date: np.datetime64, months: int) -> dict[str: list, str: list]:
        """
        return real forwards values from the database
        """
        forwards = []
        day_ahead_list = []
        pricing_month = np.datetime64(pricing_date, 'M')
        for i in range(months):
            forward_this_month = pricing_month + np.timedelta64(i + 1, "M")
            forward_this_month = forward_this_month.item().strftime('%Y-%m-01')
            indexes = get_frd_advanced(pricing_date.item().strftime('%Y-%m-%d'),
                                       beg_date=forward_this_month)
            cur_forward = indexes['forward']
            day_ahead = indexes['day_ahead']
            forwards.append(cur_forward['value'].iloc[0])
            day_ahead_list.append(day_ahead)
        print(f'forwards = {forwards}')
        return {'forwards':forwards, 'day_ahead':day_ahead_list}


    def map_months_season_coeff(self, f:list) -> None:
        """
        map season coefficients with exact month in np.datetime64('M') format
        f - list of seasonal coefficients
        """
        cur_date = np.datetime64(self._pricing_date, 'M')
        for m in range(self._months_range + 1):
            self.season_coefs.update({cur_date: f[m % 12]})
            cur_date += np.timedelta64(1, 'M')

    def calc_forward(self, mu: float, k: float, sigma: float,
                     f: list[float], spot: float = None) -> np.ndarray:
        """
        calculate forwards with respect to seasonal coefficients
        mu, k, sigma, f - 1F model parameters
        """
        # resize list of seasonal coefficients if it's bigger than 12
        if self._months_range > 12:
            for j in range(self._months_range - 12):
                f = np.append(f, f[j % 12])
        _annual_coef: float =  1/12
        if spot is None:
            spot = np.log(self._day_ahead)
        j = np.linspace(0, self._months_range - 1, num= self._months_range)
        ln_F = np.log(f) + np.exp(-k * j * _annual_coef) * (spot / f[0]) + \
               np.array([(mu - sigma ** 2 / ( 2 * k ))] * self._months_range) * \
               ( 1 - np.exp(-k * j * _annual_coef)) + 1 / 2 * \
               (1 - np.exp(- 2 * k * j * _annual_coef)) * sigma ** 2 / (2 * k)

        return np.exp(ln_F)

    def forward_sims(self, mu: float, k: float, sigma: float,
                     f: list[float], sims_spot: np.array, total_days: int,
                     n_sims: int) -> np.ndarray:
        """
        calculate forwards with respect to seasonal coefficients
        mu, k, sigma, f - 1F model parameters
        sims_spot - list of spot simulations
        n_sims - number of simulations
        """
        _annual_coef: float =  1/12
        j = np.linspace(0, self._months_range - 1, num= self._months_range)
        j = np.resize(j, (n_sims, total_days, self._months_range))
        _f = np.resize(np.log(f), (n_sims, total_days, self._months_range))
        Dt = np.resize(sims_spot, (n_sims, total_days, self._months_range))
        tail = np.array([(mu - sigma ** 2 / ( 2 * k ))] * self._months_range) * \
               ( 1 - np.exp(-k * j * _annual_coef)) + 1 / 2 * \
               (1 - np.exp(- 2 * k * j * _annual_coef)) * sigma ** 2 / (2 * k)
        tail = np.resize(tail, (n_sims, total_days, self._months_range))

        ln_F = _f + np.exp(-k * j * _annual_coef) * Dt + tail

        return np.array(ln_F)

    def fit_model(self,
                  custom_sigma: float= None,
                  days_range_for_volat: int= None) -> None:
        """
        Finding mu, k, sigma and f (seasonal coefficients) for 1F model.
        - Set-up initial values for optimized params.
        - Set-up constraints and bounds
        - Run optimization with L-BFGS-B algorithm with no constraints because
        according to official scipy documentation it doesn't support constraints,
        but this algorithm is claimed to be one of the most efficient one among
        all other algorithms available with optimize.minimize module. Obtained
        values are transferred as initial point to the next step.
        - Run optimization with SLSQP algorithm with all constraints and with
        initial values taken as a result of previous step of optimization.
        Two optimization mechanisms are required because one works better but
        doesn't support constrained optimization, and the other works worse but
        supports constrained optimization.
        """
        # initial data set-up
        f_0 = np.ones(self._f_range)
        mu_0, k_0, sigma_0 = np.array([1]), np.array([1]), np.array([1])
        x_0 = np.append([mu_0, k_0, sigma_0], f_0)
        # constrains coefficients for optimized variables. It works as follows
        # lower_bound < constr_coeff.dot(solution) < upper_bound
        constr_coeff = np.ones(len(x_0))
        # make mu_0, k_0, sigma_0 unconstrained
        constr_coeff[0], constr_coeff[1], constr_coeff[2] = 0, 0, 0
        constr = ({'type': 'eq', 'fun': lambda x:  np.sum(x[3:]) - self._f_range})
        # bounds for variables - not constraints!
        lower_bound = []
        upper_bound = []
        for _ in x_0:
            lower_bound.append(0.1)
            upper_bound.append(10)
        if custom_sigma is not None:
            self.custom_sigma = custom_sigma
            #lower_bound[2] = custom_sigma
            #upper_bound[2] = custom_sigma
            days_range = days_range_for_volat
            # TODO replace 90 with exact number - from delivery start to the middle of delivery period
            constr = (
                {'type': 'eq', 'fun': lambda x: np.sum(x[3:]) - self._f_range},
                {'type': 'eq', 'fun': lambda x:
                    x[2]**2 - self.custom_sigma**2 * 2 * x[1] * (days_range/365) /
                    (1 - np.exp(- 2 * x[1] * (days_range/365)))
                }
                )

        bounds = Bounds(lower_bound, upper_bound)
        self.res = minimize(self.error_function, x_0, method='L-BFGS-B',
                            bounds= bounds)
        print(f"optimization result success = {self.res.success}")
        x_0 = self.res.x
        self.res = minimize(self.error_function, x_0, method='SLSQP',
                            bounds= bounds, constraints= constr)
        print(f"optimization result success = {self.res.success}")
        print(self.res.x)
        # upload season coefficients to self
        self.map_months_season_coeff(self.res.x[3:])

    def error_function(self, x: list) -> float:
        """
        return error value (mean square error) for optimized function.
        x - list values for 1F model.
        """
        mu: float = x[0]
        k: float = x[1]
        sigma: float = x[2]
        f: list[float] = x[3:]
        forward = np.array(self._forward)
        return float(np.sum((self.calc_forward(mu, k, sigma, f) - forward)**2))

    def simulate_spot(self, mu: float, k: float,
                      sigma: float, delivery_days:int,
                      n_sims: int= 2500) -> np.array:
        """
        calculate deseasonalized forwards in logarithm format
        mu, k, sigma - parameters for 1F model
        wiener - Wiener process approximation
        """
        # empty array for recurrent equation
        ln_D_days = np.zeros((n_sims, delivery_days))
        ln_D_days[:, 0] = np.resize(np.log(self._day_ahead), n_sims)
        eps = np.random.normal(0, 1, (n_sims, delivery_days))
        dt = 1 / 365
        for n in range(1, delivery_days):
            ln_D_days[:, n] = np.exp(-k * dt) * ln_D_days[:, n - 1] + \
                           (mu - sigma ** 2 / (2 * k)) * (1 - np.exp(-k * dt)) + \
                           sigma * np.sqrt(1 / (2 * k) * (1 - np.exp(-2 * k * dt))) * eps[:, n]
                           #(1 - np.exp(-2 * k * dt)) * sigma ** 2 / (4 * k ) + \

        # add seasonal coefficients
        ln_P_days = np.zeros((n_sims, delivery_days))
        # TODO REMOVE THIS TEMP
        if self.res is None:
            self.map_months_season_coeff([1]*12)
        cur_date = np.datetime64(self._pricing_date, 'D')
        for n in range(0, delivery_days):
            # add seasonal coeffs
            cur_date += np.timedelta64(1, 'D')
            cur_month = np.datetime64(cur_date, 'M')
            ln_P_days[:, n] = ln_D_days[:, n] + np.resize(np.log(self.season_coefs[cur_month]),
                                                          n_sims)

        return ln_P_days

    def get_simulations(self, delivery_start: np.datetime64,
                        delivery_end: np.datetime64, n_sims: int= 2500,
                        custom_sigma: float= None, show_sims: bool= True):
        """
        return simulations in format usable for flex_counter and other modules
        the format is:
        {'futures': prices,
        'spot': np.array(spot_prices_sim),
        'da': day_ahead_price,
        'atm': atm_price}
        """
        # get the initial data
        delivery_days_range = (int((delivery_start - self._pricing_date) /
                                  np.timedelta64(1, 'D')),
                               int((delivery_end - self._pricing_date) /
                                  np.timedelta64(1, 'D')))
        total_days = int((delivery_end - self._pricing_date) /
                         np.timedelta64(1, 'D'))

        self._delivery_days_range = delivery_days_range
        self._total_days = total_days

        if self.res is None:
            f = [1] * 12
            mu, k, sigma = 1, 1, 1
        else:
            f = self.res.x[3:]
            mu, k, sigma = self.res.x[0], self.res.x[1], self.res.x[2]
        if custom_sigma is not None:
            sigma = custom_sigma
        # measure time
        time_1 = datetime.datetime.now()
        spot_sims = self.simulate_spot(mu, k, sigma,
                                       delivery_days_range[1], n_sims)
        # spot_sims returned in array format
        # the format requires slice for DA prices
        forward_sims = self.forward_sims(mu, k, sigma, f, spot_sims,
                                         total_days, n_sims)
        print(f"Time for simulations = {datetime.datetime.now() - time_1}")
        spot_sims = np.exp(spot_sims)
        forward_sims = np.exp(forward_sims)
        spot_sims = spot_sims[:, delivery_days_range[0]: delivery_days_range[1]]

        #show plot
        if show_sims:
            spot_sims_all = self.simulate_spot(mu, k, sigma,
                                               self._full_days, n_sims)
            spot_sims_all = np.exp(spot_sims_all)
            self.show_sims(sims_to_plot = spot_sims_all, n_sims= 0)


        return  {'futures': forward_sims, 'spot': spot_sims, 'da': self._day_ahead}

    def show_forward(self) -> None:
        """
        Show plot of forward prices.
        show_sims - bool parameter,plot simulations if True.
        """
        f = self.res.x[3:]
        mu, k, sigma = self.res.x[0], self.res.x[1], self.res.x[2]
        print(f"sigma = {sigma}")
        calculated_forward = self.calc_forward(mu, k, sigma, f)
        x_axis = np.arange(0, self._months_range, 1)
        plt.plot(x_axis, self._forward, 'r-', label= 'real forward')
        plt.plot(x_axis, calculated_forward, 'b-.', label= 'simulation')
        plt.xlabel("months")
        plt.ylabel("price")
        plt.title(f"real forward vs 1F model simulation \n at {datetime.datetime.now()}")
        plt.legend()
        print("f = {0} \n mu = {1} \n k = {2} \n sigma = {3} \n".
              format(f, mu, k, sigma))
        plt.show()

    @staticmethod
    def show_imvol(vols: list) -> None:
        """
        Show implied volatilities
        """
        x_axis = np.arange(0, len(vols), 1)
        plt.plot(x_axis, vols, 'r-', label= 'implied volats')
        plt.xlabel("months")
        plt.ylabel("value")
        plt.title(f"Implied volatilities")
        plt.legend()
        plt.show()

    def show_sims(self,
                  sims_to_plot: np.array,
                  n_sims: int= 25,
                  y_lim = None) -> None:
        """
        """
        #self.spot_sims_all
        x_axis = np.arange(0, self._months_range * 30, 30)
        plt.plot(x_axis, self._forward, 'r-', label= 'real forward')
        spot_start = self._delivery_days_range[0]
        x_axis = np.arange(spot_start, len(sims_to_plot[0]) + spot_start, 1)
        for sim in sims_to_plot[0: n_sims]:
            plt.plot(x_axis, sim, 'b-.')
        plt.plot(x_axis, np.mean(sims_to_plot, axis= 0),
                 'g--', label= 'avg_simulation')
        if y_lim is not None:
            plt.ylim([0, y_lim])
        plt.xlabel("days")
        plt.ylabel("price")
        plt.title(f"real forward vs 1F DA simulations \n at "
                  f"{datetime.datetime.now()}")
        plt.legend()
        plt.show()

def get_impl_vol(price_date,
                 start: np.datetime64,
                 end: np.datetime64) -> (float, list):
    """
    Calculate implied volatility
    :price_date: date of pricing
    :start: first month of delivery period
    :end: last month of delivery period
    return
    tuple(mean_impl_vol: float, impl_vol: list)
    """
    im_vol = []
    vols = IceVolat(price_date, 'all','atm').get_table()
    cur_day = np.datetime64(start, 'D')
    while cur_day != end:
        im_vol.append(vols[cur_day.item().strftime('%Y-%m-%d')].iloc[0])
        cur_month = np.datetime64(cur_day, 'M')
        next_month = cur_month + np.timedelta64(1, 'M')
        cur_day = np.datetime64(next_month, 'D')

    print(f"mean impl_vol = {np.mean(im_vol)}")

    return np.mean(im_vol), im_vol

def count_days_mid_deliv(pricing: np.datetime64,
                         start: np.datetime64,
                         end: np.datetime64,) -> int:
    """
    TODO write comments
    """
    # middle date of delivery
    middle_of_delivery = \
        start + np.timedelta64(
            int(((end - start) / np.timedelta64(1, 'D')) / 2), 'D'
        )
    days = (middle_of_delivery - pricing) / np.timedelta64(1, 'D')
    return days

if __name__ == '__main__':
    # initial params setting
    date_of_pricing: np.datetime64= np.datetime64('2022-11-15')
    deliv_start: np.datetime64= np.datetime64('2023-01-01')
    deliv_end: np.datetime64= np.datetime64('2023-04-01')
    num_months: int= 36
    num_sims: int= 2500
    # build forward curve
    one_f_model = OneFactorModel(pricing_date= date_of_pricing,
                                 n_months= num_months)
    # SET-UP volatilities params and show_plot
    use_implied_volatility: bool= True
    show_imvol: bool = True
    if use_implied_volatility:
        imvol, imvol_list = get_impl_vol(date_of_pricing, deliv_start, deliv_end)
        print(imvol_list)
        # middle of delivery period
        midd_delivery_days = count_days_mid_deliv(np.datetime64(date_of_pricing),
                                                  deliv_start,
                                                  deliv_end)
        # had to explicitly write float(imvol) below in function call because
        # otherwise PyCharm displays warning like 'expected float, got ndarray'
        # because it thinks that imvol is ndarray but not float, but it's
        # actually float. It's an issue with numpy.
        one_f_model.fit_model(custom_sigma= float(imvol),
                              days_range_for_volat=midd_delivery_days)
        if show_imvol:
            one_f_model.show_imvol(imvol_list)
    else:
        one_f_model.fit_model()
    one_f_model.show_forward()

    # get spot and forward simulations
    calc_premia: bool = True
    if calc_premia:
        print('---calculating simulations---')
        sims = one_f_model.get_simulations(n_sims= num_sims,
                                           delivery_start= deliv_start,
                                           delivery_end= deliv_end)

        one_f_model.show_sims(sims_to_plot=sims['spot'], n_sims=5)
        count_vol_flex: bool = False
        if count_vol_flex:
            contract = VolumeFlex(delivery_start= deliv_start,
                                  delivery_end= deliv_end,
                                  delivery_point= 'TTF',
                                  date_of_pricing= np.datetime64(date_of_pricing),
                                  date_of_volat= np.datetime64(date_of_pricing),
                                  indexes= [1, 0, 1], max_acq= 100*90,
                                  min_aqc= int(100*90*0.8), max_dcq= 100,
                                  min_dcq= 0,
                                  n_sims= num_sims)

            premium = contract.count_premium(custom_sims=sims)
            print(premium)
            print('-----Base class------')
            premium = contract.count_premium()
            print(premium)




