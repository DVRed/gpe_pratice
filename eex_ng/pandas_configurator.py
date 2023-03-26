import pandas as pd


class PandasConfigurator:

    def __init__(self):
        self.df = pd.DataFrame({
            'date': [],
            'prices_name': [],
            'price': [],
            'hub': [],
            'unit': [],
            'currency': [],
            'price_type': [],
            'products': [],
            'id_source': [],
            'beg_date': [],
            'end_date': [],
            'product_type': []
        })

    def append(self, date, prices_name, price, hub, unit, currency, price_type,
               products, product_type, id_source=None, beg_date=None, end_date=None):
        self.df = pd.concat([self.df, pd.DataFrame({
            'date': date,
            'prices_name': prices_name,
            'price': price,
            'hub': hub,
            'unit': unit,
            'currency': currency,
            'price_type': price_type,
            'products': products,
            'id_source': id_source,
            'beg_date': beg_date,
            'end_date': end_date,
            'product_type': product_type
        })])


if __name__ == '__main__':
    a = PandasConfigurator()
    print(a.df)
