import requests
from datetime import timedelta, datetime

"""
Finds the limit of daysback for EEX requests
"""

headers = {
    'Origin': 'https://www.eex.com',
    'Referer': 'https://www.eex.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
}


def find_daysback_limit_for_indices() -> int:

    start_date = datetime.today()
    end_date = datetime.today()

    last_size = -1
    new_size = 0

    url = 'https://webservice-eex.gvsi.com/query/json/getDaily/symbol/ontradeprice/offtradeprice/close/' \
          'onexchsingletradevolume/onexchtradevolumeeex/offexchtradevolumeeex/openinterest/tradedatetimegmt/'

    symbol = '"#E.CEGH_WDRP"'

    while last_size != new_size:
        last_size = new_size
        start_date = start_date - timedelta(days=100)

        params = {
            'priceSymbol': symbol,
            'daysback': str((end_date - start_date).days + 1),
            'chartstopdate': end_date.strftime('%Y/%m/%d'),
            'dailybarinterval': 'Days',
            'aggregatepriceselection': 'First'
        }

        r = requests.get(url, params=params, headers=headers)

        new_size = len(r.json()['results']['items'])
        if new_size != 0:
            # 2/17/2021 12:00:00 PM. Tested on 7 april 2023
            print(r.json()['results']['items'][0]['tradedatetimegmt'])

        print(new_size)  # 779 max

    return new_size


def find_daysback_limit_for_spot() -> int:

    start_date = datetime.today()
    end_date = datetime.today()

    last_size = -1
    new_size = 0

    url = 'https://webservice-eex.gvsi.com/query/json/getDaily/ontradeprice/onexchsingletradevolume/close' \
          '/onexchtradevolumeeex/tradedatetimegmt/'

    symbol = '"#E.CEGH_WDRP"'

    while new_size != last_size:
        last_size = new_size
        start_date = start_date - timedelta(days=100)
        params = {
            'priceSymbol': symbol,
            'chartstartdate': start_date.strftime('%Y/%m/%d'),
            'chartstopdate': end_date.strftime('%Y/%m/%d'),
            'dailybarinterval': 'Days',
            'aggregatepriceselection': 'First'
        }
        r = requests.get(url, params=params, headers=headers)

        new_size = len(r.json()['results']['items'])
        if new_size != 0:
            # max = 2/17/2021 12:00:00 PM tested on 07 april 2023
            print(r.json()['results']['items'][0]['tradedatetimegmt'])
        print(new_size)  # 779 max

    return new_size


# TODO need more tests and checks for the correctness of the results
def find_daysback_limit_for_futures():

    url = 'https://webservice-eex.gvsi.com/query/json/getChain/gv.pricesymbol/gv.displaydate/gv.expirationdate' \
          '/tradedatetimegmt/gv.eexdeliverystart/ontradeprice/close/onexchsingletradevolume/onexchtradevolumeeex' \
          '/offexchtradevolumeeex/openinterest/'

    symbol = '"/E.G8_DAILY"'

    on_date = datetime.today()
    step = 100
    r = 0

    while 1:
        new_size = 0
        k = 0
        while new_size == 0:
            on_date = on_date - timedelta(days=1)
            params = {
                'optionroot': symbol,
                'onDate': on_date.strftime('%Y/%m/%d')
            }
            r = requests.get(url, params=params, headers=headers)
            new_size = len(r.json()['results']['items'])
            k = k + 1
            # if we get more than 15 empty responses in a row, step back and decrease step
            if k > 15:
                on_date = on_date + timedelta(days=step) + timedelta(days=15)
                # if step is min return
                if step == 1:
                    return (datetime.today() - on_date).days  # last not zero
                else:
                    step //= 10
                break
        on_date = on_date - timedelta(days=step)

        # print('Last not 0 date: ', on_date)
        # print('Delta time ' + str((datetime.today() - on_date).days))
        #
        # if new_size != 0:
        #     print(r.json()['results']['items'][0]['tradedatetimegmt'])
        # print(new_size)  # max was 651
        #
        # max was 2021-06-25. Tested on 07 april 2023
        # print(on_date)


if __name__ == '__main__':
    # returned value: 779. The earliest response date 2/17/2021 12:00:00 PM tested on 7 april 2023
    print(find_daysback_limit_for_indices())
    # returned value: 779. The earliest response date 2/17/2021 12:00:00 PM tested on 7 april 2023
    print(find_daysback_limit_for_spot())
    # returned value: 652. The earliest response date 6/25/2021 12:00:00 PM tested on 7 april 2023
    print(find_daysback_limit_for_futures())