import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta


# TODO what's range to use?
# currently using date range is:
#   begin_date: 1st day of current month 8 years ago
#   end_date: last day of current month in next year
def get_date_range() -> (datetime, datetime):
    begin_date = datetime.now() - relativedelta(years=8)
    begin_date = begin_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = begin_date + relativedelta(years=9)
    begin_date = begin_date.replace(day=1)
    end_date = end_date.replace(day=calendar.monthrange(end_date.year, end_date.month)[1])
    return begin_date, end_date
