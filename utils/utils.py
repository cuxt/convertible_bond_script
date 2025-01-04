from chinese_calendar import is_workday


def is_trade_day(date):
    """
    判断是否是交易日

    :param date: 日期
    :return: 交易日返回True, 非交易日返回False
    """
    if is_workday(date):
        if date.isoweekday() < 6:
            return True
    return False
