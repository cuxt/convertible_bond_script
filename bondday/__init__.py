import datetime
import numpy as np
from utils.utils import is_trade_day
from .bonddb import BondDB


class BondDay(object):
    db_init_executed = False

    @classmethod
    def db_init(cls):
        """
        初始化数据库

        :return:
        """
        if not cls.db_init_executed:
            print('初始化数据库中...')
            BondDB.init()
            print('初始化完成')
            cls.db_init_executed = True

    @classmethod
    def ratio(cls, current_date, conditions, database='convertible_bond'):
        """
        计算比率

        :param current_date: 日期
        :param conditions: sql
        :param database: 数据库名
        :return: 元组(日期, 筛选后剩余数量, 总数)
        """
        bond_day = current_date.strftime('%Y-%m-%d')
        if not is_trade_day(current_date):
            return bond_day, None, None

        table_name = current_date.strftime('%Y%m%d')

        # 交易代码
        code = BondDB.query(table_name, "代码", conditions['ratio_total'], database)
        if isinstance(code, list):
            total = len(code)
            data_remain = BondDB.query(table_name, "代码", conditions, database)
            remain = len(data_remain)
            return bond_day, remain, total
        else:
            print(f"\n{bond_day}:{code}")
            return bond_day, None, None

    @classmethod
    def bond_math(cls, current_date, column, conditions, model='median', database='convertible_bond'):
        """
        计算中位数、平均数

        :param current_date: 日期
        :param column: 需要计算的列
        :param conditions: sql
        :param model: median,avg
        :param database: 数据库名
        :return: 元组(日期, 中位数/平均数)
        """
        bond_day = current_date.strftime('%Y-%m-%d')
        if not is_trade_day(current_date):
            return bond_day, None

        table_name = current_date.strftime('%Y%m%d')

        # 交易代码
        data = BondDB.query(table_name, column, conditions, database)

        if isinstance(data, list) and len(data) == 0:
            return bond_day, '-'
        elif isinstance(data, int):
            print(f"\n{bond_day}:{data}")
            return bond_day, None

        try:
            if model == 'median':
                return bond_day, np.median(data)
            elif model == 'avg':
                return bond_day, np.mean(data)
            elif model == 'max':
                return bond_day, max(data)
            elif model == 'min':
                return bond_day, min(data)
            elif model == 'std_0':
                # 有偏样本标准差
                return bond_day, np.std(data, ddof=0)
            elif model == 'std_1':
                # 无偏样本标准差
                return bond_day, np.std(data, ddof=1)
        except Exception as e:
            print('\n', '出现异常', e)
            print(data)


def main():
    BondDay.db_init()
    conditions = {
        "main": [
            '"债券类型" = \'可转债\'',
            '"转换价值" <= 80',
        ],
        "ratio_total": [
            '"债券类型" = \'可转债\''
        ]
    }
    bond_date = datetime.date(2024, 11, 20)
    data = BondDay.ratio(bond_date, conditions)
    print(data)


if __name__ == '__main__':
    main()
