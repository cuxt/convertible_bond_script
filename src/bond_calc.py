import duckdb
from chinese_calendar import is_workday


class BondCalc:
    def __init__(self, db_path=':memory:'):
        self.con = duckdb.connect(db_path)

    def close(self):
        self.con.close()

    def is_trade_day(self, date):
        if is_workday(date):
            if date.isoweekday() < 6:
                return True
        return False

    def ratio(self, trade_date):
        if not self.is_trade_day(trade_date):
            return None

        str_date = trade_date.strftime("%Y%m%d")

        query = f"""
        SELECT *
        FROM read_csv('data/{str_date}.csv', auto_detect=true, header=true)
        """

        return self.con.execute(query).fetchall()
