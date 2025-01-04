from datetime import datetime
import duckdb
from utils.utils import is_trade_day
from pathlib import Path


class BondDao(object):
    def __init__(self, db_path=':memory:'):
        self.con = duckdb.connect(db_path)

    def close(self):
        self.con.close()

    def ratio(self, trade_date):
        if not is_trade_day(trade_date):
            print(f"{trade_date}不是交易日")
            return None

        str_date = trade_date.strftime("%Y%m%d")
        current_dir = Path(__file__).resolve().parent
        csv_path = current_dir.parent / 'data' / f'{str_date}.csv'
        print(csv_path)
        query = f"""
        SELECT *
        FROM read_csv('{csv_path}', auto_detect=true, header=true)
        """
        try:
            result = self.con.execute(query).fetchall()
            print(f"Query result: {result}")
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None


def main():
    db_session = BondDao()
    trade_day = datetime.strptime('2024-12-27', '%Y-%m-%d')
    data = db_session.ratio(trade_day)
    db_session.close()


if __name__ == '__main__':
    main()
