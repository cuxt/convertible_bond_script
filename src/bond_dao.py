import duckdb
from pathlib import Path


class BondDao(object):
    def __init__(self, db_path=':memory:'):
        self.db_path = db_path
        self.con = None

    def __enter__(self):
        self.con = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.con:
            self.con.close()

    def close(self):
        if self.con:
            self.con.close()

    def query(self, sql, params=None):
        """
        执行 SQL 查询并返回结果。
        :param sql: 要执行的 SQL 查询语句
        :param params: SQL 查询中的参数（可选）
        :return: 查询结果（列表形式）
        """
        if not self.con:
            raise RuntimeError("Database connection is not established.")
        cursor = self.con.cursor()
        if params:
            # 执行语句时使用参数
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        # 获取查询结果
        result = cursor.fetchall()
        return result


def main():
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir.parent / 'data' / f'{20241230}.csv'
    query = f"""
           SELECT *
           FROM read_csv('{csv_path}', auto_detect=true, header=true) WHERE "代码" = ?
           """
    with BondDao() as dao:
        result = dao.query(query, ("117218.SZ",))
        print("查询结果：", result)


if __name__ == '__main__':
    main()
