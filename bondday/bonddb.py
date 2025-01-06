import datetime
import os

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class BondDB(object):
    @classmethod
    def init(cls, database='convertible_bond'):
        # 建立SQLite数据库连接
        engine = create_engine(f'sqlite:///{database}.db')

        # 创建inspect对象
        inspector = inspect(engine)

        # 遍历文件夹中的CSV文件
        folder_path = 'data'
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                table_name = file.split('.')[0]  # 使用文件名作为表名
                # 检查数据库中是否已经存在相应的表
                if not inspector.has_table(table_name):
                    # 如果数据库中不存在相应的表，则导入CSV文件内容
                    try:
                        df = pd.read_csv(os.path.join(folder_path, file), encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(os.path.join(folder_path, file), encoding='gbk')
                    df.to_sql(table_name, con=engine, if_exists='replace', index=False)

        # 关闭数据库连接
        engine.dispose()
        return True

    @classmethod
    def query(cls, table, column, sql_dict, database='convertible_bond'):
        # 建立SQLite数据库连接
        engine = create_engine(f'sqlite:///{database}.db')

        # 构建查询语句
        query = f"SELECT `{column}` FROM `{table}`"

        if 'main' in sql_dict:
            query += " WHERE " + " AND ".join(sql_dict['main'])

            # 排序
            if 'sort' in sql_dict and sql_dict['sort'] is not None:
                query += f" ORDER BY `{sql_dict['sort']}`"
            if 'sort_type' in sql_dict and sql_dict['sort_type'] == 'desc':
                query += " desc"

            # 限制返回数量
            if 'limit' in sql_dict and sql_dict['limit'] != -1:
                if sql_dict['limit'] <= 0:
                    print(" limit 必须大于 0")
                else:
                    query += f" LIMIT {sql_dict['limit']}"
        else:
            query += " WHERE " + " AND ".join(sql_dict)

        query += " AND " + f"`{column}` is not null"

        try:
            df = pd.read_sql_query(query, con=engine)
            data_bond = df.values.tolist()
        except sqlalchemy.exc.OperationalError as e:
            error_message = str(e)

            if "no such table" in error_message:
                # 数据表不存在
                data_bond = -1
            elif "no such column" in error_message:
                # 列不存在
                data_bond = -2
            else:
                data_bond = -3

        # 关闭数据库连接
        engine.dispose()

        return data_bond


# 使用示例
if __name__ == "__main__":
    conditions = {
        "main": [
            "债券类型 = '可转债'",
            "`纯债到期收益率(%)` > 3",
            "((`转换价值` - `纯债价值`) / `纯债价值`) > 0.2"
        ],
        "sort": None,
        "sort_type": "",
        "limit": -1,
        "ratio_total": [
            "债券类型 = '可转债'"
        ]
    }
    bond_day = datetime.date(2024, 12, 26)
    data = BondDB.query(bond_day, '码', conditions["main"], database='../convertible_bond')
    print(data)
