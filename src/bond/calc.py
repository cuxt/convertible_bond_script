from pathlib import Path

import numpy as np
import toml
from src.bond.dao import BondDao


class BondCalc(object):
    def __init__(self):
        self.root_path = Path(__file__).resolve().parent.parent.parent
        self.config = self.load_config()
        self.columns_str = self.format_column()

    def load_config(self):
        config_path = self.root_path / 'config' / 'csv.toml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        return config

    def format_column(self):
        columns_dict = {f'"{k}"': v for k, v in self.config['columns'].items()}
        columns_str = str(columns_dict).replace("'", "")
        return columns_str

    def ratio(self, filename, conditions):
        file_path = self.root_path / 'data' / f'{filename}.csv'

        base_query = f"""
            SELECT "代码" FROM read_csv(
                '{file_path}',
                delim='{self.config['type']['delim']}',
                quote='{self.config['type']['quote']}',
                escape='{self.config['type']['escape']}',
                header={str(self.config['type']['header']).lower()},
                ignore_errors={str(self.config['type']['ignore_errors']).lower()},
                columns={self.columns_str}
            )
        """

        if conditions and 'ratio_total' in conditions and conditions['ratio_total']:
            query = base_query + " WHERE " + " AND ".join(conditions['ratio_total'])
        else:
            query = base_query

        # 交易代码
        with BondDao() as dao:
            code = dao.query(query)

        if not isinstance(code, list):
            return filename, None, None
        if conditions and 'main' in conditions and conditions['main']:
            query = base_query + " WHERE " + " AND ".join(conditions['main'])
        else:
            query = base_query
        total = len(code)
        with BondDao() as dao:
            data_remain = dao.query(query)
        remain = len(data_remain)
        return filename, remain, total

    def math_func(self, filename, conditions, column, model='median'):
        file_path = self.root_path / 'data' / f'{filename}.csv'

        base_query = f"""
                    SELECT "{column}" FROM read_csv(
                        '{file_path}',
                        delim='{self.config['type']['delim']}',
                        quote='{self.config['type']['quote']}',
                        escape='{self.config['type']['escape']}',
                        header={str(self.config['type']['header']).lower()},
                        ignore_errors={str(self.config['type']['ignore_errors']).lower()},
                        columns={self.columns_str}
                    )
                """

        if conditions and 'main' in conditions and conditions['main']:
            query = base_query + " WHERE " + " AND ".join(conditions['main'])
        else:
            query = base_query

        with BondDao() as dao:
            data = dao.query(query)

        if isinstance(data, list) and len(data) == 0:
            return filename, '-'
        elif isinstance(data, int):
            print(f"\n{filename}:{data}")
            return filename, None

        try:
            if model == 'median':
                return filename, np.median(data)
            elif model == 'avg':
                return filename, np.mean(data)
            elif model == 'max':
                return filename, max(data)
            elif model == 'min':
                return filename, min(data)
            elif model == 'std_0':
                # 有偏样本标准差
                return filename, np.std(data, ddof=0)
            elif model == 'std_1':
                # 无偏样本标准差
                return filename, np.std(data, ddof=1)
        except Exception as e:
            print('\n', '出现异常', e)
            print(data)


def main():
    conditions = {
        "main": [
            '"债券类型" = \'可转债\'',
            '"转换价值" <= 80',
        ],
        "ratio_total": [
            '"债券类型" = \'可转债\''
        ]
    }
    bond_calc = BondCalc()
    data = bond_calc.math_func('20241120', conditions, '转股溢价率(%)')
    print(data)


if __name__ == '__main__':
    main()
