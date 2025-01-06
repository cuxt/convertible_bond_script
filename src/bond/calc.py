from pathlib import Path
import toml
from src.bond.dao import BondDao


class BondCalc(object):
    def __init__(self):
        self.root_path = Path(__file__).resolve().parent.parent.parent
        self.config = self.load_config()

    def load_config(self):
        config_path = self.root_path / 'config' / 'csv.toml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        return config

    def ratio(self, filename, conditions):
        file_path = self.root_path / 'data' / f'{filename}.csv'

        columns_dict = {f'"{k}"': v for k, v in self.config['columns'].items()}
        columns_str = str(columns_dict).replace("'", "")

        base_query = f"""
            SELECT "代码" FROM read_csv(
                '{file_path}',
                delim='{self.config['type']['delim']}',
                quote='{self.config['type']['quote']}',
                escape='{self.config['type']['escape']}',
                header={str(self.config['type']['header']).lower()},
                ignore_errors={str(self.config['type']['ignore_errors']).lower()},
                columns={columns_str}
            )
        """

        if conditions and 'ratio_total' in conditions and conditions['ratio_total']:
            query = base_query + " WHERE " + " AND ".join(conditions['ratio_total'])
        else:
            query = base_query
        print(query)

        # 交易代码
        with BondDao() as dao:
            code = dao.query(query)

        if not isinstance(code, list):
            return filename, None, None
        if conditions and 'main' in conditions and conditions['main']:
            query = base_query + " WHERE " + " AND ".join(conditions['main'])
        else:
            query = base_query
        print(query)
        total = len(code)
        with BondDao() as dao:
            data_remain = dao.query(query)
        remain = len(data_remain)
        return filename, remain, total

    @classmethod
    def math_func(cls):
        pass


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
    data = bond_calc.ratio('20241120', conditions)
    print(data)


if __name__ == '__main__':
    main()
