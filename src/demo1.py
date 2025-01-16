import csv
import datetime
from datetime import timedelta
from pathlib import Path
import toml
import pandas as pd
from environs import Env

from iFinD import IFinD
from src.bond.calc import BondCalc
from utils.utils import is_trade_day

root_path = Path(__file__).resolve().parent.parent

env = Env()
env.read_env()

# 设置日期：上周五-本周五
today = datetime.datetime.now()
# today = datetime.date(2025, 1, 4)
days_ahead = 4 - today.weekday()

# current_friday_date = today + timedelta(days=days_ahead)
# last_friday_date = current_friday_date - timedelta(days=7)
current_friday_date = datetime.date(2025, 1, 13)
last_friday_date = datetime.date(2025, 1, 13)

start_date = last_friday_date.strftime("%Y%m%d")
end_date = current_friday_date.strftime("%Y%m%d")

setting_path = root_path / 'config' / 'setting.toml'
with open(setting_path, 'r', encoding='utf-8') as f:
    setting = toml.load(f)

bond_calc = BondCalc()


def main():
    # 配置文件地址
    config_path = root_path / 'config' / 'demo' / 'demo1.toml'
    # 输出地址
    output_path = root_path / 'output' / f'{start_date}-{end_date}.csv'
    with open(config_path, 'r', encoding='utf-8') as file:
        config = toml.load(file)

    sync_data()

    data_list = []
    for item in config:
        data = calc_func(config[item])
        data_list.extend(data)
        insert_with_filler(data_list, None)

    # 创建DataFrame
    df = pd.DataFrame(data_list)

    # 保存为CSV文件
    df.to_csv(output_path, index=False, header=False)


def sync_data():
    sync_date_str = setting['sync_date']
    sync_date = datetime.datetime.strptime(sync_date_str, '%Y%m%d')
    sync_date = sync_date.date() + timedelta(days=1)

    if sync_date > current_friday_date:
        return

    fetch_data(sync_date, current_friday_date)

    setting['sync_date'] = end_date
    with open(setting_path, 'w', encoding='utf-8') as file:
        toml.dump(setting, file)


def fetch_data(start_date, end_date):
    print('获取数据。。。')

    user = env.json('THS_USER')

    iFind = IFinD(user['username'], user['password'])

    total_days = (end_date - start_date).days + 1
    for current_date in range(total_days):
        current_date = start_date + timedelta(days=current_date)
        current_date_str = current_date.strftime('%Y%m%d')

        if not is_trade_day(current_date):
            print(f"{current_date_str}非交易日")
            continue

        data_pool = iFind.get_data_pool(current_date_str)
        if data_pool is None:
            raise Exception(f"{current_date_str}获取数据失败")

        codes = []
        for i in range(len(data_pool["jydm"])):
            codes.append(data_pool["jydm"][i])

        # ths_bond_latest_credict_rating_bond 债券最新评级
        # ths_bond_balance_bond 债券余额
        payload = {
            "codes": ','.join(codes),
            "indipara": [
                {
                    "indicator": "ths_bond_latest_credict_rating_bond",
                    "indiparams": [
                        "100",
                        "100"
                    ]
                },
                {
                    "indicator": "ths_bond_balance_bond",
                    "indiparams": [
                        current_date_str
                    ]
                }
            ]
        }
        basic_data = iFind.get_basic_data(payload)
        save_to_csv(data_pool, basic_data, current_date_str)

        file_path = root_path / 'data' / f'{current_date_str}.csv'
        # 隐含波动率
        df = pd.read_csv(file_path)
        codes = df['代码'].tolist()
        implied_volatility = iFind.get_implied_volatility(codes, current_date_str)
        df['隐含波动率'] = implied_volatility
        df.to_csv(file_path, index=False, encoding='utf-8')


def save_to_csv(data, basic_data, yesterday):
    desired_data = []

    for i in range(len(data["jydm"])):
        row_data = [data["jydm"][i], data["jydm_mc"][i], data["p00868_f002"][i], data["p00868_f016"][i],
                    data["p00868_f007"][i], data["p00868_f006"][i], data["p00868_f001"][i], data["p00868_f028"][i],
                    data["p00868_f011"][i], data["p00868_f005"][i], data["p00868_f014"][i], data["p00868_f008"][i],
                    data["p00868_f003"][i], data["p00868_f026"][i], data["p00868_f023"][i], data["p00868_f004"][i],
                    data["p00868_f012"][i], data["p00868_f017"][i], data["p00868_f024"][i], data["p00868_f019"][i],
                    data["p00868_f027"][i], data["p00868_f018"][i], data["p00868_f022"][i], data["p00868_f021"][i],
                    data["p00868_f015"][i], data["p00868_f010"][i], data["p00868_f025"][i], data["p00868_f009"][i],
                    data["p00868_f029"][i], data["p00868_f013"][i], data["p00868_f020"][i], data["p00868_f030"][i],
                    basic_data[i]['table']['ths_bond_latest_credict_rating_bond'][0],
                    basic_data[i]['table']['ths_bond_balance_bond'][0]
                    ]

        # 使用null代替--
        row_data = ["null" if val == "--" else val for val in row_data]

        desired_data.append(row_data)

    new_headers = ["代码", "名称", "交易日期", "前收盘价", "开盘价", "最高价", "最低价", "收盘价", "涨跌",
                   "涨跌幅(%)", "已计息天数", "应计利息", "剩余期限(年)", "当期收益率(%)", "纯债到期收益率(%)",
                   "纯债价值", "纯债溢价", "纯债溢价率(%)", "转股价格", "转股比例", "转换价值", "转股溢价",
                   "转股溢价率(%)", "转股市盈率", "转股市净率", "套利空间", "平价/底价", "期限(年)", "发行日期",
                   "票面利率/发行参考利率(%)", "交易市场", "债券类型", "债券最新评级", "债券余额"]

    current_dir = Path(__file__).resolve().parent
    file_path = current_dir.parent / 'data' / f'{yesterday}.csv'
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(new_headers)
        csvwriter.writerows(desired_data)

    print(f"{yesterday}数据已保存到data/{yesterday}.csv")


def calc_func(config):
    title = config['title']
    column = config['column']
    ctype = config['ctype']
    conditions = config['conditions']
    column_name = setting['column_name']

    data_list = []
    insert_with_filler(data_list, title)
    insert_with_filler(data_list, *column_name[ctype])

    print(title)
    total_days = (current_friday_date - last_friday_date).days + 1
    for current_date in range(total_days):
        current_date = last_friday_date + timedelta(days=current_date)
        current_date_str = current_date.strftime('%Y%m%d')

        if not is_trade_day(current_date):
            print(f"{current_date_str}非交易日")
            insert_with_filler(data_list, current_date_str)
            continue

        # 计算数据
        if ctype == "ratio":
            data_tuple = bond_calc.ratio(current_date_str, conditions)
        else:
            data_tuple = bond_calc.math_func(current_date_str, conditions, column, ctype)

        insert_with_filler(data_list, *data_tuple)

    return data_list


def insert_with_filler(lst, *elements):
    missing_count = 3 - len(elements)
    new_elements = list(elements) + [None] * missing_count
    lst.append(new_elements)


if __name__ == '__main__':
    main()
