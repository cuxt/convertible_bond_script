import csv
import json
from datetime import datetime, timedelta
from time import sleep

import requests
from chinese_calendar import is_workday
from environs import Env

env = Env()
env.read_env()


class THS:
    def __init__(self, trade_day):
        self.base_url = 'https://quantapi.51ifind.com/api/v1'
        self.trade_day = trade_day
        self.access_token = self.get_access_token()
        self.headers = {
            "Content-Type": "application/json",
            "access_token": self.access_token
        }

    def get_data_pool(self):
        formData = {"reportname": "p00868", "functionpara": {"edate": self.trade_day, "zqlx": "全部"},
                    "outputpara": "jydm,jydm_mc,p00868_f002,p00868_f016,p00868_f007,p00868_f006,p00868_f001,"
                                  "p00868_f028,p00868_f011,p00868_f005,p00868_f014,p00868_f008,p00868_f003,"
                                  "p00868_f026,p00868_f023,p00868_f004,p00868_f012,p00868_f017,p00868_f024,"
                                  "p00868_f019,p00868_f027,p00868_f018,p00868_f022,p00868_f021,p00868_f015,"
                                  "p00868_f010,p00868_f025,p00868_f009,p00868_f029,p00868_f013,p00868_f020,p00868_f030"}

        response = requests.post(self.base_url + '/data_pool', headers=self.headers, data=json.dumps(formData))

        if response.status_code == 200:
            data = response.json()
            return data["tables"][0]["table"]
        else:
            send_msg(f"可转债: {response.json()}")
            return None

    def get_basic_data(self, payload):
        response = requests.post(self.base_url + '/basic_data_service', headers=self.headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            return data["tables"]
        else:
            send_msg(f"可转债: {response.json()}")
            return None

    @staticmethod
    def get_access_token(retries=3, delay=5):
        baseurl = 'https://api.xbxin.com/ths/token'
        payload = {
            'key': env.str('KEY')
        }

        response = requests.post(baseurl, json=payload)

        if response.status_code != 200:
            send_msg(f"可转债：{response.json()['data']['error']}")
            return None

        response = response.json()
        refresh_token = response['data']['token']
        url = 'https://ft.10jqka.com.cn/api/v1/get_access_token'
        headers = {"ContentType": "application/json", "refresh_token": refresh_token}

        attempt = 0
        while attempt < retries:
            try:
                response = requests.post(url=url, headers=headers)
                response = json.loads(response.content)
                if response['errorcode'] != 0:
                    send_msg(f"可转债：{response['errmsg']}")
                    access_token = None
                else:
                    access_token = response['data']['access_token']

                return access_token
            except requests.exceptions.SSLError:
                attempt += 1
                if attempt < retries:
                    sleep(delay)
                else:
                    send_msg("可转债：获取access_token失败，请检查网络")
                    return None


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

    with open(f'data/{yesterday}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(new_headers)
        csvwriter.writerows(desired_data)


def is_trade_day(date):
    if is_workday(date):
        if date.isoweekday() < 6:
            return True
    return False


def send_msg(content):
    url = "https://api.xbxin.com/msg/admin/corp"
    token = env.str("BX_TOKEN")

    headers = {
        'Authorization': f'Bearer {token}',
    }

    data = {
        "title": "同花顺",
        "desc": "log",
        "content": content
    }

    requests.post(url, json=data, headers=headers)


def main():
    # today = datetime(2024, 11, 22)
    today = datetime.now()
    trade_day = today - timedelta(days=1)
    if is_trade_day(trade_day):
        trade_day = trade_day.strftime("%Y%m%d")
        ths = THS(trade_day)
        if ths.access_token is None:
            print("获取access_token失败")
            return
        data_pool = ths.get_data_pool()

        if data_pool is None:
            send_msg("可转债：获取交易数据失败")
            return

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
                        trade_day
                    ]
                }
            ]
        }
        basic_data = ths.get_basic_data(payload)
        # print(basic_data)
        save_to_csv(data_pool, basic_data, trade_day)


if __name__ == '__main__':
    main()
