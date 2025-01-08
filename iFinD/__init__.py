import json
import requests
from .encrypt import iFinDEncrypt


class IFinD(object):
    def __init__(self, username, password):
        self.base_url = 'https://quantapi.51ifind.com/api/v1'
        self.username = username
        self.password = password
        self.cookie = self.get_cookie()
        self.refresh_token = self.get_refresh_token()
        self.access_token = self.get_access_token()
        self.headers = {
            "Content-Type": "application/json",
            "access_token": self.access_token
        }

    def get_cookie(self):
        ths_encrypt = iFinDEncrypt(self.username, self.password)
        return ths_encrypt.get_cookie()

    def get_refresh_token(self):
        url = self.base_url + '/get_refresh_token'
        headers = {
            'user-Agent': self.cookie['version'],
            'pragma': 'no-cache',
            'cookie': f"THSFT_USERID={self.cookie['THSFT_USERID']}; jgbsessid={self.cookie['jgbsessid']}; userid={self.cookie['userid']}; ifindlang=cn"
        }

        response = requests.get(url, headers=headers)
        return response.json()['data']['refresh_token']

    def get_access_token(self):
        """
        获取 access_token

        :return: access_token
        """
        url = self.base_url + '/get_access_token'
        header = {"ContentType": "application/json", "refresh_token": self.refresh_token}
        response = requests.post(url=url, headers=header)

        print(response)
        access_token = json.loads(response.content)['data']['access_token']

        return access_token

    def get_data(self, request_url, form_data):
        """
        获取数据，需要消耗额度

        :param request_url: 请求地址
        :param form_data: 请求参数
        :return:
        """
        requestHeaders = {"Content-Type": "application/json", "access_token": self.access_token}

        # 发送POST请求
        response = requests.post(request_url, headers=requestHeaders, data=json.dumps(form_data))

        # 检查请求是否成功
        if response.status_code == 200:
            # 解析 JSON 响应
            data = response.json()
            # 保存json数据
            with open('data.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            return data
        else:
            # 打印错误信息
            print("请求出错: {}".format(response.status_code))

    def get_data_free(self, date):
        jgbsessid = self.cookie['jgbsessid']
        url = 'http://ft.10jqka.com.cn/thsft/topicreport?reqtype=p00868'

        headers = {
            'Host': 'ft.10jqka.com.cn',
            'Content-Length': '129',
            'Accept': 'application/json, text/plain, */*',
            'Origin': 'http://ft.10jqka.com.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/84.0.4147.105 Safari/537.36',
            'sw8': '1-OGU2ZGYzNjctYzZhZC00M2FmLTk3OTgtNmQwYzgyODJiMDFk'
                   '-OGE3YTQ2YWMtMTE2Y00NDY0LWI0YmUtZmFmYTVlNmQ5ODM1-0-aWZpbmQtamF2YS10aGVtYXRpYy1iZmY8YnJvd3Nlcj4'
                   '=-cGNfY2xpZW50XzEuMA==-L3Jvb3Q=-ZnQuMTBqcWthLmNvbS5jbg==',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Cookie': f'jgbsessid={jgbsessid}',
            'Referer': 'http://ft.10jqka.com.cn/standardgwapi/bff/thematic_bff/topic/B0005.html?version=1.10.12.405'
                       '&mac=74-4C-A1-D5-77-AA',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7'
        }

        data = {
            'edate': date.strftime('%Y%m%d'),
            "zqlx": "全部",
            "begin": '1',
            "count": '700',
            'webPage': '1'
        }

        try:
            response = requests.post(url, headers=headers, data=data).json()
            # print(response)
            return response
        except requests.exceptions.RequestException as e:
            print(e)
            return

    def get_data_pool(self, trade_day):
        formData = {"reportname": "p00868", "functionpara": {"edate": trade_day, "zqlx": "全部"},
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
            print(f"请求出错: {response.json()}")
            return None

    def get_basic_data(self, payload):
        response = requests.post(self.base_url + '/basic_data_service', headers=self.headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            return data["tables"]
        else:
            print(f"可转债: {response.json()}")
            return None
