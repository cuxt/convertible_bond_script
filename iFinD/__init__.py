import json
import requests
from .encrypt import iFinDEncrypt
from environs import Env


class THS(object):
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.cookie = self.get_cookie()

    def get_cookie(self):
        ths_encrypt = iFinDEncrypt(self.username, self.password)
        return ths_encrypt.get_cookie()

    def get_access_token(self):
        """
        获取 access_token

        :return: access_token
        """
        env = Env()
        env.read_env()

        refresh_token = env.str("REFRESH_TOKEN")
        print(refresh_token)
        url = 'https://ft.10jqka.com.cn/api/v1/get_access_token'
        header = {"ContentType": "application/json", "refresh_token": refresh_token}
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
        #  获取 access_token
        access_token = self.get_access_token()

        requestHeaders = {"Content-Type": "application/json", "access_token": access_token}

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
