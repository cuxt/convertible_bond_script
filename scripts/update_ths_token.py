import base64
import hashlib

import requests
from Cryptodome.Cipher import PKCS1_v1_5
from Cryptodome.PublicKey import RSA
from environs import Env

env = Env()
env.read_env()


def get_cookie(username, password):
    username = encode(username)
    password = encode(hex_md5(password))

    url = "http://ft.10jqka.com.cn/thsft/jgbservice"

    params = {
        "reqtype": "verify",
        "account": username,
        "passwd": password,
        "qsid": "800",
        "securities": "同花顺iFinD",
        "jgbversion": "1.10.12.405",
        "Encrypt": "1"
    }

    response = requests.get(url, params=params)

    set_cookie = response.headers['Set-Cookie'].replace(" ", '')

    cookie_list = set_cookie.split(';')
    cookies = {}
    for cookie in cookie_list:
        key_value_pairs = cookie.split(',')
        for pair in key_value_pairs:
            key_value = pair.split('=')
            cookies[key_value[0]] = key_value[1]

    return cookies


def encode(string):
    modulus = 'CB99A3A4891FFECEDD94F455C5C486B936D0A37247D750D299D66A711F5F7C1EF8C17EAFD2E1552081DFFD1F78966593D81A499B802B18B0D76EF1D74F217E3FD98E8E05A906245BEDD810557DFB8F653118E59293A08C1E51DDCFA2CC13251A5BE301B080A0C93A587CB71BAED18AEF9F1E27DA6877AFED6BC5649DB12DD021'
    publicExponent = '10001'
    return encrypt_encode(string, modulus, publicExponent)


def encrypt_encode(string, modulus, publicExponent):
    rsa_key = RSA.construct((int(modulus, 16), int(publicExponent, 16)))

    cipher = PKCS1_v1_5.new(rsa_key)

    b_bytes = bytes(string, 'utf-8')
    encrypted_bytes = cipher.encrypt(b_bytes)

    if not encrypted_bytes:
        raise ValueError("加密失败")

    encrypted_hex = encrypted_bytes.hex()
    encrypted_base64 = base64.b64encode(bytes.fromhex(encrypted_hex)).decode('utf-8')

    return encrypted_base64


def hex_md5(c):
    # 创建MD5哈希对象
    md5_hasher = hashlib.md5()
    # 将输入字符串编码为UTF-8并更新MD5哈希对象
    md5_hasher.update(c.encode('utf-8'))
    # 返回MD5加密结果的十六进制表示
    return md5_hasher.hexdigest()


def get_refresh_token(cookie):
    url = 'https://quantapi.51ifind.com/api/v1/get_refresh_token'
    headers = {
        'user-Agent': cookie['version'],
        'pragma': 'no-cache',
        'cookie': f"THSFT_USERID={cookie['THSFT_USERID']}; jgbsessid={cookie['jgbsessid']}; userid={cookie['userid']}; ifindlang=cn"
    }

    response = requests.get(url, headers=headers)
    return response.json()['data']['refresh_token']


def save_token(token):
    headers = {
        'Authorization': f'Bearer {env.str('KEY')}'
    }

    response = requests.post('https://api.xbxin.com/kv/write', json={
        'key': 'ths',
        'value': token
    }, headers=headers)

    print(response.json())


def main():
    user = env.json('THS_USER')

    cookie = get_cookie(user['username'], user['password'])
    token = get_refresh_token(cookie)
    save_token(token)


if __name__ == '__main__':
    main()
