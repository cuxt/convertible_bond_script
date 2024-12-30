## convertible_bond_script

一个基于iFind和Wind数据接口获取可转债数据，并对数据进行处理的脚本。

## 安装指南

该项目目前采用Python语言编写，以下是安装指南：
1. 克隆项目代码库到本地：
    ```bash
    git clone git@github.com:zhoubinxin/convertible_bond_script.git
    ```
2. 进入项目目录：
    ```bash
    cd convertible_bond_script
    ```
3. 安装依赖项：
    ```bash
    pip install -r requirements.txt
    ```
4. 运行程序：
    ```bash
    python main.py
    ```

## 数据

- 所有数据位于data文件夹下，文件名格式为：yyyyMMdd.csv。
- 可前往 [https://dev.xbxin.com/bond](https://dev.xbxin.com/bond) 在线查看
- 数据条目： 代码，名称，交易日期，前收盘价，开盘价，最高价，最低价，收盘价，涨跌，涨跌幅(%)，已计息天数，应计利息，剩余期限(年)，当期收益率(%)
，纯债到期收益率(%)，纯债价值，纯债溢价，纯债溢价率(%)，转股价格，转股比例，转换价值，转股溢价，转股溢价率(%)
，转股市盈率，转股市净率，套利空间，平价/底价，期限(年)，发行日期，票面利率/发行参考利率(%)，交易市场，债券类型

## 许可证

该项目基于 [Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)。