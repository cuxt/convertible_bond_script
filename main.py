from datetime import datetime, timedelta

from tqdm import tqdm

from utils.config_utils import get_config
from utils.logging_utils import setup_logging
from src.bond_calc import BondCalc


def main():
    # 日志
    log_path = 'log.log'
    setup_logging(log_path)

    # 配置
    setting = get_config('dev')
    configurations = setting.configurations

    for config in configurations:
        get_result(config)


def get_result(config):
    # 数据周期：上周五-本周五
    # today = datetime.now()
    today=datetime(2023, 1, 1)
    last_friday, this_friday = get_friday_range(today)

    bond = BondCalc()

    total_days = (this_friday - last_friday).days + 1
    with tqdm(total=total_days, desc="进度", dynamic_ncols=True) as pbar:
        for i in range(total_days):
            current_date = last_friday + timedelta(days=i)

            str_date = current_date.strftime('%Y%m%d')
            pbar.set_postfix_str(str_date)

            if config.action == 'ratio':
                data = bond.ratio(current_date)

                # print(data)
            pbar.update(1)

    bond.close()


def get_friday_range(today=None):
    days_ahead = 4 - today.weekday()

    this_friday = today + timedelta(days=days_ahead)
    last_friday = this_friday - timedelta(days=7)

    return last_friday, this_friday


if __name__ == '__main__':
    main()
