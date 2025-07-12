import csv
from pathlib import Path
import pandas as pd
from environs import Env
from iFinD import IFinD


def update_nature_of_business():
    # 设置路径
    root_path = Path(__file__).resolve().parent.parent.parent
    data_path = root_path / "data"
    print(data_path)

    # 获取环境变量
    env = Env()
    env.read_env()
    user = env.json("THS_USER")

    # 初始化 IFinD
    ifind = IFinD(user["username"], user["password"])

    # 获取所有 CSV 文件
    csv_files = list(data_path.glob("*.csv"))


    for csv_file in csv_files:
        # print(f"处理文件: {csv_file.name}")

        # 读取 CSV 文件
        df = pd.read_csv(csv_file)

        # 检查是否已有"发行人企业性质"列
        if "发行人企业性质" in df.columns and not df["发行人企业性质"].isna().all():
            # print(f"文件 {csv_file.name} 已有企业性质数据，跳过")
            continue

        # 获取所有债券代码
        codes = df["代码"].tolist()
        print(codes)
        # 获取企业性质
        try:
            print(f"处理文件: {csv_file.name}")
            nature_of_business = ifind.get_nature_of_business(codes)
            print(nature_of_business)

            # 添加或更新"发行人企业性质"列
            df["发行人企业性质"] = nature_of_business

            # 保存更新后的文件
            df.to_csv(csv_file, index=False, encoding="utf-8")
            print(f"文件 {csv_file.name} 更新完成")

        except Exception as e:
            print(f"处理文件 {csv_file.name} 时发生错误: {str(e)}")
            continue


if __name__ == "__main__":
    update_nature_of_business()