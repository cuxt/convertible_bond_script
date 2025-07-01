from pathlib import Path

import numpy as np
import toml
from scipy import optimize
from scipy.stats import linregress
from sklearn.metrics import r2_score
from src.bond.dao import BondDao


class BondCalc(object):
    def __init__(self):
        self.root_path = Path(__file__).resolve().parent.parent.parent
        self.config = self.load_config()
        self.columns_str = self.format_column()

    def load_config(self):
        config_path = self.root_path / "config" / "csv.toml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        return config

    def format_column(self):
        columns_dict = {f'"{k}"': v for k, v in self.config["columns"].items()}
        columns_str = str(columns_dict).replace("'", "")
        return columns_str

    def ratio(self, filename, conditions):
        file_path = self.root_path / "data" / f"{filename}.csv"

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

        if conditions and "ratio_total" in conditions and conditions["ratio_total"]:
            query = base_query + " WHERE " + " AND ".join(conditions["ratio_total"])
        else:
            query = base_query

        # 交易代码
        with BondDao() as dao:
            code = dao.query(query)

        if not isinstance(code, list):
            return filename, None, None
        if conditions and "main" in conditions and conditions["main"]:
            query = base_query + " WHERE " + " AND ".join(conditions["main"])
        else:
            query = base_query
        total = len(code)
        with BondDao() as dao:
            data_remain = dao.query(query)
        remain = len(data_remain)
        return filename, remain, total

    def math_func(self, filename, conditions, column, model="median"):
        file_path = self.root_path / "data" / f"{filename}.csv"

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

        if conditions and "main" in conditions and conditions["main"]:
            query = base_query + " WHERE " + " AND ".join(conditions["main"])
        else:
            query = base_query

        with BondDao() as dao:
            data = dao.query(query)

        if isinstance(data, list) and len(data) == 0:
            return filename, None
        elif isinstance(data, int):
            print(f"\n{filename}:{data}")
            return filename, None

        # 过滤None值
        filtered_data = [x[0] for x in data if x[0] is not None]

        if not filtered_data:
            return filename, None

        try:
            if model == "median":
                return filename, np.median(filtered_data)
            elif model == "avg":
                return filename, np.mean(filtered_data)
            elif model == "max":
                return filename, max(filtered_data)
            elif model == "min":
                return filename, min(filtered_data)
            elif model == "std_0":
                # 有偏样本标准差
                return filename, np.std(filtered_data, ddof=0)
            elif model == "std_1":
                # 无偏样本标准差
                return filename, np.std(filtered_data, ddof=1)
        except Exception as e:
            print("\n", f"match_func({model}): ", e)
            print(data)

    def custom(self, filename, conditions):
        file_path = self.root_path / "data" / f"{filename}.csv"
        query = f"""
            {conditions['main'][0]} FROM read_csv(
                '{file_path}',
                delim='{self.config['type']['delim']}',
                quote='{self.config['type']['quote']}',
                escape='{self.config['type']['escape']}',
                header={str(self.config['type']['header']).lower()},
                ignore_errors={str(self.config['type']['ignore_errors']).lower()},
                columns={self.columns_str}
            )
        """
        with BondDao() as dao:
            data = dao.query(query)

        if not isinstance(data, list):
            return filename, None
        print(data)
        try:
            value = data[0][0]
            return filename, value
        except IndexError:
            print("Index out of range. Please check the list and index.")
            return filename, None

    def math_func_with_fitting(self, filename, conditions, column, model="median", fit_config=None):
        """
        带拟合功能的数学函数计算
        """
        file_path = self.root_path / "data" / f"{filename}.csv"

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

        if conditions and "main" in conditions and conditions["main"]:
            query = base_query + " WHERE " + " AND ".join(conditions["main"])
        else:
            query = base_query

        with BondDao() as dao:
            data = dao.query(query)

        if isinstance(data, list) and len(data) == 0:
            return filename, None, None
        elif isinstance(data, int):
            print(f"\n{filename}:{data}")
            return filename, None, None

        # 过滤None值
        filtered_data = [x[0] for x in data if x[0] is not None]

        if not filtered_data:
            return filename, None, None

        # 检查是否需要触发拟合
        if fit_config and fit_config.get("enable_fitting", False):
            min_threshold = fit_config.get("min_data_threshold", 5)
            if len(filtered_data) < min_threshold:
                print(f"数据量不足({len(filtered_data)} < {min_threshold})，尝试拟合...")

                # 获取拟合数据
                fit_result = self._perform_fitting(filename, fit_config, column)
                if fit_result is not None:
                    fitted_value, r_squared = fit_result
                    return filename, fitted_value, r_squared

        # 原始计算逻辑
        try:
            if model == "median":
                return filename, np.median(filtered_data), None
            elif model == "avg":
                return filename, np.mean(filtered_data), None
            elif model == "max":
                return filename, max(filtered_data), None
            elif model == "min":
                return filename, min(filtered_data), None
            elif model == "std_0":
                return filename, np.std(filtered_data, ddof=0), None
            elif model == "std_1":
                return filename, np.std(filtered_data, ddof=1), None
        except Exception as e:
            print("\n", f"match_func({model}): ", e)
            print(data)
            return filename, None, None

    def _perform_fitting(self, filename, fit_config, column):
        """
        执行拟合操作
        """
        try:
            # 获取拟合用的历史数据（X-Y数据对）
            historical_data_pairs = self._get_historical_data_for_fitting(fit_config, column, filename)
            # print(historical_data_pairs)
            if not historical_data_pairs or len(historical_data_pairs) < 3:
                print("历史数据不足，无法进行拟合")
                return None

            # 分离X和Y数据
            x_data = np.array([pair[0] for pair in historical_data_pairs])
            y_data = np.array([pair[1] for pair in historical_data_pairs])

            # 执行拟合
            fit_result = self._fit_data(x_data, y_data, fit_config)

            if fit_result is None:
                return None

            fitted_params, r_squared, fit_type = fit_result

            # 检查拟合质量
            min_r_squared = fit_config.get("quality", {}).get("min_r_squared", 0.6)
            if r_squared < min_r_squared:
                print(f"拟合质量不佳 (R² = {r_squared:.3f} < {min_r_squared})")

                fallback = fit_config.get("quality", {}).get("fallback_on_poor_fit", True)
                if not fallback:
                    return None

            # 获取预测目标值
            predict_conditions = fit_config.get("predict_conditions", {})
            predict_value = predict_conditions.get("predict_value", 100)

            # 预测指定X值对应的Y值
            predicted_value = self._predict_value(predict_value, fitted_params, fit_type)

            # 输出拟合信息
            if fit_config.get("quality", {}).get("include_quality_info", True):
                predict_column = predict_conditions.get("predict_column", "转换价值")
                print(f"拟合类型: {fit_type}, R²: {r_squared:.3f}")
                print(f"预测当{predict_column}={predict_value}时，{column}={predicted_value:.3f}")

            # 返回预测值和R²值
            return predicted_value, r_squared

        except Exception as e:
            print(f"拟合过程出错: {e}")
            return None

    def _get_historical_data_for_fitting(self, fit_config, column, current_filename):
        """
        获取用于拟合的历史数据
        """
        # 获取拟合条件和排除条件
        fit_conditions = fit_config.get("fit_conditions", {})
        exclude_conditions = fit_config.get("exclude_conditions", {})
        predict_conditions = fit_config.get("predict_conditions", {})

        # 获取预测列信息
        predict_column = predict_conditions.get("predict_column", "转换价值")
        target_column = predict_conditions.get("target_column", column)

        # 获取历史数据用于拟合
        from datetime import datetime, timedelta
        import os

        x_data = []  # 预测列的值（如转换价值）
        y_data = []  # 目标列的值（如转股溢价率）

        # 解析当前文件的日期
        try:
            current_date = datetime.strptime(current_filename, "%Y%m%d").date()
        except ValueError:
            print(f"无法解析当前文件日期: {current_filename}")
            return []

        # 获取data目录下所有可用的csv文件
        data_dir = self.root_path / "data"
        available_files = []

        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                try:
                    # 从文件名提取日期
                    date_str = file.stem
                    if len(date_str) == 8 and date_str.isdigit():
                        file_date = datetime.strptime(date_str, "%Y%m%d").date()
                        # 只选择当前日期之前的文件
                        if file_date < current_date:
                            available_files.append((file_date, file))
                except:
                    continue

        # 按日期排序（最新的在前），取配置指定数量的历史文件
        available_files.sort(key=lambda x: x[0], reverse=True)
        historical_files_count = fit_config.get("historical_files_count", 30)  # 从配置读取，默认30
        recent_files = available_files[:historical_files_count]

        print(f"当前计算日期: {current_date}")
        print(f"找到 {len(recent_files)} 个可用的历史数据文件（配置使用最近{historical_files_count}个文件）")

        for file_date, file_path in recent_files:
            date_str = file_date.strftime("%Y%m%d")

            try:
                # 构建拟合数据查询（包含X轴和Y轴数据）
                base_query = f"""
                    SELECT "{predict_column}", "{target_column}" FROM read_csv(
                        '{file_path}',
                        delim='{self.config['type']['delim']}',
                        quote='{self.config['type']['quote']}',
                        escape='{self.config['type']['escape']}',
                        header={str(self.config['type']['header']).lower()},
                        ignore_errors={str(self.config['type']['ignore_errors']).lower()},
                        columns={self.columns_str}
                    )
                """

                # 添加拟合筛选条件
                conditions_list = []
                if fit_conditions and "main" in fit_conditions and fit_conditions["main"]:
                    conditions_list.extend(fit_conditions["main"])

                # 添加排除条件（使用NOT条件）
                if exclude_conditions and "main" in exclude_conditions and exclude_conditions["main"]:
                    for exclude_condition in exclude_conditions["main"]:
                        # 将排除条件转换为NOT条件
                        conditions_list.append(f"NOT ({exclude_condition})")

                if conditions_list:
                    query = base_query + " WHERE " + " AND ".join(conditions_list)
                else:
                    query = base_query

                with BondDao() as dao:
                    data = dao.query(query)

                if isinstance(data, list) and len(data) > 0:
                    daily_count = 0
                    for row in data:
                        if row[0] is not None and row[1] is not None:
                            try:
                                x_val = float(row[0])  # 转换价值
                                y_val = float(row[1])  # 转股溢价率
                                x_data.append(x_val)
                                y_data.append(y_val)
                                daily_count += 1
                            except (ValueError, TypeError):
                                continue

                    # if daily_count > 0:
                    #     print(f"{date_str}: 获取到 {daily_count} 条拟合数据")

            except Exception as e:
                print(f"获取拟合历史数据失败 {date_str}: {e}")
                continue

        print(f"总共获取到 {len(x_data)} 条拟合数据点")

        # 返回X-Y数据对，而不是单纯的Y数据
        if len(x_data) > 0 and len(y_data) > 0:
            return list(zip(x_data, y_data))

        return []

    def _fit_data(self, x_data, y_data, fit_config):
        """
        根据配置执行数据拟合
        """
        fit_type = fit_config.get("fit_type", "linear")
        params = fit_config.get("params", {})

        try:
            if fit_type == "linear":
                return self._linear_fit(x_data, y_data)
            elif fit_type == "polynomial":
                degree = params.get("polynomial_degree", 2)
                return self._polynomial_fit(x_data, y_data, degree)
            elif fit_type == "exponential":
                return self._exponential_fit(x_data, y_data)
            elif fit_type == "logarithmic":
                return self._logarithmic_fit(x_data, y_data)
            elif fit_type == "power":
                return self._power_fit(x_data, y_data)
            else:
                print(f"不支持的拟合类型: {fit_type}")
                return None

        except Exception as e:
            print(f"拟合失败 ({fit_type}): {e}")
            return None

    def _linear_fit(self, x_data, y_data):
        """线性拟合"""
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        r_squared = r_value ** 2
        return [slope, intercept], r_squared, "linear"

    def _polynomial_fit(self, x_data, y_data, degree):
        """多项式拟合"""
        coeffs = np.polyfit(x_data, y_data, degree)
        y_pred = np.polyval(coeffs, x_data)
        r_squared = r2_score(y_data, y_pred)
        return coeffs, r_squared, f"polynomial_{degree}"

    def _exponential_fit(self, x_data, y_data):
        """指数拟合: y = a * exp(b * x)"""

        def exp_func(x, a, b):
            return a * np.exp(b * x)

        # 初始猜测
        popt, _ = optimize.curve_fit(exp_func, x_data, y_data, maxfev=1000)
        y_pred = exp_func(x_data, *popt)
        r_squared = r2_score(y_data, y_pred)
        return popt, r_squared, "exponential"

    def _logarithmic_fit(self, x_data, y_data):
        """对数拟合: y = a * ln(x) + b"""
        # 避免x=0的情况
        x_data_safe = np.where(x_data <= 0, 0.001, x_data)

        def log_func(x, a, b):
            return a * np.log(x) + b

        popt, _ = optimize.curve_fit(log_func, x_data_safe, y_data, maxfev=1000)
        y_pred = log_func(x_data_safe, *popt)
        r_squared = r2_score(y_data, y_pred)
        return popt, r_squared, "logarithmic"

    def _power_fit(self, x_data, y_data):
        """幂函数拟合: y = a * x^b"""
        # 避免x=0的情况
        x_data_safe = np.where(x_data <= 0, 0.001, x_data)

        def power_func(x, a, b):
            return a * np.power(x, b)

        popt, _ = optimize.curve_fit(power_func, x_data_safe, y_data, maxfev=1000)
        y_pred = power_func(x_data_safe, *popt)
        r_squared = r2_score(y_data, y_pred)
        return popt, r_squared, "power"

    def _predict_value(self, x, params, fit_type):
        """根据拟合参数预测值"""
        if fit_type == "linear":
            slope, intercept = params
            return slope * x + intercept
        elif fit_type.startswith("polynomial"):
            return np.polyval(params, x)
        elif fit_type == "exponential":
            a, b = params
            return a * np.exp(b * x)
        elif fit_type == "logarithmic":
            a, b = params
            x_safe = max(x, 0.001)
            return a * np.log(x_safe) + b
        elif fit_type == "power":
            a, b = params
            x_safe = max(x, 0.001)
            return a * np.power(x_safe, b)
        else:
            return None


def main():
    conditions = {
        "main": [
            "\"债券类型\" = '可转债'",
            '"转换价值" <= 80',
        ],
        "ratio_total": ["\"债券类型\" = '可转债'"],
    }
    bond_calc = BondCalc()
    data = bond_calc.math_func("20180102", conditions, "转股溢价率(%)")
    print(data)


if __name__ == "__main__":
    main()
