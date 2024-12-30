import logging
import os


def setup_logging(log_file):
    """
    设置日志记录函数示例
    """
    logs_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # 设置日志文件路径
    log_path = os.path.join(logs_dir, log_file)

    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def log_info(message):
    """
    记录信息日志函数示例
    """
    logging.info(message)


def log_error(message):
    """
    记录错误日志函数示例
    """
    logging.error(message)
