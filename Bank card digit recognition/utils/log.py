import sys
from datetime import datetime
from logging import DEBUG

"""
-- 创建一个log类
-- 


"""

class logutils:
    """
    一个简单的日志工具类，支持分级打印，所有输出到终端。
    日志级别（从低到高）：
        DEBUG: 调试信息
        INFO: 普通信息
        WARNING: 警告信息
        ERROR: 错误信息
        CRITICAL: 严重错误
    """
    DEBUG    = 10
    INFO     = 20
    WARNING  = 30
    ERROR    = 40
    CRITICAL = 50

    LOG_level = {
        DEBUG:"DEBUG",
        INFO:"INFO",
        WARNING:"WARNING",
        ERROR:"ERROR",
        CRITICAL:"CRITICAL"
    }

    def __init__(self,level= INFO,output_stream=sys.stdout):
        self.level = level


    def _log(self,level,msg):
        if level > self.level:
            print(msg)