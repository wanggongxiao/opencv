import sys
from datetime import datetime


class LogTool:
    """
    一个简单的日志工具类，支持分级打印，所有输出到终端。
    日志级别（从低到高）：
        DEBUG: 调试信息
        INFO: 普通信息
        WARNING: 警告信息
        ERROR: 错误信息
        CRITICAL: 严重错误
    """
    # 日志级别常量
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    # 级别名称映射
    LEVEL_NAMES = {
        DEBUG: 'DEBUG',
        INFO: 'INFO',
        WARNING: 'WARNING',
        ERROR: 'ERROR',
        CRITICAL: 'CRITICAL'
    }

    def __init__(self, level=INFO, output_stream=sys.stdout):
        """
        初始化日志工具。

        参数：
            level: 日志级别，低于此级别的日志不会被打印
            output_stream: 输出流，默认为标准输出
        """
        self.level = level
        self.output_stream = output_stream
        self._color_enabled = hasattr(output_stream, 'isatty') and output_stream.isatty()

        # 检查OpenCV是否可用
        self.opencv_available = self._check_opencv_available()

    def _check_opencv_available(self):
        """检查OpenCV是否可用"""
        try:
            import cv2
            return True
        except ImportError:
            return False

    def _log(self, level, message, *args):
        """
        内部日志方法，根据级别判断是否输出。

        参数：
            level: 日志级别
            message: 日志消息，可以包含格式化占位符
            *args: 格式化参数
        """
        if level >= self.level:
            # 格式化消息
            formatted_message = message % args if args else message

            # 获取当前时间
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 级别名称
            level_name = self.LEVEL_NAMES.get(level, 'UNKNOWN')

            # 构建日志行
            log_line = f"[{timestamp}] [{level_name}] {formatted_message}"

            # 如果支持颜色，添加颜色
            if self._color_enabled:
                color_code = self._get_color_code(level)
                reset_code = '\033[0m'
                log_line = f"{color_code}{log_line}{reset_code}"

            # 输出日志到终端
            print(log_line, file=self.output_stream, flush=True)

    def _get_color_code(self, level):
        """
        根据日志级别返回终端颜色代码。

        参数：
            level: 日志级别

        返回：
            颜色代码字符串
        """
        if level == self.DEBUG:
            return '\033[36m'  # 青色
        elif level == self.INFO:
            return '\033[32m'  # 绿色
        elif level == self.WARNING:
            return '\033[33m'  # 黄色
        elif level == self.ERROR:
            return '\033[31m'  # 红色
        elif level == self.CRITICAL:
            return '\033[1;31m'  # 粗体红色
        else:
            return '\033[0m'  # 默认

    def set_level(self, level):
        """
        设置日志级别。

        参数：
            level: 新的日志级别
        """
        self.level = level

    def debug(self, message, *args):
        """
        打印调试级别的日志。

        参数：
            message: 日志消息
            *args: 格式化参数
        """
        self._log(self.DEBUG, message, *args)

    def info(self, message, *args):
        """
        打印信息级别的日志。

        参数：
            message: 日志消息
            *args: 格式化参数
        """
        self._log(self.INFO, message, *args)

    def warning(self, message, *args):
        """
        打印警告级别的日志。

        参数：
            message: 日志消息
            *args: 格式化参数
        """
        self._log(self.WARNING, message, *args)

    def error(self, message, *args):
        """
        打印错误级别的日志。

        参数：
            message: 日志消息
            *args: 格式化参数
        """
        self._log(self.ERROR, message, *args)

    def critical(self, message, *args):
        """
        打印严重错误级别的日志。

        参数：
            message: 日志消息
            *args: 格式化参数
        """
        self._log(self.CRITICAL, message, *args)

    def enable_color(self, enabled=True):
        """
        启用或禁用颜色输出。

        参数：
            enabled: 是否启用颜色
        """
        self._color_enabled = enabled and hasattr(self.output_stream, 'isatty') and self.output_stream.isatty()

    def log_image_info(self, image, level=INFO, message=None):
        """
        记录OpenCV图像信息到终端，不保存文件。

        参数：
            image: OpenCV图像（numpy数组）
            level: 日志级别
            message: 日志消息，如果为None则使用默认消息
        """
        if not self.opencv_available:
            self.warning("OpenCV未安装，无法处理图像")
            return False

        try:
            import cv2
            import numpy as np

            # 获取图像信息
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            dtype = image.dtype
            min_val = float(np.min(image))
            max_val = float(np.max(image))
            mean_val = float(np.mean(image))

            # 构建信息字符串
            info_msg = f"图像信息: 尺寸={width}x{height}, 通道数={channels}, "
            info_msg += f"数据类型={dtype}, 值范围=[{min_val:.2f}, {max_val:.2f}], "
            info_msg += f"平均值={mean_val:.2f}"

            # 记录日志
            if message is None:
                self._log(level, info_msg)
            else:
                self._log(level, f"{message} ({info_msg})")

            return True

        except Exception as e:
            self.error(f"处理图像信息时发生错误: {str(e)}")
            return False

    def log_image_summary(self, image, level=INFO, message=None):
        """
        记录OpenCV图像统计摘要到终端。

        参数：
            image: OpenCV图像（numpy数组）
            level: 日志级别
            message: 日志消息
        """
        if not self.opencv_available:
            self.warning("OpenCV未安装，无法处理图像")
            return False

        try:
            import cv2
            import numpy as np

            # 获取图像基本信息
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]

            # 构建摘要信息
            if message:
                self._log(level, message)

            self._log(level, f"图像尺寸: {width}x{height} 像素")
            self._log(level, f"通道数: {channels}")
            self._log(level, f"数据类型: {image.dtype}")

            # 计算统计信息
            if channels == 1:
                # 灰度图像
                self._log(level, f"最小值: {np.min(image):.2f}")
                self._log(level, f"最大值: {np.max(image):.2f}")
                self._log(level, f"平均值: {np.mean(image):.2f}")
                self._log(level, f"标准差: {np.std(image):.2f}")
            else:
                # 彩色图像
                for i in range(channels):
                    channel_data = image[:, :, i]
                    self._log(level, f"通道{i}: 最小值={np.min(channel_data):.2f}, "
                                     f"最大值={np.max(channel_data):.2f}, "
                                     f"平均值={np.mean(channel_data):.2f}")

            return True

        except Exception as e:
            self.error(f"生成图像摘要时发生错误: {str(e)}")
            return False

    def display_image_ascii(self, image, width=60, level=INFO, message=None):
        """
        将图像转换为ASCII字符在终端显示。

        参数：
            image: OpenCV图像（灰度图或彩色图）
            width: ASCII显示的宽度（字符数）
            level: 日志级别
            message: 日志消息
        """
        if not self.opencv_available:
            self.warning("OpenCV未安装，无法处理图像")
            return False

        try:
            import cv2
            import numpy as np

            # 记录开始信息
            if message:
                self._log(level, message)

            # 将图像转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 调整大小
            height = int(gray.shape[0] * width / gray.shape[1])
            resized = cv2.resize(gray, (width, height))

            # ASCII字符集（从暗到亮）
            ascii_chars = '@%#*+=-:. '

            # 归一化到0-1范围
            normalized = resized.astype(float) / 255.0

            # 转换为ASCII
            ascii_art = []
            for row in normalized:
                ascii_row = []
                for pixel in row:
                    idx = int(pixel * (len(ascii_chars) - 1))
                    ascii_row.append(ascii_chars[idx])
                ascii_art.append(''.join(ascii_row))

            # 输出ASCII图像
            self._log(level, "图像ASCII表示:")
            for line in ascii_art:
                self._log(level, line)

            return True

        except Exception as e:
            self.error(f"生成ASCII图像时发生错误: {str(e)}")
            return False

    def create_test_image(self, width=400, height=300, color=(0, 0, 255)):
        """
        创建测试图像。

        参数：
            width: 图像宽度
            height: 图像高度
            color: BGR颜色元组

        返回：
            创建的图像数组
        """
        if not self.opencv_available:
            self.warning("OpenCV未安装，无法创建测试图像")
            return None

        try:
            import cv2
            import numpy as np

            # 创建纯色图像
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:] = color

            return image
        except Exception as e:
            self.error(f"创建测试图像时发生错误: {str(e)}")
            return None


# 测试用例
if __name__ == "__main__":
    print("=== 测试LogTool类（只输出到终端）===")

    # 创建日志工具实例
    logger = LogTool()
    logger.set_level(LogTool.ERROR)
    print("\n1. 默认INFO级别下的日志输出：")
    logger.debug("这条debug日志不会显示")
    logger.info("这是一条info日志")
    logger.warning("这是一条warning日志")
    logger.error("这是一条error日志")
    logger.critical("这是一条critical日志")

    print("\n2. 设置级别为DEBUG后的日志输出：")

    logger.debug("现在这条debug日志会显示")
    logger.info("info日志仍然显示")

    print("\n3. 设置级别为ERROR后的日志输出：")
    logger.warning("这条warning日志不会显示")
    logger.error("只有error和critical级别会显示")
    logger.critical("critical日志显示")

    print("\n4. 带参数的日志消息：")
    logger.info("用户 %s 在 %s 登录", "张三", "2023-10-01 10:00:00")
    logger.error("文件 %s 打开失败，错误码: %d", "test.txt", 404)

    print("\n5. 测试颜色输出（如果终端支持）：")
    logger.enable_color(True)
    logger.debug("青色调试信息")
    logger.info("绿色普通信息")
    logger.warning("黄色警告信息")
    logger.error("红色错误信息")
    logger.critical("粗体红色严重错误")

    print("\n6. 创建不同输出流的日志工具：")
    # 可以输出到标准错误
    error_logger = LogTool(output_stream=sys.stderr)
    error_logger.error("这条错误日志输出到stderr")

    print("\n7. 测试OpenCV图像信息输出：")
    if logger.opencv_available:
        # 创建测试图像
        test_image = logger.create_test_image(width=640, height=480, color=(255, 0, 0))

        if test_image is not None:
            # 记录图像信息
            logger.info("测试图像创建成功")
            logger.log_image_info(test_image, message="测试图像")

            # 记录图像摘要
            logger.log_image_summary(test_image, message="图像统计摘要：")

            # 创建灰度图像测试
            import cv2

            gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            logger.log_image_summary(gray_image, message="灰度图像统计摘要：")

            # 测试ASCII显示
            print("\n8. 测试图像ASCII显示：")
            small_image = logger.create_test_image(width=20, height=10, color=(0, 255, 0))
            logger.display_image_ascii(small_image, width=20,
                                       message="绿色测试图像的ASCII表示：")

            # 测试不同颜色的图像
            blue_image = logger.create_test_image(width=80, height=40, color=(255, 0, 0))
            logger.display_image_ascii(blue_image, width=40,
                                       message="蓝色测试图像的ASCII表示：")

            logger.info("OpenCV图像输出测试完成")
        else:
            logger.error("无法创建测试图像")
    else:
        logger.warning("OpenCV未安装，跳过图像输出测试")
        logger.info("如需使用图像输出功能，请安装OpenCV: pip install opencv-python")

    print("\n=== 测试完成 ===")

    # 输出总结
    logger.info("所有日志都只输出到终端，不保存任何文件")
    logger.info("使用enable_color(True/False)控制颜色输出")
    logger.info("使用set_level()方法设置日志级别")
