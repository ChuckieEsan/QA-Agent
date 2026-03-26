"""
日志模块 - 统一的日志配置和管理
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from src.config.setting import settings


class CustomFormatter(logging.Formatter):
    """自定义日志格式器，支持彩色输出"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[94m',    # 蓝色
        'INFO': '\033[92m',     # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',    # 红色
        'CRITICAL': '\033[41m', # 红底白字
        'RESET': '\033[0m'      # 重置颜色
    }
    
    # 图标 (不使用 Emoji)
    ICONS = {
        'DEBUG': '[D]',
        'INFO': '[I]',
        'WARNING': '[W]',
        'ERROR': '[E]',
        'CRITICAL': '[C]'
    }
    
    def __init__(self, use_color: bool = True):
        """
        初始化格式化器
        
        Args:
            use_color: 是否使用颜色输出
        """
        self.use_color = use_color
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        super().__init__(fmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 保存原始信息
        original_levelname = record.levelname
        original_msg = record.msg
        
        # 添加图标
        icon = self.ICONS.get(record.levelname, '')
        if icon:
            record.msg = f"{icon} {record.msg}"
        
        # 添加颜色
        if self.use_color and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # 格式化
        result = super().format(record)
        
        # 恢复原始信息
        record.levelname = original_levelname
        record.msg = original_msg
        
        return result


class GovPulseLogger:
    """GovPulse项目日志管理器"""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, name: str = "govpulse") -> logging.Logger:
        """
        获取日志记录器（单例模式）
        
        Args:
            name: 日志器名称，通常使用模块名
        
        Returns:
            配置好的日志记录器
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 创建新日志器
        logger = logging.getLogger(name)
        
        # 设置日志级别
        log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 控制台处理器
            console_handler = cls._create_console_handler()
            logger.addHandler(console_handler)
            
            # 文件处理器（如果启用）
            if settings.logging.file_enabled:
                file_handler = cls._create_file_handler()
                logger.addHandler(file_handler)
        
        # 存储并返回
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _create_console_handler(cls) -> logging.StreamHandler:
        """创建控制台处理器"""
        handler = logging.StreamHandler(sys.stdout)
        
        # 判断是否为终端，决定是否使用颜色
        use_color = sys.stdout.isatty()
        formatter = CustomFormatter(use_color=use_color)
        
        handler.setFormatter(formatter)
        handler.setLevel(getattr(logging, settings.logging.level.upper(), logging.INFO))
        
        return handler
    
    @classmethod
    def _create_file_handler(cls) -> logging.Handler:
        """创建文件处理器"""
        log_file = settings.logging.file_path
        
        # 确保日志目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用轮转文件处理器
        handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count,
            encoding='utf-8'
        )
        
        # 文件日志使用简单格式
        formatter = logging.Formatter(settings.logging.format)
        handler.setFormatter(formatter)
        
        # 文件日志通常记录所有级别
        handler.setLevel(logging.DEBUG)
        
        return handler
    
    @classmethod
    def update_log_level(cls, level: str):
        """
        动态更新所有日志器的日志级别
        
        Args:
            level: 新的日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        for logger in cls._loggers.values():
            logger.setLevel(log_level)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(log_level)
        
        # 更新配置
        settings.logging.level = level
    
    @classmethod
    def add_custom_handler(cls, handler: logging.Handler, logger_name: str = None):
        """
        为日志器添加自定义处理器
        
        Args:
            handler: 日志处理器
            logger_name: 日志器名称，None表示所有日志器
        """
        if logger_name:
            logger = cls.get_logger(logger_name)
            logger.addHandler(handler)
        else:
            for logger in cls._loggers.values():
                logger.addHandler(handler)


class RequestIdFilter(logging.Filter):
    """为日志添加请求ID过滤器（用于追踪请求）"""
    
    def __init__(self):
        self.request_id = "N/A"
        super().__init__()
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self.request_id
        return True
    
    def set_request_id(self, request_id: str):
        """设置当前请求ID"""
        self.request_id = request_id


class PerformanceLogger:
    """性能监控专用日志器"""
    
    def __init__(self):
        self.logger = GovPulseLogger.get_logger("performance")
        self._start_times = {}
    
    def start_timer(self, operation: str):
        """开始计时"""
        self._start_times[operation] = datetime.now()
        self.logger.debug(f"⏱️  开始: {operation}")
    
    def end_timer(self, operation: str, extra_info: Dict[str, Any] = None):
        """结束计时并记录耗时"""
        if operation not in self._start_times:
            self.logger.warning(f"未找到开始时间: {operation}")
            return
        
        elapsed = datetime.now() - self._start_times[operation]
        
        log_data = {
            "operation": operation,
            "elapsed_seconds": elapsed.total_seconds(),
            "elapsed_ms": elapsed.total_seconds() * 1000
        }
        
        if extra_info:
            log_data.update(extra_info)
        
        self.logger.info(f"⏱️  完成: {operation} - 耗时: {elapsed.total_seconds():.3f}s")
        
        # 如果耗时过长，记录警告
        if elapsed.total_seconds() > 5.0:
            self.logger.warning(f"⚠️  操作 {operation} 耗时过长: {elapsed.total_seconds():.3f}s")
        
        del self._start_times[operation]
        
        return log_data
    
    def log_metric(self, name: str, value: float, unit: str = ""):
        """记录指标"""
        self.logger.info(f"📊 指标: {name} = {value} {unit}".strip())


def get_logger(name: str = "govpulse") -> logging.Logger:
    """
    获取日志记录器（主要导出函数）
    
    Args:
        name: 日志器名称，通常使用 __name__
    
    Example:
        logger = get_logger(__name__)
        logger.info("这是一条日志")
    """
    return GovPulseLogger.get_logger(name)


def setup_logging():
    """初始化日志系统（可在应用启动时调用）"""
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info(f"🚀 启动 GovPulse 系统 v{settings.version}")
    logger.info(f"📁 项目根目录: {settings.paths.project_root}")
    logger.info(f"📝 日志级别: {settings.logging.level}")
    logger.info(f"💾 日志文件: {settings.logging.file_path}")
    logger.info("=" * 60)


# ========== 上下文管理器 ==========

class LoggingContext:
    """日志上下文管理器，用于临时修改日志级别"""
    
    def __init__(self, level: str, logger_name: str = None):
        """
        Args:
            level: 临时日志级别
            logger_name: 日志器名称，None表示所有日志器
        """
        self.level = level
        self.logger_name = logger_name
        self.original_levels = {}
    
    def __enter__(self):
        """进入上下文，保存原级别并设置新级别"""
        if self.logger_name:
            logger = get_logger(self.logger_name)
            self.original_levels[self.logger_name] = logger.level
            logger.setLevel(getattr(logging, self.level.upper()))
        else:
            # 保存所有日志器的级别
            for name, logger in GovPulseLogger._loggers.items():
                self.original_levels[name] = logger.level
                logger.setLevel(getattr(logging, self.level.upper()))
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原级别"""
        for name, level in self.original_levels.items():
            logger = get_logger(name)
            logger.setLevel(level)


if __name__ == "__main__":
    """测试日志系统"""
    setup_logging()
    
    logger = get_logger(__name__)
    
    # 测试不同级别的日志
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    
    # 测试带参数的日志
    user_query = "雨露计划什么时候发放"
    similarity = 0.85
    logger.info(f"检索查询: '{user_query}', 最高相似度: {similarity:.2%}")
    
    # 测试性能日志
    perf_logger = PerformanceLogger()
    perf_logger.start_timer("向量检索")
    # 模拟耗时操作
    import time
    time.sleep(0.1)
    perf_logger.end_timer("向量检索", {"结果数量": 10})
    
    logger.info("日志系统测试完成")