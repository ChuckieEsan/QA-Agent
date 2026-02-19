from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

class BaseDBClient(ABC):
    """
    数据库客户端基类. 定义通用接口, 解耦数据库方法的具体实现
    """
    
    def __init__(self, config):
        self.config = config
        self.client: Optional[Any] = None
        
    @abstractmethod
    def connect(self) -> None:
        """建立数据库连接"""
        pass
    
    @abstractmethod
    def get_client(self) -> Any:
        """获取初始化完成的客户端实例"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭连接/释放资源"""
        pass
    
    def __enter__(self):
        """支持上下文管理器"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动释放资源"""
        self.close()