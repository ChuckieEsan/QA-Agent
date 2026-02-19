"""
Milvus 向量数据库客户端实现
提供与 BaseDBClient 接口的完整实现，封装 Milvus 操作
"""

from typing import Optional, Dict, List
from pymilvus import MilvusClient
from src.app.infra.db.base_db import BaseDBClient
from src.app.infra.utils.logger import get_logger
from src.config.setting import settings

logger = get_logger(__name__)


class MilvusDBClient(BaseDBClient):
    """
    Milvus 数据库客户端
    实现 BaseDBClient 接口，提供 Milvus 数据库的连接和操作封装
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 Milvus 客户端

        Args:
            config: 可选的配置字典，如果为 None 则使用 settings.vectordb 配置
        """
        # 使用传入的配置或默认配置
        config = config or {
            "db_path": str(settings.vectordb.db_path),
            "collection_name": settings.vectordb.collection_name,
            "vector_dimension": settings.vectordb.vector_dimension,
            "metric_type": settings.vectordb.metric_type,
            "enable_dynamic_field": settings.vectordb.enable_dynamic_field,
        }

        super().__init__(config)
        self.db_path: str = self.config.get("db_path")
        self.collection_name: str = self.config.get("collection_name")
        self.vector_dimension: int = self.config.get("vector_dimension", 1024)
        self.metric_type: str = self.config.get("metric_type", "COSINE")
        self.enable_dynamic_field: bool = self.config.get("enable_dynamic_field", True)

        # Milvus 客户端实例
        self._client: Optional[MilvusClient] = None

    def connect(self) -> None:
        """
        建立数据库连接
        初始化 Milvus 客户端
        """
        try:
            logger.info(f"🔌 连接 Milvus Lite: {self.db_path}")

            # 创建 Milvus 客户端
            self._client = MilvusClient(uri=self.db_path)

            logger.info(f"✅ Milvus 连接成功")

            # 检查集合是否存在，不存在则创建
            self._ensure_collection_exists()

        except Exception as e:
            logger.error(f"❌ Milvus 连接失败: {e}")
            raise

    def get_client(self) -> MilvusClient:
        """
        获取初始化完成的客户端实例

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 如果客户端未初始化
        """
        if self._client is None:
            raise RuntimeError("Milvus 客户端未初始化，请先调用 connect()")
        return self._client

    def close(self) -> None:
        """
        关闭连接/释放资源
        """
        try:
            if self._client is not None:
                logger.info("🔌 关闭 Milvus 连接")
                # MilvusClient Lite 不需要显式关闭，但可以清理引用
                self._client = None
                logger.info("✅ Milvus 连接已关闭")
        except Exception as e:
            logger.error(f"❌ 关闭 Milvus 连接时出错: {e}")

    # ==================== 集合管理方法 ====================

    def _ensure_collection_exists(self) -> None:
        """
        确保集合存在，不存在则创建
        """
        try:
            # 检查集合是否存在
            if not self._client.has_collection(self.collection_name):
                logger.info(f"🔨 集合 {self.collection_name} 不存在，正在创建...")

                # 创建集合
                self._client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.vector_dimension,
                    metric_type=self.metric_type,
                    auto_id=True,
                    enable_dynamic_field=self.enable_dynamic_field,
                )

                logger.info(f"✅ 集合 {self.collection_name} 创建成功")
            else:
                logger.info(f"✅ 集合 {self.collection_name} 已存在")

        except Exception as e:
            logger.error(f"❌ 集合管理失败: {e}")
            raise

    def has_collection(self) -> bool:
        """
        检查集合是否存在

        Returns:
            bool: 集合是否存在
        """
        return self._client.has_collection(self.collection_name)

    def drop_collection(self) -> None:
        """
        删除集合
        """
        if self._client.has_collection(self.collection_name):
            logger.warning(f"🗑️ 正在删除集合 {self.collection_name}")
            self._client.drop_collection(self.collection_name)
            logger.info(f"✅ 集合 {self.collection_name} 已删除")

    def describe_collection(self) -> Dict:
        """
        获取集合详细信息

        Returns:
            Dict: 集合的详细信息
        """
        return self._client.describe_collection(self.collection_name)

    # ==================== 数据操作方法 ====================

    def insert(
        self,
        data: List[Dict],
        batch_size: Optional[int] = None
    ) -> Dict:
        """
        插入数据到集合

        Args:
            data: 要插入的数据列表，每个元素是包含向量和元数据的字典
            batch_size: 批量插入大小（可选）

        Returns:
            Dict: 插入结果，包含插入的 IDs

        Example:
            >>> client.insert([
            ...     {
            ...         "vector": [0.1, 0.2, ..., 0.3],
            ...         "text": "示例文本",
            ...         "department": "部门名称",
            ...         "metadata": {"title": "标题", "time": "2024-01-01"}
            ...     }
            ... ])
        """
        try:
            logger.info(f"📥 向集合 {self.collection_name} 插入 {len(data)} 条数据")

            # 执行插入
            result = self._client.insert(
                collection_name=self.collection_name,
                data=data,
                batch_size=batch_size
            )

            logger.info(f"✅ 插入成功，插入 {result.get('insert_count', 0)} 条数据")
            return result

        except Exception as e:
            logger.error(f"❌ 插入数据失败: {e}")
            raise

    def search(
        self,
        vectors: List[List[float]],
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        **kwargs
    ) -> List[List[Dict]]:
        """
        向量相似度搜索

        Args:
            vectors: 查询向量列表
            top_k: 返回结果数量
            output_fields: 要返回的字段列表
            filter_expr: 过滤表达式（可选）
            **kwargs: 其他搜索参数

        Returns:
            List[List[Dict]]: 搜索结果列表

        Example:
            >>> results = client.search(
            ...     vectors=[[0.1, 0.2, ..., 0.3]],
            ...     top_k=5,
            ...     output_fields=["text", "department", "metadata"]
            ... )
        """
        try:
            # 执行搜索
            results = self._client.search(
                collection_name=self.collection_name,
                data=vectors,
                limit=top_k,
                output_fields=output_fields or ["*"],
                filter=filter_expr,
                **kwargs
            )

            return results

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            raise

    def query(
        self,
        filter_expr: str,
        output_fields: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        条件查询（非向量搜索）

        Args:
            filter_expr: 过滤表达式
            output_fields: 要返回的字段列表
            limit: 返回结果数量限制

        Returns:
            List[Dict]: 查询结果列表
        """
        try:
            results = self._client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["*"],
                limit=limit
            )
            return results

        except Exception as e:
            logger.error(f"❌ 条件查询失败: {e}")
            raise

    def delete(
        self,
        filter_expr: str
    ) -> Dict:
        """
        根据过滤条件删除数据

        Args:
            filter_expr: 过滤表达式

        Returns:
            Dict: 删除结果
        """
        try:
            logger.info(f"🗑️ 删除满足条件的数据: {filter_expr}")
            result = self._client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
            logger.info(f"✅ 删除完成: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ 删除数据失败: {e}")
            raise

    def upsert(
        self,
        data: List[Dict],
        batch_size: Optional[int] = None
    ) -> Dict:
        """
        更新或插入数据（如果主键存在则更新，否则插入）

        Args:
            data: 要更新或插入的数据列表
            batch_size: 批量处理大小（可选）

        Returns:
            Dict: 操作结果
        """
        try:
            logger.info(f"🔄 更新/插入 {len(data)} 条数据")
            result = self._client.upsert(
                collection_name=self.collection_name,
                data=data,
                batch_size=batch_size
            )
            logger.info(f"✅ 更新/插入完成")
            return result

        except Exception as e:
            logger.error(f"❌ 更新/插入失败: {e}")
            raise

    # ==================== 统计信息方法 ====================

    def get_collection_stats(self) -> Dict:
        """
        获取集合统计信息

        Returns:
            Dict: 集合的统计信息
        """
        try:
            stats = self._client.get_collection_stats(self.collection_name)
            return stats
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            raise

    def get_entity_count(self) -> int:
        """
        获取集合中的实体数量

        Returns:
            int: 实体数量
        """
        stats = self.get_collection_stats()
        return stats.get("row_count", 0)

    # ==================== 工具方法 ====================

    def create_index(
        self,
        field_name: str = "vector",
        index_type: str = "AUTOINDEX",
        metric_type: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        创建索引（Milvus Lite 通常自动创建）

        Args:
            field_name: 字段名称
            index_type: 索引类型
            metric_type: 度量类型
            **kwargs: 其他索引参数
        """
        try:
            logger.info(f"🔨 为字段 {field_name} 创建索引")
            self._client.create_index(
                collection_name=self.collection_name,
                field_name=field_name,
                index_type=index_type,
                metric_type=metric_type or self.metric_type,
                **kwargs
            )
            logger.info(f"✅ 索引创建成功")
        except Exception as e:
            logger.error(f"❌ 创建索引失败: {e}")
            raise

    def load_collection(self) -> None:
        """
        加载集合到内存（Milvus Lite 通常自动加载）
        """
        try:
            logger.info(f"💾 加载集合 {self.collection_name} 到内存")
            self._client.load_collection(self.collection_name)
            logger.info(f"✅ 集合加载成功")
        except Exception as e:
            logger.error(f"❌ 加载集合失败: {e}")
            raise

    # ==================== 工厂方法 ====================

    @classmethod
    def from_settings(cls) -> "MilvusDBClient":
        """
        从项目配置创建 Milvus 客户端实例

        Returns:
            MilvusDBClient: 配置好的客户端实例
        """
        return cls()


# ==================== 单例模式支持 ====================

_milvus_client_instance: Optional[MilvusDBClient] = None


def get_milvus_client() -> MilvusDBClient:
    """
    获取 Milvus 客户端单例实例

    Returns:
        MilvusDBClient: 单例客户端实例
    """
    global _milvus_client_instance
    if _milvus_client_instance is None:
        _milvus_client_instance = MilvusDBClient.from_settings()
        _milvus_client_instance.connect()
    return _milvus_client_instance


def get_milvus_client_from_config(config: Dict) -> MilvusDBClient:
    """
    从自定义配置创建 Milvus 客户端实例

    Args:
        config: 配置字典

    Returns:
        MilvusDBClient: 配置好的客户端实例
    """
    client = MilvusDBClient(config)
    client.connect()
    return client


if __name__ == "__main__":
    # 示例1: 使用单例模式
    print("=" * 60)
    print("示例1: 使用单例模式")
    print("=" * 60)

    client = get_milvus_client()

    # 获取统计信息
    count = client.get_entity_count()
    print(f"📊 集合中现有数据量: {count}")

    # 示例2: 使用上下文管理器
    print("\n" + "=" * 60)
    print("示例2: 使用上下文管理器")
    print("=" * 60)

    config = {
        "db_path": str(settings.vectordb.db_path),
        "collection_name": settings.vectordb.collection_name,
        "vector_dimension": settings.vectordb.vector_dimension,
    }

    with MilvusDBClient(config) as client_ctx:
        stats = client_ctx.get_collection_stats()
        print(f"📊 集合统计信息: {stats}")

    print("\n✅ 示例运行完成")
