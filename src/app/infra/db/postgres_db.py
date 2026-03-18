"""
PostgreSQL 向量数据库客户端实现
提供与 BaseDBClient 接口的完整实现，封装 PostgreSQL + pgvector 操作
"""

from typing import Optional, Dict, List, Any
import threading
import json
from datetime import datetime
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import numpy as np

from src.app.infra.db.base_db import BaseDBClient
from src.app.infra.utils.logger import get_logger
from src.config.setting import settings

logger = get_logger(__name__)


class PostgresDBClient(BaseDBClient):
    """
    PostgreSQL 数据库客户端（单例模式）
    实现 BaseDBClient 接口，提供 PostgreSQL + pgvector 数据库的连接和操作封装
    """

    _instance: Optional["PostgresDBClient"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[Dict] = None):
        """单例模式实现 - 确保只创建一个实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PostgresDBClient, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 PostgreSQL 客户端（单例）

        Args:
            config: 可选的配置字典，如果为 None 则使用 settings.postgres_db 配置
        """
        # 防止重复初始化
        if self._initialized:
            return

        # 使用传入的配置或默认配置
        postgres_config = settings.postgres_db
        self.config = config or {
            "host": postgres_config.host,
            "port": postgres_config.port,
            "user": postgres_config.user,
            "password": postgres_config.password,
            "database": postgres_config.database,
            "vector_dimension": postgres_config.vector_dimension,
            "metric_type": postgres_config.metric_type,
        }

        super().__init__(self.config)
        self.host: str = self.config.get("host", "localhost")
        self.port: int = self.config.get("port", 5432)
        self.user: str = self.config.get("user", "root")
        self.password: str = self.config.get("password", "root")
        self.database: str = self.config.get("database", "db")
        self.vector_dimension: int = self.config.get("vector_dimension", 1024)
        self.metric_type: str = self.config.get("metric_type", "cosine")

        # PostgreSQL 连接实例
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._cursor: Optional[psycopg2.extensions.cursor] = None

        # 标记为已初始化
        self._initialized = True

        # 自动连接数据库
        self.connect()

    def connect(self) -> None:
        """
        建立数据库连接
        初始化 PostgreSQL 客户端
        """
        try:
            logger.info(f"连接 PostgreSQL: {self.host}:{self.port}/{self.database}")

            # 创建连接
            self._conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self._cursor = self._conn.cursor()

            # 确保 pgvector 扩展已安装
            self._cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()

            logger.info(f"PostgreSQL 连接成功")

        except Exception as e:
            logger.error(f"PostgreSQL 连接失败: {e}")
            raise

    def get_client(self) -> psycopg2.extensions.connection:
        """
        获取初始化完成的客户端实例

        Returns:
            psycopg2.extensions.connection: PostgreSQL 连接实例

        Raises:
            RuntimeError: 如果客户端未初始化
        """
        if self._conn is None:
            raise RuntimeError("PostgreSQL 客户端未初始化，请先调用 connect()")
        return self._conn

    def close(self) -> None:
        """
        关闭连接/释放资源
        """
        try:
            if self._cursor is not None:
                self._cursor.close()
                self._cursor = None
            if self._conn is not None:
                logger.info("关闭 PostgreSQL 连接")
                self._conn.close()
                self._conn = None
                logger.info("PostgreSQL 连接已关闭")
        except Exception as e:
            logger.error(f"关闭 PostgreSQL 连接时出错: {e}")

    # ==================== 辅助方法 ====================

    def _build_columns_from_data(self, data: List[Dict]) -> List[str]:
        """
        从数据中动态获取字段列表

        Args:
            data: 数据列表

        Returns:
            字段名列表
        """
        if not data:
            return ["vector", "text", "metadata"]

        first_item = data[0]
        # 排除 vector（单独处理），其他字段直接使用
        return ["vector"] + [k for k in first_item.keys() if k != "vector"]

    def _ensure_table_exists(self, collection_name: str) -> None:
        """
        确保表存在，不存在则创建

        Args:
            collection_name: 表名称
            columns: 字段列表
        """
        try:
            # 检查表是否存在
            self._cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                (collection_name,)
            )
            table_exists = self._cursor.fetchone()[0]

            if not table_exists:
                logger.info(f"表 {collection_name} 不存在，正在创建...")

                # 构建字段定义
                field_defs = ["id SERIAL PRIMARY KEY", "vector vector(1024)"]

                # 根据表名确定额外字段（与 ingest.py 数据结构一致）
                if collection_name == "gov_cases":
                    # ingest.py 中的 gov_cases 字段结构
                    extra_fields = [
                        "text TEXT",
                        "department VARCHAR(255)",
                        "title VARCHAR(500)",
                        "question TEXT",
                        "answer TEXT",
                        "doc_type VARCHAR(50)",
                        "metadata JSONB",
                        "created_at TIMESTAMP DEFAULT NOW()"
                    ]
                elif collection_name == "gov_powers":
                    # ingest.py 中的 gov_powers 字段结构
                    extra_fields = [
                        "text TEXT",
                        "department VARCHAR(255)",
                        "power_type VARCHAR(50)",
                        "power_name VARCHAR(255)",
                        "doc_type VARCHAR(50)",
                        "created_at TIMESTAMP DEFAULT NOW()"
                    ]
                else:
                    # 默认表结构
                    extra_fields = ["text TEXT", "metadata JSONB", "created_at TIMESTAMP DEFAULT NOW()"]

                field_defs.extend(extra_fields)

                # 创建表
                create_sql = f'CREATE TABLE {collection_name} ({", ".join(field_defs)})'
                self._cursor.execute(create_sql)

                # 创建向量索引
                index_sql = f"CREATE INDEX IF NOT EXISTS idx_{collection_name}_vector ON {collection_name} USING ivfflat (vector {self.metric_type})"
                self._cursor.execute(index_sql)

                self._conn.commit()
                logger.info(f"表 {collection_name} 创建成功")
            else:
                logger.info(f"表 {collection_name} 已存在")

        except Exception as e:
            logger.error(f"表管理失败: {e}")
            self._conn.rollback()
            raise

    def _convert_vector_to_array(self, vector) -> str:
        """
        将向量转换为 PostgreSQL 数组格式

        Args:
            vector: 向量（list 或 numpy array）

        Returns:
            PostgreSQL 数组格式字符串
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        return "[" + ",".join(str(x) for x in vector) + "]"

    def _parse_filter_expr(self, filter_expr: str) -> str:
        """
        将 Milvus 风格的 filter_expr 转换为 SQL WHERE 子句

        Args:
            filter_expr: Milvus 风格的过滤表达式

        Returns:
            SQL WHERE 子句
        """
        if not filter_expr:
            return "1=1"

        # 简单实现：处理常见的比较操作
        # 例如: 'doc_id == "xxx"' -> 'doc_id = "xxx"'
        sql_expr = filter_expr.replace("==", "=").replace("!=", "<>")

        return sql_expr

    # ==================== 集合/表管理方法 ====================

    def has_collection(self, collection_name: str) -> bool:
        """
        检查表是否存在

        Returns:
            bool: 表是否存在
        """
        self._cursor.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (collection_name,)
        )
        return self._cursor.fetchone()[0]

    def drop_collection(self, collection_name: str) -> None:
        """
        删除表
        """
        if self.has_collection(collection_name):
            logger.warning(f"正在删除表 {collection_name}")
            self._cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(
                sql.Identifier(collection_name)
            ))
            self._conn.commit()
            logger.info(f"表 {collection_name} 已删除")

    def describe_collection(self, collection_name) -> Dict:
        """
        获取表详细信息

        Returns:
            Dict: 表的详细信息
        """
        self._cursor.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s",
            (collection_name,)
        )
        columns = self._cursor.fetchall()
        return {
            "collection_name": collection_name,
            "columns": [{"name": col[0], "type": col[1]} for col in columns]
        }

    # ==================== 数据操作方法 ====================

    def insert(
        self,
        collection_name: str,
        data: List[Dict],
        batch_size: Optional[int] = None
    ) -> Dict:
        """
        插入数据到表

        Args:
            collection_name: 表名称
            data: 要插入的数据列表，每个元素是包含向量和元数据的字典
            batch_size: 批量插入大小（可选）

        Returns:
            Dict: 插入结果，包含插入的 IDs
        """
        try:
            logger.info(f"向表 {collection_name} 插入 {len(data)} 条数据")

            # 确保表存在
            columns = self._build_columns_from_data(data)
            self._ensure_table_exists(collection_name, columns)

            # 准备插入数据
            insert_count = 0
            doc_ids = []

            for i in range(0, len(data), batch_size or len(data)):
                batch = data[i:i + (batch_size or len(data))]
                values_list = []

                for item in batch:
                    # 构建值列表
                    values = []
                    for col in columns:
                        if col == "vector":
                            values.append(self._convert_vector_to_array(item.get("vector")))
                        elif col == "metadata":
                            values.append(json.dumps(item.get("metadata", {})))
                        elif col == "created_at":
                            values.append(item.get("created_at") or datetime.now())
                        else:
                            values.append(item.get(col))

                    values_list.append(values)

                # 执行批量插入
                placeholders = ",".join(["%s"] * len(columns))
                insert_sql = f"INSERT INTO {collection_name} ({','.join(columns)}) VALUES ({placeholders})"

                self._cursor.executemany(insert_sql, values_list)
                insert_count += len(batch)
                doc_ids.extend([item.get("doc_id", f"auto_{i+j}") for j, item in enumerate(batch)])

            self._conn.commit()

            logger.info(f"插入成功，插入 {insert_count} 条数据")
            return {"insert_count": insert_count, "doc_ids": doc_ids}

        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            self._conn.rollback()
            raise

    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        **kwargs
    ) -> List[List[Dict]]:
        """
        向量相似度搜索

        Args:
            collection_name: 表名称
            vectors: 查询向量列表
            top_k: 返回结果数量
            output_fields: 要返回的字段列表
            filter_expr: 过滤表达式（可选）
            **kwargs: 其他搜索参数

        Returns:
            List[List[Dict]]: 搜索结果列表

        """
        try:
            results = []

            for vector in vectors:
                # 构建查询 SQL
                vector_str = self._convert_vector_to_array(vector)

                # 选择输出字段
                select_fields = output_fields or ["*"]

                # 构建 WHERE 子句
                where_clause = ""
                if filter_expr:
                    where_clause = f"WHERE {self._parse_filter_expr(filter_expr)}"

                # 使用余弦相似度搜索
                select_sql = f"""
                    SELECT {','.join(select_fields)}, (vector <=> '{vector_str}') as distance
                    FROM {collection_name}
                    {where_clause}
                    ORDER BY vector <=> '{vector_str}'
                    LIMIT {top_k}
                """

                self._cursor.execute(select_sql)
                rows = self._cursor.fetchall()

                # 转换为字典列表
                result_list = []
                for row in rows:
                    row_dict = {}
                    for idx, field in enumerate(select_fields):
                        row_dict[field] = row[idx]
                    result_list.append(row_dict)

                results.append(result_list)

            return results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise

    def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        条件查询（非向量搜索）

        Args:
            collection_name: 表名称
            filter_expr: 过滤表达式
            output_fields: 要返回的字段列表
            limit: 返回结果数量限制

        Returns:
            List[Dict]: 查询结果列表
        """
        try:
            select_fields = output_fields or ["*"]
            where_clause = self._parse_filter_expr(filter_expr)

            select_sql = f"""
                SELECT {','.join(select_fields)}
                FROM {collection_name}
                WHERE {where_clause}
                LIMIT {limit}
            """

            self._cursor.execute(select_sql)
            rows = self._cursor.fetchall()

            # 转换为字典列表
            results = []
            for row in rows:
                row_dict = {}
                for idx, field in enumerate(select_fields):
                    row_dict[field] = row[idx]
                results.append(row_dict)

            return results

        except Exception as e:
            logger.error(f"条件查询失败: {e}")
            raise

    def delete(
        self,
        collection_name: str,
        filter_expr: str
    ) -> Dict:
        """
        根据过滤条件删除数据

        Args:
            collection_name: 表名称
            filter_expr: 过滤表达式

        Returns:
            Dict: 删除结果
        """
        try:
            logger.info(f"删除满足条件的数据: {filter_expr}")

            where_clause = self._parse_filter_expr(filter_expr)
            delete_sql = f"DELETE FROM {collection_name} WHERE {where_clause}"

            self._cursor.execute(delete_sql)
            delete_count = self._cursor.rowcount
            self._conn.commit()

            logger.info(f"删除完成: 删除 {delete_count} 条数据")
            return {"delete_count": delete_count}

        except Exception as e:
            logger.error(f"删除数据失败: {e}")
            self._conn.rollback()
            raise

    def upsert(
        self,
        collection_name: str,
        data: List[Dict],
        batch_size: Optional[int] = None
    ) -> Dict:
        """
        更新或插入数据

        Args:
            collection_name: 表名称
            data: 要更新或插入的数据列表
            batch_size: 批量处理大小（可选）

        Returns:
            Dict: 操作结果
        """
        # 根据需求说明：不需要去重逻辑，doc_id 由外部业务层处理
        # 直接调用 insert 即可
        return self.insert(collection_name, data, batch_size)

    # ==================== 统计信息方法 ====================

    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        获取表统计信息

        Args:
            collection_name: 表名称

        Returns:
            Dict: 表的统计信息
        """
        try:
            self._cursor.execute(f"SELECT COUNT(*) FROM {collection_name}")
            row_count = self._cursor.fetchone()[0]

            return {
                "collection_name": collection_name,
                "row_count": row_count
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            raise

    def get_entity_count(self, collection_name: str) -> int:
        """
        获取表中的实体数量

        Returns:
            int: 实体数量
        """
        stats = self.get_collection_stats(collection_name)
        return stats.get("row_count", 0)

    # ==================== 工具方法 ====================

    def load_collection(self, collection_name) -> None:
        """
        加载表（PostgreSQL 无需显式加载，直接返回成功）

        Args:
            collection_name: 表名称
        """
        try:
            if self.has_collection(collection_name):
                logger.info(f"表 {collection_name} 已存在")
            else:
                logger.warning(f"表 {collection_name} 不存在")
        except Exception as e:
            logger.error(f"检查表失败: {e}")
            raise

    # ==================== 工厂方法 ====================

    @classmethod
    def from_settings(cls) -> "PostgresDBClient":
        """
        从项目配置创建/获取 PostgreSQL 客户端单例实例

        Returns:
            PostgresDBClient: 单例客户端实例
        """
        return cls()


if __name__ == "__main__":
    # 示例1: 使用单例模式
    logger.info("=" * 60)
    logger.info("示例1: 使用单例模式")
    logger.info("=" * 60)

    client = PostgresDBClient()

    # 获取统计信息
    if client.has_collection("gov_cases"):
        count = client.get_entity_count("gov_cases")
        logger.info(f"📊 表中现有数据量: {count}")
    else:
        logger.info("表 gov_cases 不存在")

    logger.info("\n✅ 示例运行完成")