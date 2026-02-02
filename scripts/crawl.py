#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
泸州市网络问政平台定向页数采集脚本
支持：指定页数范围、断点续传、代理轮换、进度显示
"""

import asyncio
import argparse
import json
import logging
import time
import random
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.append(os.getcwd())
from app.core.config import settings
from app.utils.data import clean_text

import aiosqlite
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PWTimeout

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s',
    handlers=[
        logging.FileHandler(f'log/spider_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingConfig:
    """爬取配置"""
    start_page: int = 1
    end_page: int = 10
    max_retries: int = 3
    delay_min: float = 2.0
    delay_max: float = 5.0
    batch_size: int = 5
    proxy_timeout: int = 30
    page_timeout: int = 30000
    checkpoint_interval: int = 2  # 每N页保存检查点
    
    @property
    def total_pages(self) -> int:
        return self.end_page - self.start_page + 1

@dataclass
class QuestionItem:
    """
    问政数据结构（根据要求调整）
    - dept: 问政对象（部门）
    - question: 问政内容（HTML）
    - category: 本地后续标注，初始为NULL
    - 不包含status字段
    """
    id: str
    title: str
    dept: str
    question: str
    answer: str
    category: Optional[str] = None
    question_time: str = ""
    answer_time: str = ""
    url: str = ""
    crawl_time: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "dept": self.dept,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "question_time": self.question_time,
            "answer_time": self.answer_time,
            "url": self.url,
            "crawl_time": self.crawl_time
        }

class ProxyRotator:
    """代理池轮换管理器"""
    
    def __init__(self, proxies: List[str]):
        self.proxies = proxies
        self.current_index = 0
        self.failed_proxies = set()
        self.success_count = {}
        self.lock = asyncio.Lock()
        
        for p in self.proxies:
            self.success_count[p] = 0
        
        if not self.proxies:
            logger.warning("⚠️  代理池为空，将使用直连（风险：IP可能被封）")
        else:
            logger.info(f"✅ 代理池已加载 {len(proxies)} 个代理")
    
    def get_next_proxy(self) -> Optional[str]:
        """获取下一个可用代理（智能轮询）"""
        if not self.proxies:
            return None
            
        available = [p for p in self.proxies if p not in self.failed_proxies]
        if not available:
            logger.warning("⚠️  所有代理均失败，重置代理池")
            self.failed_proxies.clear()
            available = self.proxies
            
        # 优先使用成功率高的代理
        available.sort(key=lambda x: self.success_count[x], reverse=True)
        proxy = available[0]
        self.current_index = (self.current_index + 1) % len(available)
        return proxy
    
    def mark_success(self, proxy: Optional[str]):
        """标记代理成功"""
        if proxy and proxy in self.success_count:
            self.success_count[proxy] += 1
    
    def mark_failed(self, proxy: Optional[str]):
        """标记代理失败"""
        if proxy:
            self.failed_proxies.add(proxy)
            logger.warning(f"❌ 代理已标记失败: {proxy[:20]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取代理统计"""
        return {
            "total": len(self.proxies),
            "available": len(self.proxies) - len(self.failed_proxies),
            "failed": len(self.failed_proxies),
            "top_proxy": max(self.success_count.items(), key=lambda x: x[1]) if self.success_count else None
        }

class DatabaseManager:
    """异步SQLite数据库管理"""
    
    def __init__(self, db_path: str = "luzhou_wenzheng.db"):
        self.db_path = db_path
        
    async def init_db(self):
        """初始化数据库表"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS wenzheng (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    dept TEXT,
                    question TEXT,
                    answer TEXT,
                    category TEXT,
                    question_time TEXT,
                    answer_time TEXT,
                    url TEXT,
                    crawl_time TEXT
                )
            """)
            
            # 创建索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_dept ON wenzheng(dept)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_time ON wenzheng(question_time)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_crawl ON wenzheng(crawl_time)")
            await db.commit()
        logger.info(f"✅ 数据库初始化完成: {self.db_path}")
    
    async def insert_item(self, item: QuestionItem):
        """插入数据（存在则替换）"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO wenzheng 
                (id, title, dept, question, answer, category, question_time, answer_time, url, crawl_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.title, item.dept, item.question, item.answer,
                item.category, item.question_time, item.answer_time, item.url, item.crawl_time
            ))
            await db.commit()
    
    async def exists(self, item_id: str) -> bool:
        """检查ID是否已存在"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT 1 FROM wenzheng WHERE id = ?", (item_id,)) as cursor:
                return await cursor.fetchone() is not None
    
    async def get_stats(self) -> Dict[str, int]:
        """获取数据统计"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*), COUNT(category) FROM wenzheng")
            total, categorized = await cursor.fetchone()
            return {
                "total_records": total,
                "categorized": categorized,
                "uncategorized": total - categorized
            }
    
    async def get_by_page_range(self, start_time: str, end_time: str) -> List[Dict]:
        """获取指定时间范围内的数据（用于验证）"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM wenzheng WHERE crawl_time BETWEEN ? AND ?", 
                (start_time, end_time)
            )
            columns = [description[0] for description in cursor.description]
            rows = await cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]

class ProgressTracker:
    """进度追踪器"""
    
    def __init__(self, start_page: int, end_page: int):
        self.start_page = start_page
        self.end_page = end_page
        self.total_pages = end_page - start_page + 1
        
        self.current_page_num = 0      # 当前实际页码（如 113）
        self.completed_count = 0       # 已完成页数（1, 2, 3...）
        self.success_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        self.start_time = time.time()
        self.lock = asyncio.Lock()
    
    async def update(self, page_num: int, success: int = 0, failed: int = 0, skipped: int = 0):
        """更新进度（page_num 是实际页码）"""
        async with self.lock:
            self.current_page_num = page_num
            self.completed_count += 1    # 每调用一次，完成一页
            self.success_items += success
            self.failed_items += failed
            self.skipped_items += skipped
    
    def display(self):
        """显示进度条"""
        elapsed = time.time() - self.start_time
        
        if self.total_pages > 0:
            percent = (self.completed_count / self.total_pages) * 100
        else:
            percent = 0
        
        # 估算剩余时间（基于平均速度）
        if self.completed_count > 0:
            avg_time_per_page = elapsed / self.completed_count
            remaining_pages = self.total_pages - self.completed_count
            eta_seconds = avg_time_per_page * remaining_pages
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "计算中..."
        
        # 进度条
        bar_length = 40
        filled = int(bar_length * min(percent, 100) / 100)  # min 防止超 100%
        bar = "█" * filled + "░" * (bar_length - filled)
        
        sys.stdout.write(f"\r\033[K")
        sys.stdout.write(f"进度: |{bar}| {percent:.1f}% "
                        f"({self.completed_count}/{self.total_pages}页) "
                        f"当前页:{self.current_page_num} "
                        f"成功:{self.success_items} "
                        f"跳过:{self.skipped_items} "
                        f"失败:{self.failed_items} | "
                        f"用时:{timedelta(seconds=int(elapsed))} "
                        f"剩余:{eta}")
        sys.stdout.flush()
    
    def summary(self) -> Dict[str, Any]:
        """获取摘要"""
        elapsed = time.time() - self.start_time
        return {
            "任务范围": f"{self.start_page}-{self.end_page}",
            "总页数": self.total_pages,
            "已完成": self.completed_count,
            "当前页": self.current_page_num,
            "成功入库": self.success_items,
            "跳过(已存在)": self.skipped_items,
            "失败": self.failed_items,
            "总耗时": str(timedelta(seconds=int(elapsed))),
            "平均速度": f"{self.success_items/max(elapsed/60, 1):.1f}条/分钟"
        }

        
class LZEPSpider:
    """爬虫主类"""
    
    def __init__(self, config: ScrapingConfig, proxy_rotator: Optional[ProxyRotator] = None):
        self.config = config
        self.db = DatabaseManager(db_path=settings.RAW_DATA_DB_PATH)
        self.proxy_rotator = proxy_rotator
        self.progress = ProgressTracker(config.start_page, config.end_page)
        self.base_url = "https://wen.lzep.cn"
        self.list_pattern = "/node/reply/{}.html"
        
    async def init(self):
        await self.db.init_db()
        
    async def create_browser_context(self, playwright, proxy: Optional[str] = None):
        """创建浏览器上下文"""
        browser_options = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",  # 关键：WSL内存处理
                "--disable-gpu",            # 关键：禁用GPU加速
                "--disable-web-security",
                "--disable-features=TranslateUI",
                "--disable-extensions",
                "--disable-plugins",
                "--single-process",         #  WSL建议单进程模式
                "--no-zygote",             #  WSL建议
            ]}
        
        if proxy:
            browser_options["proxy"] = {"server": proxy}
            
        browser = await playwright.chromium.launch(**browser_options)
        
        context = await browser.new_context(
            user_agent=random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            ]),
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            ignore_https_errors=True,
        )
        
        # 反检测脚本
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """)
        
        return browser, context
    
    async def random_delay(self):
        """随机延迟"""
        delay = random.uniform(self.config.delay_min, self.config.delay_max)
        await asyncio.sleep(delay)
    
    async def fetch_list_page(self, page: Page, page_num: int) -> Tuple[List[Dict], bool]:
        """
        获取列表页数据
        返回: (items, has_more)
        """
        url = f"{self.base_url}{self.list_pattern.format(page_num)}"
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=self.config.page_timeout)
            await self.random_delay()
            
            # 提取列表
            items = await page.locator("#content-list li").all()
            pattern = r"/wen/(\d+)\.html"
            urls = []
            for item in items:
                link_locator = item.locator("h4 a")
                href = await link_locator.get_attribute("href")
                match = re.search(pattern, href)
                urls.append({
                    "id": match.group(1),
                    "url": f"https://wen.lzep.cn{href}"
                })
            
            return urls, len(urls) > 0
            
        except PWTimeout:
            logger.error(f"⏱️  第{page_num}页加载超时")
            return [], False
        except Exception as e:
            logger.error(f"❌ 获取列表页 {page_num} 失败: {e}")
            return [], False
    
    async def parse_detail(self, page: Page, item: Dict) -> Optional[QuestionItem]:
        """解析详情页"""
        try:
            await page.goto(item["url"], wait_until="networkidle", timeout=self.config.page_timeout)
            await self.random_delay()
            
            question_element = page.locator(".troub-wrap")
            answer_element = page.locator(".return-wrap")

            data = dict()
            data["id"] = item["id"]
            data["url"] = item["url"]
            data["title"] = await question_element.locator("h4").text_content()
            data["dept"] = await question_element.locator(".info >> li", has_text="问政对象").locator("span").text_content()
            data["question_time"] = await question_element.locator(".time").first.text_content()
            data["question"] = await question_element.locator(".content-text").first.text_content()
            
            data["answer_time"] = await answer_element.locator(".time").first.text_content()
            data["answer"] = await answer_element.locator(".content-text").first.text_content()
            data["category"] = None
            
            print(data)
            
            return QuestionItem(
                **data,
                crawl_time=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ 解析详情页失败 {item['url']}: {e}")
            return None
    
    async def crawl_single_page(self, playwright, page_num: int) -> Tuple[int, int, int]:
        """
        爬取单页
        返回: (成功数, 跳過数, 失败数)
        """
        proxy = self.proxy_rotator.get_next_proxy() if self.proxy_rotator else None
        browser = None
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for attempt in range(self.config.max_retries):
            try:
                browser, context = await self.create_browser_context(playwright, proxy)
                list_page = await context.new_page()
                detail_page = await context.new_page()
                
                items, has_more = await self.fetch_list_page(list_page, page_num)
                
                if not has_more:
                    return 0, 0, 0
                
                for item in items:
                    # 检查是否已存在
                    if await self.db.exists(item['id']):
                        skip_count += 1
                        continue
                    
                    detail = await self.parse_detail(detail_page, item)
                    if detail:
                        await self.db.insert_item(detail)
                        success_count += 1
                    else:
                        fail_count += 1
                    
                    # 每处理5条小休息
                    if (success_count + fail_count) % 5 == 0:
                        await asyncio.sleep(random.uniform(1, 2))
                
                # 标记代理成功
                if self.proxy_rotator:
                    self.proxy_rotator.mark_success(proxy)
                
                return success_count, skip_count, fail_count
                
            except Exception as e:
                logger.error(f"❌ 第{page_num}页第{attempt+1}次尝试失败: {str(e)[:100]}")
                if self.proxy_rotator and proxy:
                    self.proxy_rotator.mark_failed(proxy)
                    proxy = self.proxy_rotator.get_next_proxy()
                await asyncio.sleep(2 ** attempt)  # 指数退避
                
            finally:
                if browser:
                    await browser.close()
        
        return success_count, skip_count, fail_count
    
    async def run(self):
        """主运行函数"""
        await self.init()
        
        logger.info(f"🚀 开始爬取：第{self.config.start_page}页 到 第{self.config.end_page}页")
        logger.info(f"📊 预估总页数: {self.config.total_pages}，代理池状态: {self.proxy_rotator.get_stats() if self.proxy_rotator else '未启用'}")
        
        async with async_playwright() as playwright:
            for page_num in range(self.config.start_page, self.config.end_page + 1):
                try:
                    success, skipped, failed = await self.crawl_single_page(playwright, page_num)
                    await self.progress.update(page_num, success, failed, skipped)
                    self.progress.display()
                    
                    # 检查点保存（每N页）
                    if page_num % self.config.checkpoint_interval == 0:
                        stats = await self.db.get_stats()
                        logger.info(f"\n💾 检查点 - 已爬取{page_num}页，数据库总计: {stats['total_records']}条")
                    
                    # 批次间长延迟（每batch_size页）
                    if page_num < self.config.end_page and page_num % self.config.batch_size == 0:
                        rest = random.uniform(8, 15)
                        logger.info(f"\n😴 已完成{self.config.batch_size}页，休息{rest:.1f}秒...")
                        await asyncio.sleep(rest)
                        
                except KeyboardInterrupt:
                    logger.info(f"\n⛔ 用户中断，当前进度: 第{page_num}页")
                    break
                except Exception as e:
                    logger.error(f"\n💥 第{page_num}页处理异常: {e}")
                    continue
            
            # 最终统计
            self.progress.display()
            print()  # 换行
            summary = self.progress.summary()
            db_stats = await self.db.get_stats()
            
            logger.info("=" * 60)
            logger.info("📈 爬取完成统计:")
            for k, v in summary.items():
                logger.info(f"   {k}: {v}")
            logger.info(f"📦 数据库总计: {db_stats['total_records']}条 (已标注{db_stats['categorized']}条)")
            logger.info("=" * 60)

def load_proxies() -> List[str]:
    """加载代理"""
    # 1. 环境变量 PROXY_LIST (逗号分隔)
    env_proxies = os.environ.get("PROXY_LIST", "")
    if env_proxies:
        return [p.strip() for p in env_proxies.split(",") if p.strip()]
    
    # 2. 文件 proxies.txt
    try:
        with open("proxies.txt", "r", encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            if lines:
                return lines
    except FileNotFoundError:
        pass
    
    # 3. 本地代理（Clash等）
    # return ["http://127.0.0.1:7890"]
    
    return []

def parse_page_range(page_arg: str) -> Tuple[int, int]:
    """解析页数参数"""
    if '-' in page_arg:
        start, end = map(int, page_arg.split('-'))
        return min(start, end), max(start, end)
    else:
        page = int(page_arg)
        return page, page

def main():
    parser = argparse.ArgumentParser(
        description="泸州市网络问政平台定向页数采集工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --pages 1-10              # 爬取第1到10页
  %(prog)s --pages 5-20 --proxy      # 爬取第5到20页，使用代理
  %(prog)s --start 1 --end 50        # 爬取第1到50页
  %(prog)s --page 3                  # 仅爬取第3页
  %(prog)s --pages 1-100 --delay 3-8 # 自定义延迟3-8秒
        """
    )
    
    # 页数参数（互斥组）
    page_group = parser.add_mutually_exclusive_group(required=True)
    page_group.add_argument('--pages', '-p', type=str, help='页数范围，如 "1-10" 或 "5"')
    page_group.add_argument('--start', '-s', type=int, help='起始页')
    page_group.add_argument('--page', type=int, help='单页模式，指定某一页')
    
    parser.add_argument('--end', '-e', type=int, help='结束页（与--start配合使用）')
    parser.add_argument('--delay', '-d', type=str, default="2-5", help='延迟范围，如 "2-5"（秒）')
    parser.add_argument('--proxy', action='store_true', help='启用代理池（从proxies.txt或环境变量读取）')
    parser.add_argument('--db', default="luzhou_wenzheng.db", help='数据库文件路径')
    parser.add_argument('--batch', '-b', type=int, default=5, help='每批次页数（默认5）')
    parser.add_argument('--retry', '-r', type=int, default=3, help='失败重试次数（默认3）')
    
    args = parser.parse_args()
    
    # 解析页数
    if args.pages:
        start_page, end_page = parse_page_range(args.pages)
    elif args.page:
        start_page, end_page = args.page, args.page
    else:
        if args.end is None:
            parser.error("--start 需要配合 --end 使用")
        start_page, end_page = args.start, args.end
    
    # 解析延迟
    delay_parts = args.delay.split('-')
    delay_min = float(delay_parts[0])
    delay_max = float(delay_parts[1]) if len(delay_parts) > 1 else delay_min + 3
    
    # 加载代理
    proxies = load_proxies() if args.proxy else []
    proxy_rotator = ProxyRotator(proxies) if proxies else None
    
    # 创建配置
    config = ScrapingConfig(
        start_page=start_page,
        end_page=end_page,
        delay_min=delay_min,
        delay_max=delay_max,
        batch_size=args.batch,
        max_retries=args.retry
    )
    
    # 运行
    spider = LZEPSpider(config, proxy_rotator)
    try:
        asyncio.run(spider.run())
    except KeyboardInterrupt:
        print("\n\n⛔ 用户强制退出")
    except Exception as e:
        logger.exception(f"程序异常: {e}")

if __name__ == "__main__":
    main()