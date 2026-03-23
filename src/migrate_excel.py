"""
migrate_excel.py — 一次性数据迁移脚本
将 Excel 数据源全部导入 SQLite 数据库，执行一次即可。
后续增量更新请使用 data_processor_v2.py 的 ingest_all() 函数。

用法（从项目根目录执行）：
    python src/migrate_excel.py
"""
import logging
import os
import sys

# 确保工作目录为项目根目录，使相对路径正确解析
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from db import init_db, _conn, DB_PATH, upsert_lifecycle_sales

logger = logging.getLogger(__name__)
from data_processor_v2 import ingest_specs, ingest_daily_sales
from train_lifecycle import extract_lifecycle_data


def migrate():
    logger.info("=" * 50)
    logger.info("数据迁移：Excel → SQLite")
    logger.info(f"目标数据库：{DB_PATH}")
    logger.info("=" * 50)

    raw_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw_data.xlsx')
    sales_path    = os.path.join(PROJECT_ROOT, 'data', 'V30, V40, V50, V60 LIFETIME SALES.xlsx')

    # 校验源文件是否存在
    for path in [raw_data_path, sales_path]:
        if not os.path.exists(path):
            logger.error(f"找不到数据文件：{path}")
            sys.exit(1)

    # 初始化数据库（带约束的正式 DDL）
    init_db()

    # --- 1. 产品规格 ---
    logger.info("[1/4] 迁移产品规格（raw_data.xlsx → product_specs）...")
    specs_count = ingest_specs(raw_data_path)

    # --- 2. 每日销量 ---
    logger.info("[2/4] 迁移每日销量（LIFETIME SALES.xlsx → daily_sales）...")
    sales_count = ingest_daily_sales(sales_path)

    # --- 3. 月度生命周期 ---
    logger.info("[3/4] 迁移月度生命周期数据（Sheet2 → lifecycle_sales）...")
    lifecycle_df = extract_lifecycle_data(raw_data_path)
    lifecycle_cols = ['series', 'month_index', 'monthly_sales']
    available = [c for c in lifecycle_cols if c in lifecycle_df.columns]
    lifecycle_count = upsert_lifecycle_sales(lifecycle_df[available])
    logger.info(f"  → 写入 {lifecycle_count} 条月度生命周期数据")

    # --- 4. 价格变更历史预置 ---
    # R4: 使用 _conn() context manager，确保连接在任何情况下都会被关闭
    logger.info("[4/4] 预置价格变更记录（price_history）...")
    price_records = [
        ('v50', '2025-06-14', -2000.0, 'V50 降价 2000 元（用户反馈）'),
    ]
    with _conn() as conn:
        conn.execute("DELETE FROM price_history")
        conn.executemany(
            "INSERT INTO price_history (model_key, effective_from, price_delta, note) "
            "VALUES (?, ?, ?, ?)",
            price_records
        )
        conn.commit()
    logger.info(f"  → 写入 {len(price_records)} 条价格变更记录")

    # --- 结果验证 ---
    logger.info("验证各表记录数：")
    with _conn() as conn:
        for table in ['product_specs', 'daily_sales', 'lifecycle_sales', 'price_history']:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  {table:20s}: {count} 条")

    logger.info(f"迁移完成！数据库文件：{DB_PATH}")
    logger.info("后续增量更新：python src/data_processor_v2.py [specs_file] [sales_file]")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    migrate()
