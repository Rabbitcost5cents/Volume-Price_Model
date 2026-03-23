"""
db.py — SQLite 数据库工具模块
集中管理所有数据库连接、建表、读写操作，供其他模块导入使用。
"""
import sqlite3
import os
import json
import logging
import pandas as pd
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 项目根目录（src/ 的上级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'sales.db')


# ---------------------------------------------------------------------------
# 连接与初始化
# ---------------------------------------------------------------------------

def get_connection():
    """返回 SQLite 连接对象（WAL 模式 + 30 秒写入超时，支持并发读写）。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _conn():
    """上下文管理器，确保连接在任何情况下都会被关闭。"""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """创建所有数据库表（幂等操作，表已存在则跳过）。"""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS product_specs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                model_raw       TEXT NOT NULL,
                model_key       TEXT NOT NULL UNIQUE,
                price_base      REAL,
                ram_gb          REAL,
                storage_gb      REAL,
                battery_mah     REAL,
                refresh_rate_hz REAL,
                main_camera_mp  REAL,
                charging_w      REAL,
                screen_res      REAL,
                ip_rating       REAL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS daily_sales (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key   TEXT NOT NULL,
                model_raw   TEXT,
                date        DATE NOT NULL,
                daily_sales INTEGER NOT NULL DEFAULT 0,
                UNIQUE(model_key, date)
            );

            CREATE TABLE IF NOT EXISTS lifecycle_sales (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                series        TEXT NOT NULL,
                month_index   INTEGER NOT NULL,
                monthly_sales REAL NOT NULL,
                UNIQUE(series, month_index)
            );

            CREATE TABLE IF NOT EXISTS price_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key      TEXT NOT NULL,
                effective_from DATE NOT NULL,
                price_delta    REAL NOT NULL,
                note           TEXT,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                split_date    DATE,
                train_size    INTEGER,
                test_size     INTEGER,
                wmape         REAL,
                mae           REAL,
                rmse          REAL,
                notes         TEXT
            );

            CREATE TABLE IF NOT EXISTS test_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                training_run_id INTEGER REFERENCES training_runs(id),
                date            DATE,
                model_key       TEXT,
                actual          REAL,
                predicted       REAL,
                error           REAL,
                abs_error       REAL
            );

            CREATE TABLE IF NOT EXISTS simulation_results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                product_name  TEXT,
                launch_date   DATE,
                duration_days INTEGER,
                agg_mode      TEXT,
                user_specs    TEXT,
                bass_params   TEXT,
                results       TEXT,
                source        TEXT
            );
        """)
        conn.commit()
    logger.info(f"Database initialized: {DB_PATH}")


def db_exists():
    """检查数据库是否存在且包含必要的基础数据。"""
    if not os.path.exists(DB_PATH):
        return False
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM product_specs")
            specs_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM daily_sales")
            sales_count = cur.fetchone()[0]
        return specs_count > 0 and sales_count > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 读取函数
# ---------------------------------------------------------------------------

def load_specs_df():
    """加载 product_specs 表，返回与 clean_specs() 兼容的 DataFrame。"""
    with _conn() as conn:
        df = pd.read_sql(
            "SELECT model_raw, model_key, price_base, ram_gb, storage_gb, "
            "battery_mah, refresh_rate_hz, main_camera_mp, charging_w, screen_res, ip_rating "
            "FROM product_specs",
            conn
        )
    return df


def load_daily_sales_df():
    """加载 daily_sales 表，返回与 process_daily_sales() 兼容的 DataFrame。"""
    with _conn() as conn:
        df = pd.read_sql(
            "SELECT model_key, model_raw, date, daily_sales FROM daily_sales",
            conn
        )
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_lifecycle_df():
    """加载 lifecycle_sales 表，返回 DataFrame（series, month_index, monthly_sales）。"""
    with _conn() as conn:
        df = pd.read_sql(
            "SELECT series, month_index, monthly_sales FROM lifecycle_sales",
            conn
        )
    return df


def load_price_history_df():
    """加载 price_history 表，返回 DataFrame。"""
    with _conn() as conn:
        df = pd.read_sql("SELECT * FROM price_history", conn)
    if not df.empty:
        df['effective_from'] = pd.to_datetime(df['effective_from'])
    return df


def get_latest_test_results():
    """返回最近一次训练的测试集预测结果 DataFrame。"""
    try:
        with _conn() as conn:
            latest = pd.read_sql(
                "SELECT id FROM training_runs ORDER BY run_timestamp DESC LIMIT 1",
                conn
            )
            if latest.empty:
                return pd.DataFrame()
            run_id = int(latest['id'].iloc[0])
            df = pd.read_sql(
                "SELECT * FROM test_results WHERE training_run_id = ?",
                conn, params=(run_id,)
            )
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame()


def get_simulation_history(limit=50):
    """返回最近的模拟预测记录列表（不含详细结果 JSON）。"""
    try:
        with _conn() as conn:
            df = pd.read_sql(
                "SELECT id, run_timestamp, product_name, launch_date, "
                "duration_days, agg_mode, source "
                "FROM simulation_results ORDER BY run_timestamp DESC LIMIT ?",
                conn, params=(limit,)
            )
        return df
    except Exception:
        return pd.DataFrame()


def get_training_history(limit=20):
    """返回最近的训练记录列表。"""
    try:
        with _conn() as conn:
            df = pd.read_sql(
                "SELECT id, run_timestamp, split_date, train_size, test_size, "
                "wmape, mae, rmse, notes "
                "FROM training_runs ORDER BY run_timestamp DESC LIMIT ?",
                conn, params=(limit,)
            )
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 管理员删除函数
# ---------------------------------------------------------------------------

def delete_training_run(run_id: int) -> bool:
    """删除指定训练记录及其关联的测试结果行，返回是否成功。"""
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM test_results WHERE training_run_id = ?", (run_id,))
            conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))
            conn.commit()
        return True
    except Exception:
        return False


def delete_simulation_result(sim_id: int) -> bool:
    """删除指定模拟记录，返回是否成功。"""
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM simulation_results WHERE id = ?", (sim_id,))
            conn.commit()
        return True
    except Exception:
        return False


def delete_all_training_runs() -> int:
    """清空所有训练记录及测试结果，返回删除的行数。"""
    try:
        with _conn() as conn:
            n = conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            conn.execute("DELETE FROM test_results")
            conn.execute("DELETE FROM training_runs")
            conn.commit()
        return n
    except Exception:
        return 0


def delete_all_simulation_results() -> int:
    """清空所有模拟记录，返回删除的行数。"""
    try:
        with _conn() as conn:
            n = conn.execute("SELECT COUNT(*) FROM simulation_results").fetchone()[0]
            conn.execute("DELETE FROM simulation_results")
            conn.commit()
        return n
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 写入函数
# ---------------------------------------------------------------------------

def save_training_run(metrics: dict, test_df: pd.DataFrame) -> int:
    """将训练指标和测试集预测结果写入数据库，返回 training_run_id。"""
    rows = [
        (
            None,  # run_id placeholder, filled after insert
            str(row['date'])[:10],
            row['model_key'],
            float(row['actual']),
            float(row['predicted']),
            float(row['error']),
            float(row['abs_error'])
        )
        for _, row in test_df.iterrows()
    ]
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO training_runs (split_date, train_size, test_size, wmape, mae, rmse, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(metrics.get('split_date', ''))[:10],
                int(metrics.get('train_size', 0)),
                int(metrics.get('test_size', 0)),
                float(metrics.get('wmape', 0)),
                float(metrics.get('mae', 0)),
                float(metrics.get('rmse', 0)),
                metrics.get('notes', '')
            )
        )
        run_id = cur.lastrowid
        rows_with_id = [(run_id,) + r[1:] for r in rows]
        cur.executemany(
            "INSERT INTO test_results "
            "(training_run_id, date, model_key, actual, predicted, error, abs_error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows_with_id
        )
        conn.commit()
    logger.info(f"Training run saved to database (run_id={run_id})")
    return run_id


def save_simulation_result(params: dict, results_df: pd.DataFrame, source: str = 'unknown'):
    """将模拟预测结果写入 simulation_results 表。"""
    export = results_df.copy()
    if 'date' in export.columns and pd.api.types.is_datetime64_any_dtype(export['date']):
        export['date'] = export['date'].dt.strftime('%Y-%m-%d')
    results_json = export.to_json(orient='records', force_ascii=False)

    with _conn() as conn:
        conn.execute(
            "INSERT INTO simulation_results "
            "(product_name, launch_date, duration_days, agg_mode, user_specs, bass_params, results, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                params.get('product_name', ''),
                str(params.get('launch_date', ''))[:10],
                int(params.get('duration_days', 0)),
                params.get('agg_mode', ''),
                json.dumps(params.get('user_specs', {}), ensure_ascii=False),
                json.dumps(params.get('bass_params', {}), ensure_ascii=False),
                results_json,
                source
            )
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 数据写入（upsert）函数
# ---------------------------------------------------------------------------

def upsert_specs(specs_df: pd.DataFrame) -> int:
    """
    将规格 DataFrame 写入 product_specs 表。
    model_key 已存在时覆盖（支持价格/规格修正），返回写入行数。
    """
    rows = [
        (
            str(row.get('model_raw', '')),
            str(row.get('model_key', '')),
            float(row.get('price_base', 0)),
            float(row.get('ram_gb', 0)),
            float(row.get('storage_gb', 0)),
            float(row.get('battery_mah', 0)),
            float(row.get('refresh_rate_hz', 0)),
            float(row.get('main_camera_mp', 0)),
            float(row.get('charging_w', 0)),
            float(row.get('screen_res', 0)),
            float(row.get('ip_rating', 0)),
        )
        for _, row in specs_df.iterrows()
        if row.get('model_key')
    ]
    with _conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO product_specs
               (model_raw, model_key, price_base, ram_gb, storage_gb, battery_mah,
                refresh_rate_hz, main_camera_mp, charging_w, screen_res, ip_rating)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        conn.commit()
    return len(rows)


def upsert_lifecycle_sales(lifecycle_df: pd.DataFrame) -> int:
    """
    将月度生命周期 DataFrame 写入 lifecycle_sales 表。
    (series, month_index) 已存在时覆盖，返回写入行数。
    """
    rows = [
        (str(row['series']), int(row['month_index']), float(row['monthly_sales']))
        for _, row in lifecycle_df.iterrows()
    ]
    with _conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO lifecycle_sales (series, month_index, monthly_sales)
               VALUES (?, ?, ?)""",
            rows
        )
        conn.commit()
    return len(rows)


def upsert_daily_sales(sales_df: pd.DataFrame) -> int:
    """
    将销量 DataFrame 写入 daily_sales 表。
    (model_key, date) 已存在时覆盖（支持数据修正），返回写入行数。
    """
    rows = [
        (
            str(row['model_key']),
            str(row.get('model_raw', row['model_key'])),
            str(row['date'])[:10],
            int(row.get('daily_sales', 0)),
        )
        for _, row in sales_df.iterrows()
        if pd.notna(row.get('date'))
    ]
    with _conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO daily_sales (model_key, model_raw, date, daily_sales)
               VALUES (?, ?, ?, ?)""",
            rows
        )
        conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# 快速验证入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    _ALLOWED_TABLES = frozenset({
        'product_specs', 'daily_sales', 'lifecycle_sales',
        'price_history', 'training_runs', 'test_results', 'simulation_results',
    })
    init_db()
    with _conn() as conn:
        for table in _ALLOWED_TABLES:
            count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", conn)['cnt'][0]
            print(f"  {table}: {count} rows")
