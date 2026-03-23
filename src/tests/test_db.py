"""
Unit tests for db.py
覆盖场景：C1 WAL 模式验证、表创建幂等性、upsert/load 往返、
         db_exists 空库返回 False、delete 函数、save_simulation_result。
所有测试使用 monkeypatch 将 DB_PATH 重定向到临时目录，不影响真实数据库。
"""
import os
import sys
import sqlite3
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """将 db.DB_PATH 重定向到临时文件并初始化数据库。"""
    import db
    test_db_path = str(tmp_path / "test_sales.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    return db


# ---------------------------------------------------------------------------
# C1: WAL 模式
# ---------------------------------------------------------------------------

class TestWALMode:
    def test_wal_mode_enabled(self, tmp_path, monkeypatch):
        """C1: 新建连接后 journal_mode 应为 wal。"""
        import db
        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "wal_test.db"))

        conn = db.get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()

        assert mode == "wal", f"Expected WAL mode, got: {mode}"

    def test_foreign_keys_enabled(self, tmp_path, monkeypatch):
        """新建连接后外键约束应已启用。"""
        import db
        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "fk_test.db"))

        conn = db.get_connection()
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        conn.close()

        assert fk == 1, "Foreign keys should be enabled"


# ---------------------------------------------------------------------------
# 表结构
# ---------------------------------------------------------------------------

class TestInitDb:
    EXPECTED_TABLES = {
        "product_specs", "daily_sales", "lifecycle_sales",
        "price_history", "training_runs", "test_results", "simulation_results",
    }

    def test_creates_all_tables(self, temp_db):
        """init_db 应创建全部 7 张表。"""
        conn = sqlite3.connect(temp_db.DB_PATH)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert self.EXPECTED_TABLES.issubset(tables)

    def test_idempotent(self, temp_db):
        """多次调用 init_db 不应报错（IF NOT EXISTS）。"""
        temp_db.init_db()
        temp_db.init_db()  # 第三次调用


# ---------------------------------------------------------------------------
# product_specs 读写
# ---------------------------------------------------------------------------

class TestSpecsRoundtrip:
    def _sample_specs(self):
        return pd.DataFrame([{
            "model_raw": "vivo V30 5G",
            "model_key": "v30",
            "price_base": 26000.0,
            "ram_gb": 8.0,
            "storage_gb": 256.0,
            "battery_mah": 5000.0,
            "refresh_rate_hz": 120.0,
            "main_camera_mp": 50.0,
            "charging_w": 44.0,
            "screen_res": 2400.0,
            "ip_rating": 68.0,
        }])

    def test_upsert_returns_count(self, temp_db):
        count = temp_db.upsert_specs(self._sample_specs())
        assert count == 1

    def test_load_returns_correct_data(self, temp_db):
        temp_db.upsert_specs(self._sample_specs())
        df = temp_db.load_specs_df()
        assert len(df) == 1
        assert df.iloc[0]["model_key"] == "v30"
        assert df.iloc[0]["price_base"] == 26000.0

    def test_upsert_overwrites_existing(self, temp_db):
        """model_key 已存在时 upsert 应覆盖旧值。"""
        temp_db.upsert_specs(self._sample_specs())
        updated = self._sample_specs().copy()
        updated["price_base"] = 24000.0
        temp_db.upsert_specs(updated)
        df = temp_db.load_specs_df()
        assert len(df) == 1
        assert df.iloc[0]["price_base"] == 24000.0


# ---------------------------------------------------------------------------
# daily_sales 读写
# ---------------------------------------------------------------------------

class TestDailySalesRoundtrip:
    def _sample_sales(self):
        return pd.DataFrame([
            {"model_key": "v30", "model_raw": "vivo V30", "date": pd.Timestamp("2024-01-01"), "daily_sales": 150},
            {"model_key": "v30", "model_raw": "vivo V30", "date": pd.Timestamp("2024-01-02"), "daily_sales": 120},
        ])

    def test_upsert_returns_count(self, temp_db):
        assert temp_db.upsert_daily_sales(self._sample_sales()) == 2

    def test_load_correct_data(self, temp_db):
        temp_db.upsert_daily_sales(self._sample_sales())
        df = temp_db.load_daily_sales_df()
        assert len(df) == 2
        assert set(df["model_key"]) == {"v30"}
        assert df["daily_sales"].sum() == 270

    def test_date_column_is_datetime(self, temp_db):
        temp_db.upsert_daily_sales(self._sample_sales())
        df = temp_db.load_daily_sales_df()
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ---------------------------------------------------------------------------
# db_exists
# ---------------------------------------------------------------------------

class TestDbExists:
    def test_returns_false_for_empty_db(self, temp_db):
        """空库（无数据）时 db_exists 应返回 False。"""
        assert temp_db.db_exists() is False

    def test_returns_true_after_data_inserted(self, temp_db):
        specs = pd.DataFrame([{
            "model_raw": "vivo V30", "model_key": "v30", "price_base": 26000.0,
            "ram_gb": 8.0, "storage_gb": 256.0, "battery_mah": 5000.0,
            "refresh_rate_hz": 120.0, "main_camera_mp": 50.0, "charging_w": 44.0,
            "screen_res": 2400.0, "ip_rating": 68.0,
        }])
        sales = pd.DataFrame([{
            "model_key": "v30", "model_raw": "vivo V30",
            "date": pd.Timestamp("2024-01-01"), "daily_sales": 100,
        }])
        temp_db.upsert_specs(specs)
        temp_db.upsert_daily_sales(sales)
        assert temp_db.db_exists() is True

    def test_returns_false_for_nonexistent_file(self, tmp_path, monkeypatch):
        import db
        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "nonexistent.db"))
        assert db.db_exists() is False


# ---------------------------------------------------------------------------
# training_runs / test_results
# ---------------------------------------------------------------------------

class TestSaveTrainingRun:
    def test_returns_run_id(self, temp_db):
        metrics = {
            "split_date": "2024-06-01",
            "train_size": 1000,
            "test_size": 200,
            "wmape": 0.15,
            "mae": 30.5,
            "rmse": 45.0,
            "notes": "test run",
        }
        test_df = pd.DataFrame([{
            "date": pd.Timestamp("2024-06-15"),
            "model_key": "v30",
            "actual": 100.0,
            "predicted": 95.0,
            "error": 5.0,
            "abs_error": 5.0,
        }])
        run_id = temp_db.save_training_run(metrics, test_df)
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_get_latest_test_results(self, temp_db):
        metrics = {
            "split_date": "2024-06-01", "train_size": 100, "test_size": 20,
            "wmape": 0.1, "mae": 20.0, "rmse": 30.0, "notes": "",
        }
        test_df = pd.DataFrame([{
            "date": pd.Timestamp("2024-06-15"),
            "model_key": "v30",
            "actual": 100.0, "predicted": 90.0, "error": 10.0, "abs_error": 10.0,
        }])
        temp_db.save_training_run(metrics, test_df)
        result = temp_db.get_latest_test_results()
        assert not result.empty
        assert result.iloc[0]["model_key"] == "v30"


# ---------------------------------------------------------------------------
# simulation_results
# ---------------------------------------------------------------------------

class TestSaveSimulationResult:
    def test_save_and_retrieve(self, temp_db):
        params = {
            "product_name": "vivo V70",
            "launch_date": "2025-01-01",
            "duration_days": 90,
            "agg_mode": "monthly",
            "user_specs": {"price": 28000},
            "bass_params": {"m": 100000, "p": 0.03},
        }
        results_df = pd.DataFrame([
            {"date": pd.Timestamp("2025-01-01"), "sales": 500.0},
            {"date": pd.Timestamp("2025-01-02"), "sales": 450.0},
        ])
        temp_db.save_simulation_result(params, results_df, source="test")

        history = temp_db.get_simulation_history(limit=10)
        assert len(history) == 1
        assert history.iloc[0]["product_name"] == "vivo V70"
        assert history.iloc[0]["source"] == "test"


# ---------------------------------------------------------------------------
# 删除函数
# ---------------------------------------------------------------------------

class TestDeleteFunctions:
    def test_delete_training_run(self, temp_db):
        metrics = {
            "split_date": "2024-01-01", "train_size": 50, "test_size": 10,
            "wmape": 0.1, "mae": 10.0, "rmse": 15.0, "notes": "",
        }
        run_id = temp_db.save_training_run(metrics, pd.DataFrame(
            columns=["date", "model_key", "actual", "predicted", "error", "abs_error"]
        ))
        assert temp_db.delete_training_run(run_id) is True
        history = temp_db.get_training_history()
        assert len(history) == 0

    def test_delete_all_training_runs(self, temp_db):
        metrics = {
            "split_date": "2024-01-01", "train_size": 50, "test_size": 10,
            "wmape": 0.1, "mae": 10.0, "rmse": 15.0, "notes": "",
        }
        empty_df = pd.DataFrame(
            columns=["date", "model_key", "actual", "predicted", "error", "abs_error"]
        )
        temp_db.save_training_run(metrics, empty_df)
        temp_db.save_training_run(metrics, empty_df)
        n = temp_db.delete_all_training_runs()
        assert n == 2
