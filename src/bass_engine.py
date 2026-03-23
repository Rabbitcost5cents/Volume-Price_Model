import json
import logging
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_CACHE_PATH = 'models/bass_params.json'

def bass_f(t, p, q):
    """Bass 模型的密度函数。p 为 0 时返回 0 避免除零错误。"""
    if np.isscalar(p):
        if p <= 0:
            return np.zeros_like(np.asarray(t, dtype=float))
    else:
        p = np.where(p <= 0, 1e-10, p)
    numerator = ((p + q)**2 / p) * np.exp(-(p + q) * t)
    denominator = (1 + (q / p) * np.exp(-(p + q) * t))**2
    return numerator / denominator

def bass_S(t, p, q, m):
    """在时间 t 的瞬时销量。"""
    return m * bass_f(t, p, q)

def fit_bass_nls(t_data, s_data):
    """
    使用非线性最小二乘法拟合 Bass 模型参数。
    t_data: 时间步长 (1, 2, 3...)
    s_data: 这些步骤中的实际销量
    返回: (p, q, m)
    """
    # 初始猜测值: p=0.03, q=0.38 (标准市场营销值), m=sum(s_data)*1.2
    sum_sales = np.sum(s_data)
    p0 = [0.01, 0.4, sum_sales * 1.1]
    
    # 限制参数在现实范围内
    # p: [0, 1], q: [0, 1]
    # m: [0, sum_sales * 3.0] (防止不切实际的巨大市场潜力)
    bounds = ([0, 0, 0], [1, 1, sum_sales * 3.0])
    
    try:
        popt, _ = curve_fit(bass_S, t_data, s_data, p0=p0, bounds=bounds)
        return popt
    except Exception as e:
        logger.warning(f"NLS Fit failed: {e}")
        return None

class BassEngine:
    def __init__(self):
        self.params = {} # {系列名称: (p, q, m)}
        self.avg_p = 0.03
        self.avg_q = 0.4
        
    def get_bass_params(self, series_name):
        """返回给定系列的 (p, q, m)，如果未找到则返回平均值。"""
        # 简单归一化以匹配 self.params 中的键
        key = series_name.upper()
        if key in self.params:
            return self.params[key]
        
        # 回退方案：使用平均值和 m 的启发式方法（其他型号的平均值）
        avg_m = np.mean([v[2] for k, v in self.params.items()]) if self.params else 100000
        return self.avg_p, self.avg_q, avg_m

    def save_to_cache(self, cache_path=_CACHE_PATH):
        """将已拟合的参数序列化到 JSON 文件供下次启动直接加载。
        使用临时文件 + os.replace 原子写入，防止中断导致缓存损坏。"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            data = {
                'params': {k: list(v) for k, v in self.params.items()},
                'avg_p': float(self.avg_p),
                'avg_q': float(self.avg_q),
            }
            tmp_path = cache_path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(data, f)
            os.replace(tmp_path, cache_path)  # 原子替换
        except Exception as e:
            logger.error("Bass cache save failed", exc_info=True)

    def load_from_cache(self, cache_path=_CACHE_PATH):
        """从 JSON 缓存加载参数，成功返回 True，失败返回 False。
        对加载的参数执行值域校验，防止缓存文件被篡改后影响预测结果。"""
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            params = {k: tuple(v) for k, v in data['params'].items()}
            avg_p  = float(data['avg_p'])
            avg_q  = float(data['avg_q'])

            # 值域校验：p ∈ (0,1]，q ∈ (0,1]，m > 0
            if not (0 < avg_p <= 1.0 and 0 < avg_q <= 1.0):
                raise ValueError(f'avg params out of range: avg_p={avg_p}, avg_q={avg_q}')
            for series, (p, q, m) in params.items():
                if not (0 < p <= 1.0 and 0 < q <= 1.0 and m > 0):
                    raise ValueError(
                        f'Bass params out of range for {series}: p={p}, q={q}, m={m}')

            self.params = params
            self.avg_p  = avg_p
            self.avg_q  = avg_q
            return True
        except Exception as e:
            logger.warning(f"Bass cache load failed, will re-fit: {e}")
            return False

    def calculate_theoretical_sales(self, series_name, months_since_launch):
        """计算每日 Bass 理论销量。"""
        p, q, m = self.get_bass_params(series_name)
        # bass_S 返回月度速率，除以 30 得到日度速率
        val = bass_S(months_since_launch, p, q, m) / 30.0
        return max(0, val)

    def train_on_sheet2(self, file_path, cache_path=_CACHE_PATH):
        """从 Sheet2（或数据库）提取数据并拟合每个系列的 Bass 参数。
        若缓存文件比数据源新，则直接加载缓存跳过拟合。"""
        # --- 缓存命中检查 ---
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            data_mtime = 0
            try:
                from db import DB_PATH, db_exists
                if db_exists():
                    data_mtime = os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0
                elif os.path.exists(file_path):
                    data_mtime = os.path.getmtime(file_path)
            except ImportError:
                if os.path.exists(file_path):
                    data_mtime = os.path.getmtime(file_path)
            if cache_mtime > data_mtime and self.load_from_cache(cache_path):
                logger.info("Bass params loaded from cache.")
                return

        try:
            from config_loader import cfg
            bass_cfg = cfg.get('bass_fit', {})
            conservative_series = [s.upper() for s in bass_cfg.get('conservative_series', ['V60'])]
            m_upper_conservative = float(bass_cfg.get('m_upper_conservative', 1.5))
            m_upper_default = float(bass_cfg.get('m_upper_default', 3.0))

            # 优先从数据库加载生命周期数据
            _use_db = False
            try:
                from db import db_exists, load_lifecycle_df
                _use_db = db_exists()
            except ImportError:
                pass

            if _use_db:
                df = load_lifecycle_df()
            else:
                from train_lifecycle import extract_lifecycle_data
                df = extract_lifecycle_data(file_path)

            if df.empty:
                return

            results = []
            for series in df['series'].unique():
                subset = df[df['series'] == series]
                subset = subset[subset['monthly_sales'] > 0]

                if len(subset) < 3:
                    continue

                t = subset['month_index'].values
                s = subset['monthly_sales'].values
                sum_s = np.sum(s)

                if series.upper() in conservative_series:
                    m_upper = sum_s * m_upper_conservative
                else:
                    m_upper = sum_s * m_upper_default

                p0 = [0.01, 0.4, sum_s * 1.1]
                bounds = ([0, 0, 0], [1, 1, m_upper])

                try:
                    popt, _ = curve_fit(bass_S, t, s, p0=p0, bounds=bounds)
                    # R5: 校验拟合结果是否在合法值域内（bounds 只约束优化过程，
                    #     数值精度误差仍可能导致轻微越界）
                    p_fit, q_fit, m_fit = float(popt[0]), float(popt[1]), float(popt[2])
                    if not (0 < p_fit <= 1.0 and 0 < q_fit <= 1.0 and m_fit > 0):
                        logger.warning(
                            f"Bass fit for {series} produced out-of-range params "
                            f"(p={p_fit:.4f}, q={q_fit:.4f}, m={m_fit:.1f}); skipping."
                        )
                    else:
                        key = series.upper()
                        self.params[key] = popt
                        results.append({'series': key, 'p': p_fit, 'q': q_fit, 'm': m_fit})
                except Exception as e:
                    logger.warning(f"Bass fit failed for {series}: {e}")

            res_df = pd.DataFrame(results)
            if not res_df.empty:
                self.avg_p = res_df['p'].mean()
                self.avg_q = res_df['q'].mean()
                logger.info("Bass Model Parameters (Monthly):\n" + res_df.to_string())
                self.save_to_cache(cache_path)

        except Exception as e:
            logger.error("Error training Bass Engine", exc_info=True)

    def predict_daily(self, start_month, end_month, m, p=None, q=None):
        """
        根据月度 Bass 参数生成每日预测曲线。
        m: 总市场潜力（生命周期的预测总销量）
        """
        p = p if p is not None else self.avg_p
        q = q if q is not None else self.avg_q
        
        # 我们逐日模拟
        # 由于 p 和 q 是根据月度数据拟合的，我们在公式中将 t 转换为月度单位
        days = np.arange(start_month * 30, end_month * 30)
        t_months = days / 30.0
        
        # 瞬时每日销量
        # 注意：bass_S(t) 是每月的密度。对于每天，我们除以 30
        daily_sales = bass_S(t_months, p, q, m) / 30.0
        
        return days, daily_sales

if __name__ == "__main__":
    # 测试
    engine = BassEngine()
    engine.train_on_sheet2('data/raw_data.xlsx')
    
    # m=100,000 的新发布示例预测
    t, s = engine.predict_daily(0, 10, 100000)
    plt.plot(t, s)
    plt.title("Simulated Daily Launch Curve (Bass Model)")
    plt.show()