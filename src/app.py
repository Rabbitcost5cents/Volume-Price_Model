import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from config_loader import cfg
from bass_engine import bass_S

logger = logging.getLogger(__name__)

class SalesSimulator:
    def __init__(self, resources):
        """
        初始化模拟器，加载所需资源。
        resources: 字典，包含以下键：
            - model: 主 XGBoost 模型
            - cold_model: 冷启动 XGBoost 模型
            - feature_cols: 主模型特征列表
            - cold_cols: 冷启动模型特征列表
            - bass_engine: 已训练的 BassEngine 实例
            - median_specs: 硬件规格中位数字典（用于回退）
        """
        self.res = resources

    def get_calendar_features(self, date):
        """根据给定日期生成日历特征。"""
        is_weekend = 1 if date.weekday() >= 5 else 0
        dow = date.weekday()
        d = date.day
        is_payday = 1 if (d == 15 or d == 30 or d == 31 or (date.month == 2 and d == 29)) else 0
        is_double = 1 if date.month == date.day else 0

        is_holiday = 1 if date.strftime('%Y-%m-%d') in cfg.get('holidays', []) else 0

        return {'dow': dow, 'is_weekend': is_weekend, 'is_payday': is_payday,
                'is_double_digit': is_double, 'is_holiday': is_holiday}

    def _get_cold_pred(self, r, base_val, days_sl, is_new_launch_mode):
        """冷启动模型预测（前 7 天 + 无历史销量场景）。"""
        cold_model = self.res['cold_model']
        cold_cols  = self.res['cold_cols']

        input_vec = [float(r.get(col, 0)) for col in cold_cols]
        raw_lift  = cold_model.predict(pd.DataFrame([input_vec], columns=cold_cols))[0]
        pred_lift = max(cfg['logic_thresholds']['min_lift'], raw_lift)
        res = base_val * pred_lift

        if is_new_launch_mode:
            # 缓冲系数与手动弹性调整
            c_start = cfg['logic_thresholds']['launch_cushion_start']
            c_end   = cfg['logic_thresholds']['launch_cushion_end']
            c_days  = cfg['logic_thresholds']['launch_cushion_days']

            if 0 <= days_sl < c_days:
                res *= (c_end + (c_start - c_end) * (c_days - days_sl) / c_days)
            elif days_sl >= c_days:
                res *= 1.05

            u_price = r.get('current_price', cfg['baselines']['price'])
            if u_price > 0:
                res *= (u_price / float(cfg['baselines']['price'])) ** cfg['elasticity']['cold_price']

            # 硬件弹性
            res *= (
                (r.get('battery_mah', cfg['baselines']['battery']) / float(cfg['baselines']['battery'])) ** cfg['elasticity']['battery'] *
                (r.get('ip_rating',   cfg['baselines']['ip_rating']) / float(cfg['baselines']['ip_rating'])) ** cfg['elasticity']['ip_rating'] *
                (r.get('ram_gb',      cfg['baselines']['ram'])       / float(cfg['baselines']['ram']))       ** cfg['elasticity']['ram'] *
                (r.get('storage_gb',  cfg['baselines']['storage'])   / float(cfg['baselines']['storage']))   ** cfg['elasticity']['storage']
            )

            return max(res, base_val * cfg['logic_thresholds']['bass_floor'])
        return res

    def _get_main_pred(self, r_lock, r_real, base_val, recent):
        """主模型预测（day 14+ 稳定期），含因果去偏与弹性调整。"""
        main_model   = self.res['model']
        feature_cols = self.res['feature_cols']

        r_lock['lag_1d']          = recent[-1]
        r_lock['rolling_7d_mean'] = sum(recent) / len(recent)
        input_vec = [float(r_lock.get(col, 0)) for col in feature_cols]

        pred_lift = max(cfg['logic_thresholds']['min_lift'],
                        main_model.predict(pd.DataFrame([input_vec], columns=feature_cols))[0])
        res = base_val * pred_lift

        # 价格弹性（主模型）
        u_price = r_real.get('current_price', cfg['baselines']['price_main'])
        b_price = r_lock.get('current_price', cfg['baselines']['price_main'])
        if b_price > 0 and u_price > 0:
            res *= (u_price / float(b_price)) ** cfg['elasticity']['main_price']

        # 硬件弹性（主模型）
        res *= (
            (r_real.get('battery_mah', cfg['baselines']['battery_lock']) / float(cfg['baselines']['battery_lock'])) ** cfg['elasticity']['battery'] *
            (r_real.get('ip_rating',   cfg['baselines']['ip_lock'])       / float(cfg['baselines']['ip_lock']))       ** cfg['elasticity']['ip_rating'] *
            (r_real.get('ram_gb',      cfg['baselines']['ram'])           / float(cfg['baselines']['ram']))           ** cfg['elasticity']['ram'] *
            (r_real.get('storage_gb',  cfg['baselines']['storage'])       / float(cfg['baselines']['storage']))       ** cfg['elasticity']['storage']
        )
        return res

    def run_simulation(self, context, user_specs, bass_params, duration_days):
        """
        执行销量模拟循环。

        参数:
            context: 字典，包含 'launch_date'、'last_date'、'last_7d_sales'、'original_specs'、'is_new_launch'
            user_specs: 当前用户输入字典（价格、规格）
            bass_params: 字典，包含 'm' 和 'p'
            duration_days: 整数，模拟天数

        返回:
            sim_results: 字典列表 [{'date': ..., 'sales': ...}]
        """
        
        # 解包模拟循环所需资源
        bass_engine = self.res['bass_engine']
        
        # 解包上下文
        launch_date = context['launch_date']
        curr_date = context['last_date'] + timedelta(days=1)
        sales_history = context.get('last_7d_sales', [0.0]*7).copy()
        original_specs = context.get('original_specs', self.res['median_specs']).copy()
        is_new_launch_mode = context.get('is_new_launch', False)
        
        # 解包输入参数
        m_potential = bass_params.get('m', 100000)
        p_user = bass_params.get('p', 0.05)
        
        # 确定 Bass 模型使用的机型键（用于判断是否为现有产品）
        # 注：提取机型键（V30/V40）的逻辑需要在此处理。
        # 可将 'model_key' 传入 context，或由 is_new_launch_mode 标志决定。
        # 原始代码中通过 `self.sim_model_var.get()` 获取。
        # 已将 model_key 添加至 context 或输入参数中传递。
        model_key = context.get('model_key', "")

        sim_results = []

        for _ in range(duration_days):
            days_sl = (curr_date - launch_date).days
            months_sl = days_sl / 30.0
            cal = self.get_calendar_features(curr_date)
            
            # 1. Calculate Bass Theoretical Baseline
            if is_new_launch_mode or "V60" in model_key:
                p = p_user 
                q = bass_engine.avg_q * cfg['bass_boosts']['q_mult']
                base_val = bass_S(months_sl, p, q, m_potential) / 30.0
                pulse_duration = cfg['bass_boosts']['pulse_duration']
                if days_sl < pulse_duration:
                    decay = (pulse_duration - days_sl) / pulse_duration
                    base_val *= (1.0 + (cfg['bass_boosts']['launch_pulse'] - 1.0) * decay)
            else:
                # 对现有旧系列使用已拟合参数
                # 从机型键中提取系列名称（如 v4012256 -> V40）
                # 假设机型键格式为 'v40...' 或 'V40...'
                if len(model_key) >= 3:
                    s_key = "V" + model_key[1:3]
                else:
                    s_key = "V40" # 回退默认值
                base_val = bass_engine.calculate_theoretical_sales(s_key, months_sl)

            # 2. Prediction Logic
            row = user_specs.copy()
            row.update(cal)
            row.update({
                'months_since_launch': months_sl,
                'days_since_launch': days_sl,
                'is_launch_day': 1 if days_sl == 0 else 0,
                'is_launch_week': 1 if days_sl < 7 else 0
            })
            
            # 获取近期销量历史
            recent = sales_history[-7:] if len(sales_history) >= 7 else ([0]*(7-len(sales_history)) + sales_history)
            
            cutoff_days = cfg['logic_thresholds']['cutoff_days']
            transition_window = cfg['logic_thresholds']['transition_window']
            use_cold_model = (days_sl < cutoff_days) or (sum(recent) == 0)
            in_transition = not use_cold_model and (days_sl < cutoff_days + transition_window)
            
            # 主模型锁定输入（因果去偏）
            locked_row = original_specs.copy()
            locked_row.update(cal)
            # 仅更新时间特征，保持基准硬件规格不变！
            locked_row.update({
                'months_since_launch': months_sl,
                'days_since_launch': days_sl,
                'is_launch_day': 1 if days_sl == 0 else 0,
                'is_launch_week': 1 if days_sl < 7 else 0
            })
            locked_row.update({
                'battery_mah': cfg['baselines']['battery_lock'], 
                'ip_rating': cfg['baselines']['ip_lock'], 
                'ram_gb': cfg['baselines']['ram'], 
                'storage_gb': cfg['baselines']['storage']
            })

            # 选择预测模型（调用类方法，不在循环内定义函数）
            if use_cold_model:
                pred_sales = self._get_cold_pred(row, base_val, days_sl, is_new_launch_mode)
            elif in_transition:
                p_cold = self._get_cold_pred(row, base_val, days_sl, is_new_launch_mode)
                p_main = self._get_main_pred(locked_row, row, base_val, recent)
                steps_past_cutoff = days_sl - cutoff_days
                alpha = max(0.0, min(1.0, steps_past_cutoff / transition_window))
                pred_sales = (1 - alpha) * p_cold + alpha * p_main
            else:
                pred_sales = self._get_main_pred(locked_row, row, base_val, recent)

            # 3. 后处理（季节性调整）
            boost = 1.0
            if cal['is_weekend']: boost *= cfg['seasonality']['weekend']
            if cal['is_payday']: boost *= cfg['seasonality']['payday']
            if cal['is_holiday']: boost *= cfg['seasonality']['holiday']
            pred_sales *= boost

            # 4. 防止销量归零（保底下限）
            if months_sl < 3.0:
                recent_vals = [s for s in recent if s > 0]
                if recent_vals:
                    recent_avg = sum(recent_vals) / len(recent_vals)
                    min_floor = max(1.0, recent_avg * 0.1)
                    if pred_sales < min_floor:
                        pred_sales = min_floor

            sim_results.append({'date': curr_date, 'sales': pred_sales})
            sales_history.append(pred_sales)
            curr_date += timedelta(days=1)
            
        return sim_results

    def aggregate_results(self, sim_results, agg_mode, launch_date):
        """根据所选模式聚合每日模拟结果。"""
        df_sim = pd.DataFrame(sim_results)
        
        if agg_mode == 'monthly':
            df_sim.set_index('date', inplace=True)
            df_agg = df_sim.resample('ME')['sales'].sum().reset_index()
            df_agg['date_label'] = df_agg['date'].dt.strftime('%Y-%m')
            return df_agg, 'date_label', 'sales'
            
        elif agg_mode == 'rolling_monthly':
            df_sim['days_sl'] = (df_sim['date'] - launch_date).dt.days
            df_sim['month_idx'] = (df_sim['days_sl'] // 30) + 1
            df_agg = df_sim.groupby('month_idx')['sales'].sum().reset_index()
            df_agg['month_label'] = df_agg['month_idx'].apply(lambda x: f"Month {x}")
            return df_agg, 'month_label', 'sales'
            
        else: # 每日模式
            return df_sim, 'date', 'sales'
