import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import hashlib
import hmac
import logging
import subprocess

_logger = logging.getLogger(__name__)
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import os
import sys

from data_processor_v2 import get_integrated_dataset
from bass_engine import BassEngine, bass_S
from config_loader import cfg
from app import SalesSimulator
from auth import hash_pw as _hash_pw, verify_pw as _verify_pw
from i18n import get_strings, get_col_labels, get_duration_labels, DURATION_OPTIONS, EDITABLE_COLS


# ── helpers ────────────────────────────────────────────────────────────────────

def _lang() -> str:
    return st.session_state.get('lang', 'en')


def S(key: str, *args) -> str:
    """Translate *key* using the current session language."""
    val = get_strings(_lang()).get(key, key)
    return val.format(*args) if args else val


# ── Admin helpers ──────────────────────────────────────────────────────────────

_ADMIN_CFG_PATH = 'models/admin.json'


# _hash_pw 和 _verify_pw 已提取到 auth.py，此处通过顶部 import 引入


def _load_cfg() -> dict:
    try:
        with open(_ADMIN_CFG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cfg(data: dict):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(_ADMIN_CFG_PATH)), exist_ok=True)
        with open(_ADMIN_CFG_PATH, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


def _load_admin_hash():
    """Returns the stored admin password hash, or None if not configured."""
    return _load_cfg().get('password_hash')


def _save_admin_hash(h: str):
    cfg = _load_cfg()
    cfg['password_hash'] = h
    _save_cfg(cfg)


def _load_app_hash():
    """Returns the stored app access password hash, or None if not configured."""
    return _load_cfg().get('app_hash')


def _save_app_hash(h: str):
    cfg = _load_cfg()
    cfg['app_hash'] = h
    _save_cfg(cfg)


# ── File signature validation (H3) ────────────────────────────────────────────

_FILE_MAGIC = {
    '.xlsx': b'PK\x03\x04',
    '.xlsm': b'PK\x03\x04',
    '.xlsb': b'PK\x03\x04',
    '.ods':  b'PK\x03\x04',
    '.xls':  b'\xd0\xcf\x11\xe0',
}


def _validate_file_magic(path: str, suffix: str) -> bool:
    """Return True if the file's magic bytes match the expected format for *suffix*.
    CSV files have no magic bytes and always pass. Unknown extensions also pass."""
    expected = _FILE_MAGIC.get(suffix.lower())
    if expected is None:
        return True
    try:
        with open(path, 'rb') as f:
            return f.read(len(expected)) == expected
    except Exception:
        return False


def _is_authenticated() -> bool:
    """用户是否已通过应用访问密码验证。"""
    return st.session_state.get('app_authenticated', False)


def _is_admin() -> bool:
    return st.session_state.get('admin_logged_in', False)


# ── 1. Resource loading (cached) ───────────────────────────────────────────────

@st.cache_resource
def load_resources(_cache_buster=None):
    resources = {}
    try:
        model = xgb.XGBRegressor()
        model.load_model('models/xgb_model.json')
        resources['model'] = model

        cold_model = xgb.XGBRegressor()
        cold_model.load_model('models/xgb_cold_start.json')
        resources['cold_model'] = cold_model

        with open('models/feature_cols.json', 'r') as f:
            resources['feature_cols'] = json.load(f)
        with open('models/cold_start_cols.json', 'r') as f:
            resources['cold_cols'] = json.load(f)

        test_results = pd.read_csv('models/test_results.csv')
        test_results['date'] = pd.to_datetime(test_results['date'])
        resources['test_results'] = test_results

        full_df = get_integrated_dataset('data/raw_data.xlsx')
        full_df['date'] = pd.to_datetime(full_df['date'])
        full_df = full_df.sort_values(by=['model_key', 'date'])
        resources['full_df'] = full_df
        resources['median_specs'] = full_df[EDITABLE_COLS].median().to_dict()

        bass_engine = BassEngine()
        bass_engine.train_on_sheet2('data/raw_data.xlsx')
        # C3: bass_engine 由 @st.cache_resource 在所有 session 间共享（单例）。
        # train_on_sheet2 完成后 params 字典不再被修改，所有下游调用
        #（calculate_theoretical_sales / get_bass_params）均为只读，不存在竞态条件。
        # 若未来需要支持在线重训，必须重新实例化并更新缓存，而非原地修改 self.params。
        resources['bass_engine'] = bass_engine

    except Exception as e:
        _logger.exception('Resource loading failed')
        st.error(S('st_res_load_err', '请检查模型文件是否完整（详情见服务器日志）。'))
        st.stop()
    return resources


def load_presets():
    try:
        if os.path.exists('models/user_configs.json'):
            with open('models/user_configs.json', 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_presets(presets):
    with open('models/user_configs.json', 'w') as f:
        json.dump(presets, f, indent=4)


# ── 2. Simulator page ──────────────────────────────────────────────────────────

def page_simulator(res, simulator):
    st.title(S('st_sim_title'))
    st.caption(S('st_sim_caption'))
    st.divider()

    col_left, col_right = st.columns([1, 2.2], gap="large")

    with col_left:
        presets = load_presets()
        preset_names = [p['name'] for p in presets]
        all_models = sorted(res['full_df']['model_key'].unique())

        lang = _lang()
        options = ["== NEW PRODUCT LAUNCH =="]
        if preset_names:
            options += ["--- USER PRESETS ---"] + preset_names
        options += ["--- HISTORICAL MODELS ---"] + all_models

        if 'selected_product' not in st.session_state:
            st.session_state.selected_product = options[0]

        def on_product_change():
            sel = st.session_state.product_selector
            preset_data = next((p for p in presets if p['name'] == sel), None)

            if sel == "== NEW PRODUCT LAUNCH ==" or preset_data:
                st.session_state.launch_date = pd.Timestamp.now().date()
                st.session_state.current_context = {
                    'is_new_launch': True,
                    'last_7d_sales': [0.0] * 7,
                    'original_specs': res['median_specs'].copy()
                }
                target_specs = preset_data['specs'] if preset_data else res['median_specs']
                for k in EDITABLE_COLS:
                    st.session_state[f"input_{k}"] = float(target_specs.get(k, 0))
                st.session_state.input_bass_m = float(preset_data.get('bass_m', 100000)) if preset_data else 100000.0
                st.session_state.input_bass_p = float(preset_data.get('bass_p', 0.05)) if preset_data else res['bass_engine'].avg_p

            elif sel not in ["--- USER PRESETS ---", "--- HISTORICAL MODELS ---"]:
                subset = res['full_df'][res['full_df']['model_key'] == sel]
                if not subset.empty:
                    latest_row = subset.iloc[-1]
                    real_launch = subset['date'].min()
                    st.session_state.launch_date = real_launch.date()
                    for k in EDITABLE_COLS:
                        st.session_state[f"input_{k}"] = float(latest_row.get(k, 0))
                    st.session_state.input_bass_m = 0.0
                    st.session_state.input_bass_p = 0.0
                    st.session_state.current_context = {
                        'is_new_launch': False,
                        'launch_date': real_launch,
                        'last_date': subset['date'].max(),
                        'last_7d_sales': subset['daily_sales'].tail(7).tolist(),
                        'original_specs': latest_row[EDITABLE_COLS].to_dict()
                    }

        st.selectbox(
            S('st_sel_prod'),
            options=options,
            key="product_selector",
            on_change=on_product_change,
            label_visibility="visible",
        )

        if 'input_current_price' not in st.session_state:
            on_product_change()

        col_labels = get_col_labels(lang)
        dur_labels = get_duration_labels(lang)

        with st.form("sim_config_form"):
            # ── 紧凑顶部行：发布日期 + 预测时长 ──────────────────────────────
            top_c1, top_c2 = st.columns(2)
            with top_c1:
                st.markdown(S('st_launch_date'))
                st.date_input(S('launch_date'), key="launch_date", label_visibility="collapsed")
            with top_c2:
                st.markdown(S('st_dur_hdr'))
                dur_idx = st.selectbox(
                    S('st_dur_hdr'),
                    options=range(len(DURATION_OPTIONS)),
                    format_func=lambda i: dur_labels[i],
                    label_visibility="collapsed",
                )

            # ── 顶部运行按钮（无需滚动即可点击）─────────────────────────────
            top_submitted = st.form_submit_button(
                S('st_run_btn'), type="primary", use_container_width=True
            )

            # ── 规格与价格（默认展开）────────────────────────────────────────
            with st.expander(S('st_specs_hdr'), expanded=True):
                left_specs  = ['current_price', 'ram_gb', 'storage_gb', 'battery_mah', 'refresh_rate_hz']
                right_specs = ['main_camera_mp', 'charging_w', 'screen_res', 'ip_rating']
                fc1, fc2 = st.columns(2)
                with fc1:
                    for col in left_specs:
                        st.number_input(col_labels[col], key=f"input_{col}", min_value=0.0)
                with fc2:
                    for col in right_specs:
                        st.number_input(col_labels[col], key=f"input_{col}", min_value=0.0)

            # ── Bass 参数（默认折叠，减少视觉噪音）──────────────────────────
            with st.expander(S('st_bass_hdr'), expanded=False):
                bc1, bc2 = st.columns(2)
                with bc1:
                    st.number_input(S('st_market_pot'), key="input_bass_m", min_value=0.0, step=1000.0)
                with bc2:
                    st.number_input(S('st_innov_coeff'), key="input_bass_p", format="%.4f", step=0.0001, min_value=0.0)

            # ── 底部运行按钮（供调整参数后就近点击）─────────────────────────
            bot_submitted = st.form_submit_button(
                S('st_run_btn'), type="primary", use_container_width=True,
                key="sim_run_btn_bottom"
            )

            submitted = top_submitted or bot_submitted

        with st.expander(S('st_preset_mgmt')):
            preset_name = st.text_input(S('st_preset_name'))
            cs, cd = st.columns(2)
            if cs.button(S('st_save_preset'), use_container_width=True):
                if not preset_name:
                    st.error(S('st_no_name_err'))
                else:
                    specs = {k: st.session_state[f"input_{k}"] for k in EDITABLE_COLS}
                    new_p = {
                        'name': preset_name, 'specs': specs,
                        'bass_m': st.session_state.input_bass_m,
                        'bass_p': st.session_state.input_bass_p
                    }
                    presets = [p for p in presets if p['name'] != preset_name]
                    presets.append(new_p)
                    if len(presets) > 10:
                        presets.pop(0)
                    save_presets(presets)
                    st.success(S('st_preset_saved', preset_name))
                    st.rerun()
            if cd.button(S('st_del_preset'), use_container_width=True):
                sel = st.session_state.get('product_selector', '')
                if sel in preset_names:
                    save_presets([p for p in presets if p['name'] != sel])
                    st.success(S('st_preset_deleted', sel))
                    st.rerun()
                else:
                    st.warning(S('st_sel_preset_warn'))

    with col_right:
        # 显示剩余配额（仅当用过至少一次后）
        from rate_limiter import RateLimiter
        _rl_disp = RateLimiter.from_list(st.session_state.get('sim_rl_ts', []))
        if st.session_state.get('sim_rl_ts'):
            st.caption(f"🔢 本窗口期剩余模拟次数：**{_rl_disp.remaining()}** / {_rl_disp.max_calls}")
        if submitted:
            _run_and_store_simulation(res, simulator, dur_idx)
        _render_simulation_results()


def _run_and_store_simulation(res, simulator, dur_idx: int):
    # ── 限速检查 ──────────────────────────────────────────────────────────────
    from rate_limiter import RateLimiter
    rl = RateLimiter.from_list(st.session_state.get('sim_rl_ts', []))
    allowed, reason = rl.check()
    if not allowed:
        st.warning(f"⏳ {reason}  （本窗口期剩余可用：**{rl.remaining()}** 次）")
        return
    # 通过检查后先记录时间戳，防止 rerun 导致重复执行
    rl.record()
    st.session_state['sim_rl_ts'] = rl.to_list()

    lang = _lang()
    context = st.session_state.get('current_context', {})
    sel_product = st.session_state.get('product_selector', '')
    presets = load_presets()
    preset_names = [p['name'] for p in presets]
    is_new = (sel_product == "== NEW PRODUCT LAUNCH ==") or (sel_product in preset_names)

    ui_launch_dt = pd.to_datetime(st.session_state.launch_date).normalize()
    if is_new:
        context['launch_date'] = ui_launch_dt
        context['last_date'] = ui_launch_dt - timedelta(days=1)
        if 'original_specs' not in context:
            context['original_specs'] = res['median_specs'].copy()
    context['is_new_launch'] = is_new
    context['model_key'] = sel_product

    _, end_day, agg_mode = DURATION_OPTIONS[dur_idx]

    user_specs = {k: st.session_state[f"input_{k}"] for k in EDITABLE_COLS}
    bass_params = {
        'm': st.session_state.input_bass_m,
        'p': st.session_state.input_bass_p,
    }

    # M4: bounds validation
    _SPEC_BOUNDS = {
        'current_price':    (1,      9_999_999),
        'battery_mah':      (500,    30_000),
        'ram_gb':           (1,      256),
        'storage_gb':       (8,      4_096),
        'refresh_rate_hz':  (1,      500),
        'main_camera_mp':   (1,      500),
        'charging_w':       (1,      500),
        'screen_res':       (100,    16_000),
        'ip_rating':        (0,      99),
    }
    bound_errors = []
    for col, (lo, hi) in _SPEC_BOUNDS.items():
        v = user_specs.get(col)
        if v is not None and not (lo <= float(v) <= hi):
            bound_errors.append(f'`{col}`: 有效范围 {lo}–{hi}，当前值 {v}')
    m_val = float(bass_params['m'])
    p_val = float(bass_params['p'])
    if not (100 <= m_val <= 10_000_000):
        bound_errors.append(f'`Bass m`: 有效范围 100–10,000,000，当前值 {m_val}')
    if not (0.001 <= p_val <= 1.0):
        bound_errors.append(f'`Bass p`: 有效范围 0.001–1.0，当前值 {p_val}')
    if bound_errors:
        st.error('参数超出有效范围：\n\n' + '\n\n'.join(f'- {e}' for e in bound_errors))
        return

    dur_label = get_duration_labels(lang)[dur_idx]

    with st.spinner(S('st_calculating')):
        try:
            sim_results = simulator.run_simulation(
                context=context,
                user_specs=user_specs,
                bass_params=bass_params,
                duration_days=end_day,
            )
            plot_df, x_col, y_col = simulator.aggregate_results(
                sim_results, agg_mode, context['launch_date']
            )

            try:
                from db import init_db, save_simulation_result
                init_db()
                save_simulation_result(
                    {
                        'product_name': sel_product,
                        'launch_date': str(context['launch_date'])[:10],
                        'duration_days': end_day,
                        'agg_mode': agg_mode,
                        'user_specs': user_specs,
                        'bass_params': bass_params,
                    },
                    plot_df,
                    source='streamlit',
                )
            except Exception:
                pass

            st.session_state['sim_output'] = {
                'plot_df':      plot_df,
                'x_col':        x_col,
                'y_col':        y_col,
                'dur_idx':      dur_idx,
                'dur_label':    dur_label,
                'product_name': sel_product,
                'agg_mode':     agg_mode,
            }
        except Exception as e:
            st.error(S('st_sim_err', e))


def _render_simulation_results():
    lang = _lang()
    if 'sim_output' not in st.session_state:
        st.info(S('st_no_sim'))
        return

    out = st.session_state['sim_output']
    plot_df      = out['plot_df']
    x_col        = out['x_col']
    y_col        = out['y_col']
    dur_label    = out.get('dur_label', get_duration_labels(lang)[out.get('dur_idx', 0)])
    product_name = out['product_name']
    agg_mode     = out['agg_mode']

    total_sales = plot_df[y_col].sum()
    peak_sales  = plot_df[y_col].max()
    avg_sales   = plot_df[y_col].mean()
    avg_key     = 'st_avg_monthly' if 'month' in agg_mode else 'st_avg_daily'

    st.markdown(S('st_result_hdr', product_name, dur_label))
    m1, m2, m3 = st.columns(3)
    m1.metric(S('st_total_sales'), S('st_units_fmt', total_sales))
    m2.metric(S('st_peak_sales'),  S('st_units_fmt', peak_sales))
    m3.metric(S(avg_key),          S('st_units_fmt', avg_sales))

    if 'month' in agg_mode:
        fig = px.bar(
            plot_df, x=x_col, y=y_col,
            labels={y_col: S('st_monthly_y'), x_col: S('st_monthly_x')},
            color_discrete_sequence=["#2563EB"],
            text=y_col,
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', marker_line_width=0)
    else:
        fig = px.line(
            plot_df, x=x_col, y=y_col,
            labels={y_col: S('st_daily_y'), x_col: S('st_daily_x')},
            markers=True,
            color_discrete_sequence=["#2563EB"],
        )
        fig.update_traces(line_width=2.5, marker_size=5)
        fig.add_traces(px.area(
            plot_df, x=x_col, y=y_col,
            color_discrete_sequence=["rgba(37,99,235,0.12)"]
        ).data)

    fig.update_layout(
        height=400,
        margin=dict(t=20, b=10, l=0, r=0),
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showline=True, linecolor='#CBD5E1'),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', tickformat=',d', zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(S('st_view_export')):
        display_df = plot_df.copy()
        if 'date' in display_df.columns and pd.api.types.is_datetime64_any_dtype(display_df['date']):
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(
            display_df.style.format({y_col: '{:,.0f}'}),
            use_container_width=True,
            hide_index=True,
        )
        json_str = display_df.to_json(orient='records', indent=4, force_ascii=False)
        st.download_button(
            label=S('st_export_btn'),
            data=json_str,
            file_name="simulation_results.json",
            mime="application/json",
        )


# ── 3. Test backtest page ──────────────────────────────────────────────────────

def page_viz(res):
    st.title(S('st_viz_title'))
    st.caption(S('st_viz_caption'))
    st.divider()

    test_df = res['test_results']
    models = sorted(test_df['model_key'].unique())
    sel = st.selectbox(S('st_sel_model'), models)
    subset = test_df[test_df['model_key'] == sel].sort_values('date')

    mae   = subset['abs_error'].mean()
    wmape = subset['abs_error'].sum() / subset['actual'].sum() if subset['actual'].sum() > 0 else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",             f"{mae:.2f}")
    m2.metric("WMAPE",           f"{wmape:.2%}")
    m3.metric(S('st_test_days'), S('st_days_fmt', len(subset)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset['date'], y=subset['actual'],
        mode='lines+markers', name=S('st_actual_trace'),
        line=dict(color='#2563EB', width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=subset['date'], y=subset['predicted'],
        mode='lines+markers', name=S('st_pred_trace'),
        line=dict(color='#EF4444', width=2, dash='dot'),
        marker=dict(size=4, symbol='x'),
    ))
    fig.update_layout(
        height=380,
        hovermode="x unified",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(showgrid=False, showline=True, linecolor='#CBD5E1'),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', tickformat=',d', zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(S('st_err_dist')):
        fig_err = px.histogram(
            subset, x='error', nbins=30,
            labels={'error': S('st_err_axis'), 'count': S('st_count_axis')},
            color_discrete_sequence=['#8B5CF6'],
        )
        fig_err.add_vline(x=0, line_dash="dash", line_color="#64748B", annotation_text="0")
        fig_err.update_layout(
            height=260, plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_err, use_container_width=True)


# ── 4. Model diagnostics page ──────────────────────────────────────────────────

def page_diag(res):
    st.title(S('st_diag_title'))
    st.caption(S('st_diag_caption'))
    st.divider()

    df   = res['test_results']
    mae  = df['abs_error'].mean()
    wmape = df['abs_error'].sum() / df['actual'].sum()
    rmse = np.sqrt((df['error'] ** 2).mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("WMAPE", f"{wmape:.2%}", help=S('st_wmape_help'))
    c2.metric("MAE",   f"{mae:.2f}",  help=S('st_mae_help'))
    c3.metric("RMSE",  f"{rmse:.2f}", help=S('st_rmse_help'))

    st.subheader(S('st_feat_imp_hdr'))
    booster  = res['model'].get_booster()
    scores   = booster.get_score(importance_type='gain')
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
    feat_df  = pd.DataFrame(sorted_s, columns=[S('st_feat_label'), S('st_gain_label')])

    gain_col = S('st_gain_label')
    feat_col = S('st_feat_label')
    fig = px.bar(
        feat_df.sort_values(gain_col), x=gain_col, y=feat_col,
        orientation='h',
        color=gain_col,
        color_continuous_scale='Blues',
        labels={gain_col: gain_col},
        text=gain_col,
    )
    fig.update_traces(texttemplate='%{text:,.1f}', textposition='outside', marker_line_width=0)
    fig.update_layout(
        height=460,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=0, r=60),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 5. History page ────────────────────────────────────────────────────────────

def page_history():
    st.title(S('st_hist_title'))
    st.caption(S('st_hist_caption'))
    st.divider()

    try:
        from db import db_exists, get_simulation_history, get_training_history
    except ImportError:
        st.error(S('st_db_missing'))
        return

    if not db_exists():
        st.info(S('st_db_not_init'))
        return

    tab_sim, tab_train = st.tabs([S('st_sim_tab'), S('st_train_tab')])

    with tab_sim:
        df_sim = get_simulation_history(limit=50)
        if df_sim.empty:
            st.info(S('st_no_sim_rec'))
        else:
            st.dataframe(df_sim, use_container_width=True, hide_index=True)

    with tab_train:
        df_train = get_training_history(limit=20)
        if df_train.empty:
            st.info(S('st_no_train_rec'))
        else:
            st.dataframe(
                df_train.style.format({'wmape': '{:.2%}', 'mae': '{:.2f}', 'rmse': '{:.2f}'}),
                use_container_width=True,
                hide_index=True,
            )


# ── 6. Data import page ────────────────────────────────────────────────────────

def _ingest_from_upload(uploaded_file, file_type: str) -> int:
    import tempfile
    suffix = os.path.splitext(uploaded_file.name)[1] or '.xlsx'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        # H3: validate file signature before processing
        if not _validate_file_magic(tmp_path, suffix):
            raise ValueError(
                f'文件签名不匹配："{uploaded_file.name}" 不是有效的 {suffix} 文件'
                f'（可能已损坏或被伪装为其他格式）。')
        if file_type == 'specs':
            from data_processor_v2 import ingest_specs
            return ingest_specs(tmp_path)
        else:
            from data_processor_v2 import ingest_daily_sales
            return ingest_daily_sales(tmp_path)
    finally:
        os.unlink(tmp_path)


def page_ingest():
    # ── 未登录时显示锁定提示 ──────────────────────────────────────────────────
    if not _is_admin():
        st.title(S('ingest_locked_title'))
        st.divider()
        st.warning(S('ingest_locked_hint'))
        if st.button(S('ingest_locked_btn'), type='primary'):
            st.session_state['_nav_override'] = 'nav_admin'
            st.rerun()
        return

    st.title(S('st_ingest_title'))
    st.caption(S('st_ingest_caption'))
    st.divider()

    col_specs, col_sales = st.columns(2, gap="large")

    with col_specs:
        st.subheader(S('st_specs_sub'))
        st.caption(S('st_specs_cap'))
        specs_file = st.file_uploader(
            S('st_up_specs'),
            type=['xlsx', 'xls', 'xlsm', 'xlsb', 'ods'],
            key='upload_specs',
        )
        if specs_file:
            st.success(S('st_file_chosen', specs_file.name))

    with col_sales:
        st.subheader(S('st_sales_sub'))
        st.caption(S('st_sales_cap'))
        sales_file = st.file_uploader(
            S('st_up_sales'),
            type=['xlsx', 'xls', 'xlsm', 'xlsb', 'ods', 'csv'],
            key='upload_sales',
        )
        if sales_file:
            st.success(S('st_file_chosen', sales_file.name))

    st.divider()

    c1, c2, c3 = st.columns([1, 1, 1.5])
    do_specs = c1.button(S('st_imp_specs_btn'), disabled=(specs_file is None), use_container_width=True)
    do_sales = c2.button(S('st_imp_sales_btn'), disabled=(sales_file is None), use_container_width=True)
    do_all   = c3.button(S('st_imp_all_btn'),
                         disabled=(specs_file is None or sales_file is None),
                         type="primary", use_container_width=True)

    results = {}

    if do_specs or do_all:
        if specs_file:
            with st.spinner(S('st_imp_specs_spin')):
                try:
                    results['specs'] = _ingest_from_upload(specs_file, 'specs')
                except Exception as e:
                    st.error(S('st_specs_err', e))

    if do_sales or do_all:
        if sales_file:
            with st.spinner(S('st_imp_sales_spin')):
                try:
                    results['sales'] = _ingest_from_upload(sales_file, 'sales')
                except Exception as e:
                    st.error(S('st_sales_err', e))

    if results:
        st.divider()
        st.subheader(S('st_imp_results'))
        m1, m2 = st.columns(2)
        if 'specs' in results:
            m1.metric(S('st_specs_rec'), f"{results['specs']}", help=S('st_specs_help'))
        if 'sales' in results:
            m2.metric(S('st_sales_rec'), f"{results['sales']}", help=S('st_sales_help'))
        st.success(S('st_imp_success'))
        st.info(S('st_retrain_hint'))


def page_admin():
    """Admin control page — login form or full admin panel."""
    st.title(S('admin_login_title') if not _is_admin() else '🔐 ' + S('admin_logged_in_as'))
    st.divider()

    if not _is_admin():
        # ── 登录表单 ──────────────────────────────────────────────────────────
        col, _ = st.columns([1, 2])
        with col:
            with st.form('admin_login_form'):
                pw = st.text_input(S('admin_pw_label'), type='password')
                submitted = st.form_submit_button(S('admin_login_btn'),
                                                  use_container_width=True,
                                                  type='primary')
            if submitted:
                stored_admin = _load_admin_hash()
                if _verify_pw(pw, stored_admin):
                    # Auto-upgrade legacy SHA-256 hash to PBKDF2
                    if stored_admin and ':' not in stored_admin:
                        _save_admin_hash(_hash_pw(pw))
                    st.session_state['admin_logged_in'] = True
                    st.rerun()
                else:
                    st.error(S('admin_wrong_pw'))
        return

    # ── 已登录：管理面板 ──────────────────────────────────────────────────────
    if st.button(S('admin_logout_btn')):
        st.session_state['admin_logged_in'] = False
        st.rerun()

    # ── 修改密码 ───────────────────────────────────────────────────────────────
    col_pw1, col_pw2 = st.columns(2, gap='large')

    with col_pw1:
        with st.expander(S('admin_change_pw_sec'), expanded=False):
            with st.form('change_pw_form'):
                new_pw = st.text_input(S('admin_new_pw'), type='password')
                cf_pw  = st.text_input(S('admin_confirm_pw'), type='password')
                if st.form_submit_button(S('admin_change_pw_btn')):
                    if not new_pw:
                        st.error(S('admin_pw_empty'))
                    elif new_pw != cf_pw:
                        st.error(S('admin_pw_mismatch'))
                    else:
                        _save_admin_hash(_hash_pw(new_pw))
                        st.success(S('admin_pw_changed'))

    with col_pw2:
        with st.expander(S('admin_app_pw_sec'), expanded=False):
            st.caption(S('admin_app_pw_hint'))
            with st.form('change_app_pw_form'):
                new_app_pw = st.text_input(S('admin_new_pw'), type='password', key='new_app_pw')
                cf_app_pw  = st.text_input(S('admin_confirm_pw'), type='password', key='cf_app_pw')
                if st.form_submit_button(S('admin_change_pw_btn')):
                    if not new_app_pw:
                        st.error(S('admin_pw_empty'))
                    elif new_app_pw != cf_app_pw:
                        st.error(S('admin_pw_mismatch'))
                    else:
                        _save_app_hash(_hash_pw(new_app_pw))
                        st.success(S('admin_pw_changed'))

    st.divider()

    # ── 训练记录 ───────────────────────────────────────────────────────────────
    st.subheader(S('admin_train_sec'))
    try:
        from db import get_training_history, delete_training_run, delete_all_training_runs
        df_train = get_training_history(limit=50)
    except Exception as e:
        _logger.exception('Failed to load training history')
        st.error('训练记录加载失败，请检查数据库。'); df_train = pd.DataFrame()

    if df_train.empty:
        st.info(S('st_no_train_rec'))
    else:
        st.dataframe(
            df_train.style.format({'wmape': '{:.2%}', 'mae': '{:.2f}', 'rmse': '{:.2f}'}),
            use_container_width=True, hide_index=True,
        )
        tc1, tc2, tc3 = st.columns([2, 1, 1])
        sel_train_id = tc1.selectbox(
            'ID', options=df_train['id'].tolist(),
            label_visibility='collapsed', key='sel_train_id',
        )
        if tc2.button(S('admin_del_sel_btn'), key='del_train', use_container_width=True):
            delete_training_run(int(sel_train_id))
            st.success(S('admin_deleted_fmt', 1)); st.rerun()
        if tc3.button(S('admin_clear_all_btn'), key='clear_train',
                      use_container_width=True, type='secondary'):
            st.session_state['_confirm_clear_train'] = True

        if st.session_state.get('_confirm_clear_train'):
            st.warning(S('admin_clear_train_msg'))
            cc1, cc2, _ = st.columns([1, 1, 3])
            if cc1.button('✅ ' + S('admin_confirm_clear'), key='confirm_clear_train'):
                n = delete_all_training_runs()
                st.session_state.pop('_confirm_clear_train', None)
                st.success(S('admin_deleted_fmt', n)); st.rerun()
            if cc2.button('❌ Cancel', key='cancel_clear_train'):
                st.session_state.pop('_confirm_clear_train', None); st.rerun()

    st.divider()

    # ── 模拟历史 ───────────────────────────────────────────────────────────────
    st.subheader(S('admin_sim_sec'))
    try:
        from db import get_simulation_history, delete_simulation_result, delete_all_simulation_results
        df_sim = get_simulation_history(limit=50)
    except Exception as e:
        _logger.exception('Failed to load simulation history')
        st.error('模拟历史加载失败，请检查数据库。'); df_sim = pd.DataFrame()

    if df_sim.empty:
        st.info(S('st_no_sim_rec'))
    else:
        st.dataframe(df_sim, use_container_width=True, hide_index=True)
        sc1, sc2, sc3 = st.columns([2, 1, 1])
        sel_sim_id = sc1.selectbox(
            'ID', options=df_sim['id'].tolist(),
            label_visibility='collapsed', key='sel_sim_id',
        )
        if sc2.button(S('admin_del_sel_btn'), key='del_sim', use_container_width=True):
            delete_simulation_result(int(sel_sim_id))
            st.success(S('admin_deleted_fmt', 1)); st.rerun()
        if sc3.button(S('admin_clear_all_btn'), key='clear_sim',
                      use_container_width=True, type='secondary'):
            st.session_state['_confirm_clear_sim'] = True

        if st.session_state.get('_confirm_clear_sim'):
            st.warning(S('admin_clear_sim_msg'))
            cc1, cc2, _ = st.columns([1, 1, 3])
            if cc1.button('✅ ' + S('admin_confirm_clear'), key='confirm_clear_sim'):
                n = delete_all_simulation_results()
                st.session_state.pop('_confirm_clear_sim', None)
                st.success(S('admin_deleted_fmt', n)); st.rerun()
            if cc2.button('❌ Cancel', key='cancel_clear_sim'):
                st.session_state.pop('_confirm_clear_sim', None); st.rerun()

    st.divider()

    # ── 重新训练 ───────────────────────────────────────────────────────────────
    st.subheader(S('admin_retrain_sec'))
    if st.button(S('admin_retrain_btn'), type='primary'):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_daily.py')
        cwd    = os.path.dirname(os.path.dirname(script))
        with st.spinner(S('admin_retrain_running')):
            try:
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True, text=True, cwd=cwd,
                )
                output = result.stdout + result.stderr
                st.code(output if output.strip() else S('admin_retrain_done'),
                        language='text')
                if result.returncode == 0:
                    st.success(S('admin_retrain_done'))
                    # 清除资源缓存，下次访问时自动重新加载新模型
                    load_resources.clear()
                else:
                    st.error(S('admin_retrain_err', f'exit code {result.returncode}'))
            except Exception as e:
                st.error(S('admin_retrain_err', e))


def page_manual():
    """User Manual page — renders the bilingual user_manual.md file."""
    manual_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_manual.md')
    try:
        with open(manual_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        _logger.exception('Could not load user manual')
        st.error('用户手册加载失败，请确认 src/user_manual.md 文件存在。')
        return

    en_marker = '# Part 2: English User Guide'
    if en_marker in content:
        zh_part, en_part = content.split(en_marker, 1)
        en_part = en_marker + en_part
    else:
        zh_part = en_part = content

    st.markdown(zh_part if _lang() == 'zh' else en_part)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Initialise language before any S() call
    if 'lang' not in st.session_state:
        st.session_state['lang'] = 'en'

    st.set_page_config(
        page_title=S('st_page_title'),
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── 首次运行：admin.json 不存在或 app_hash 缺失时强制补充设置 ──────────────
    _admin_missing = _load_admin_hash() is None
    _app_missing   = _load_app_hash()   is None

    if _admin_missing or _app_missing:
        _c1, _c2, _c3 = st.columns([1, 1.5, 1])
        with _c2:
            if _admin_missing:
                # 完整首次设置（从未配置过任何密码）
                st.markdown('## 🔐 初始安全设置 / Initial Setup')
                st.warning('首次启动：未检测到密码配置，请设置应用访问密码和管理员密码后继续。\n\n'
                           'First launch: no password configured — please set both passwords to proceed.')
                with st.form('first_run_form'):
                    app_pw1 = st.text_input('应用访问密码 / App Access Password', type='password')
                    app_pw2 = st.text_input('确认应用密码 / Confirm App Password', type='password')
                    adm_pw1 = st.text_input('管理员密码 / Admin Password', type='password')
                    adm_pw2 = st.text_input('确认管理员密码 / Confirm Admin Password', type='password')
                    if st.form_submit_button('保存并继续 / Save & Continue',
                                             use_container_width=True, type='primary'):
                        errors = []
                        if not app_pw1:
                            errors.append('应用访问密码不能为空。')
                        elif app_pw1 != app_pw2:
                            errors.append('应用访问密码两次输入不一致。')
                        if not adm_pw1:
                            errors.append('管理员密码不能为空。')
                        elif adm_pw1 != adm_pw2:
                            errors.append('管理员密码两次输入不一致。')
                        if errors:
                            for e in errors:
                                st.error(e)
                        else:
                            _save_app_hash(_hash_pw(app_pw1))
                            _save_admin_hash(_hash_pw(adm_pw1))
                            st.session_state['app_authenticated'] = True
                            st.success('密码设置成功，正在跳转…')
                            st.rerun()
            else:
                # 管理员密码已由桌面客户端配置，仅需补设用户访问密码
                st.markdown('## 🔐 补充设置 / Supplemental Setup')
                st.info('检测到管理员密码已由桌面客户端配置，请为 Streamlit 应用补充设置用户访问密码后继续。\n\n'
                        'Admin password was configured via the desktop app. '
                        'Please set the app access password to proceed.')
                with st.form('app_pw_setup_form'):
                    app_pw1 = st.text_input('应用访问密码 / App Access Password', type='password')
                    app_pw2 = st.text_input('确认应用密码 / Confirm App Password', type='password')
                    if st.form_submit_button('保存并继续 / Save & Continue',
                                             use_container_width=True, type='primary'):
                        errors = []
                        if not app_pw1:
                            errors.append('应用访问密码不能为空。')
                        elif app_pw1 != app_pw2:
                            errors.append('应用访问密码两次输入不一致。')
                        if errors:
                            for e in errors:
                                st.error(e)
                        else:
                            _save_app_hash(_hash_pw(app_pw1))
                            st.session_state['app_authenticated'] = True
                            st.success('密码设置成功，正在跳转…')
                            st.rerun()
        st.stop()

    # ── 全局访问门控：任何页面渲染前先验证应用密码 ─────────────────────────────
    if not _is_authenticated():
        _c1, _c2, _c3 = st.columns([1, 1.2, 1])
        with _c2:
            st.markdown(f"## {S('app_login_title')}")
            st.info(S('app_login_hint'))
            with st.form('app_login_form'):
                _pw = st.text_input(S('app_pw_label'), type='password')
                if st.form_submit_button(S('app_login_btn'),
                                         use_container_width=True, type='primary'):
                    stored_app = _load_app_hash()
                    if _verify_pw(_pw, stored_app):
                        # Auto-upgrade legacy SHA-256 hash to PBKDF2
                        if stored_app and ':' not in stored_app:
                            _save_app_hash(_hash_pw(_pw))
                        st.session_state['app_authenticated'] = True
                        st.rerun()
                    else:
                        st.error(S('app_wrong_pw'))
        st.stop()   # 阻止后续任何内容渲染

    with st.sidebar:
        st.markdown(S('st_sidebar_title'))
        st.caption(S('st_sidebar_ver'))
        st.divider()

        # Language selector
        lang_options = ['English', '中文']
        lang_map     = {'English': 'en', '中文': 'zh'}
        lang_reverse = {'en': 'English', 'zh': '中文'}
        chosen_display = st.selectbox(
            S('language'),
            options=lang_options,
            index=lang_options.index(lang_reverse.get(st.session_state['lang'], 'English')),
            key='lang_display_selector',
        )
        new_lang = lang_map[chosen_display]
        if new_lang != st.session_state['lang']:
            st.session_state['lang'] = new_lang
            st.rerun()

        st.divider()

        nav_keys   = ['nav_simulator', 'nav_viz', 'nav_diag', 'nav_history',
                      'nav_ingest', 'nav_admin', 'nav_manual']
        nav_labels = [S(k) for k in nav_keys]

        # 支持从其他页面（如数据导入锁定页）跳转到指定 tab
        _override = st.session_state.pop('_nav_override', None)
        default_idx = nav_keys.index(_override) if _override in nav_keys else 0

        page_idx   = st.radio(
            "navigation",
            options=range(len(nav_keys)),
            format_func=lambda i: nav_labels[i],
            label_visibility="collapsed",
            index=default_idx,
        )

        st.divider()
        st.caption(S('st_model_status'))
        try:
            mtime     = os.path.getmtime('models/xgb_model.json')
            mtime_str = pd.Timestamp(mtime, unit='s').strftime('%Y-%m-%d %H:%M')
            st.success(S('st_model_loaded', mtime_str))
        except OSError:
            st.error(S('st_model_missing'))

        st.divider()
        if st.button('🚪 ' + ('Log Out' if _lang() == 'en' else '退出登录'),
                     use_container_width=True):
            st.session_state['app_authenticated'] = False
            st.session_state['admin_logged_in']   = False
            st.rerun()

    try:
        _buster = os.path.getmtime('models/xgb_model.json')
    except OSError:
        _buster = None
    res = load_resources(_cache_buster=_buster)
    simulator = SalesSimulator(res)

    page_key = nav_keys[page_idx]
    if page_key == 'nav_simulator':
        page_simulator(res, simulator)
    elif page_key == 'nav_viz':
        page_viz(res)
    elif page_key == 'nav_diag':
        page_diag(res)
    elif page_key == 'nav_history':
        page_history()
    elif page_key == 'nav_ingest':
        page_ingest()
    elif page_key == 'nav_admin':
        page_admin()
    elif page_key == 'nav_manual':
        page_manual()


if __name__ == "__main__":
    main()
