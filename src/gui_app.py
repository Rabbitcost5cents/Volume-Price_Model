import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import pathlib
import logging
import pandas as pd
import xgboost as xgb
import json
import hashlib
import hmac
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import timedelta
import sys
import os
import threading
import queue as _queue

# 将工作目录锚定到项目根目录，确保 'models/' 和 'data/' 相对路径在任何启动方式下均可解析
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor_v2 import get_integrated_dataset

logger = logging.getLogger(__name__)

from tkinterdnd2 import TkinterDnD, DND_FILES
from bass_engine import BassEngine, bass_S
from config_loader import cfg
from app import SalesSimulator
from auth import hash_pw as _hash_pw, verify_pw as _verify_pw
from i18n import get_strings, get_col_labels, get_duration_labels, DURATION_OPTIONS, EDITABLE_COLS

# ── helpers ───────────────────────────────────────────────────────────────────

UI_SETTINGS_PATH = 'models/ui_settings.json'
ADMIN_CFG_PATH   = 'models/admin.json'
LANG_OPTIONS     = ['English', '中文']
LANG_CODE_MAP    = {'English': 'en', '中文': 'zh'}
LANG_DISPLAY_MAP = {'en': 'English', 'zh': '中文'}

# _hash_pw 和 _verify_pw 已提取到 auth.py，此处通过顶部 import 引入


def _load_admin_hash():
    """Returns the stored admin password hash, or None if not configured."""
    try:
        with open(ADMIN_CFG_PATH, 'r') as f:
            return json.load(f).get('password_hash')
    except Exception:
        return None


def _save_admin_hash(h: str):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(ADMIN_CFG_PATH)), exist_ok=True)
        try:
            with open(ADMIN_CFG_PATH, 'r') as f:
                data = json.load(f)
        except Exception:
            data = {}
        data['password_hash'] = h
        with open(ADMIN_CFG_PATH, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


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


def _load_lang() -> str:
    try:
        with open(UI_SETTINGS_PATH, 'r') as f:
            return json.load(f).get('lang', 'en')
    except Exception:
        return 'en'


def _save_lang(lang: str):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(UI_SETTINGS_PATH)), exist_ok=True)
        with open(UI_SETTINGS_PATH, 'w') as f:
            json.dump({'lang': lang}, f)
    except Exception:
        pass


# 线程安全说明（C2）：
# - _STDOUT_LOCK：保护 sys.stdout 的临时替换（_ingest_worker 中使用）。
# - self._log_queue：queue.Queue 是线程安全的，无需额外锁。
# - self.last_sim_data / self.entries：仅在主线程（GUI 事件循环）中读写，无竞态风险。
# - tkinter 规则：所有 widget 操作必须在主线程执行；工作线程通过 root.after() 投递回调。
_STDOUT_LOCK = threading.Lock()


class _QueueWriter:
    """Redirect print() output to a thread-safe queue for the GUI log widget."""
    def __init__(self, q: _queue.Queue):
        self._q = q

    def write(self, msg: str):
        if msg.strip():
            self._q.put(msg.strip())

    def flush(self):
        pass


def _parse_dnd_path(data: str) -> str:
    """Parse tkinterdnd2 drop event data; handles paths with spaces (Windows wraps in {})."""
    data = data.strip()
    if data.startswith('{'):
        end = data.find('}')
        if end > 0:
            return data[1:end]
    return data.split()[0] if data else ''


# ── Main application ──────────────────────────────────────────────────────────

class SalesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.lang = _load_lang()

        self.root.title(self.T('app_title'))
        self.root.geometry("1300x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ── 立即显示加载画面，让窗口先出现 ──────────────────────────────────
        splash = ttk.Frame(self.root)
        splash.pack(fill=tk.BOTH, expand=True)
        splash_msg = tk.StringVar(value='Loading resources, please wait…')
        ttk.Label(splash, textvariable=splash_msg,
                  font=('', 13)).pack(expand=True)
        self.root.update()          # 强制渲染窗口（在 mainloop 前显示）

        def _splash_step(text):
            splash_msg.set(text)
            self.root.update_idletasks()

        # Load artifacts
        try:
            _splash_step('Loading model and artifacts…')
            print("Loading model and artifacts...")
            self.model = xgb.XGBRegressor()
            self.model.load_model('models/xgb_model.json')

            self.cold_model = xgb.XGBRegressor()
            self.cold_model.load_model('models/xgb_cold_start.json')

            with open('models/feature_cols.json', 'r') as f:
                self.feature_cols = json.load(f)
            with open('models/cold_start_cols.json', 'r') as f:
                self.cold_cols = json.load(f)

            self.test_results = pd.read_csv('models/test_results.csv')
            self.test_results['date'] = pd.to_datetime(self.test_results['date'])

            _splash_step('Loading full dataset for simulation context…')
            print("Loading full dataset for simulation context...")
            self.full_df = get_integrated_dataset('data/raw_data.xlsx')
            self.full_df['date'] = pd.to_datetime(self.full_df['date'])
            self.full_df = self.full_df.sort_values(by=['model_key', 'date'])
            self.median_specs = self.full_df[EDITABLE_COLS].median().to_dict()
            print(f"Median Specs for Cold Start: {self.median_specs}")

            _splash_step('Initializing Bass Engine…')
            print("Initializing Bass Engine...")
            self.bass_engine = BassEngine()
            self.bass_engine.train_on_sheet2('data/raw_data.xlsx')

            resources = {
                'model':        self.model,
                'cold_model':   self.cold_model,
                'feature_cols': self.feature_cols,
                'cold_cols':    self.cold_cols,
                'bass_engine':  self.bass_engine,
                'median_specs': self.median_specs,
            }
            self.simulator = SalesSimulator(resources)
            self.user_presets = []
            self.load_presets()

        except Exception as e:
            logger.exception("Application initialization failed")
            splash.destroy()
            messagebox.showerror("Error", self.T('err_load', e))
            root.destroy()
            return

        # 加载完毕，移除 splash 画面
        splash.destroy()

        # Admin & data import state
        self.admin_logged_in = False
        self._specs_path = tk.StringVar(value='')
        self._sales_path = tk.StringVar(value='')
        self._log_queue  = _queue.Queue()

        # 模拟运行限速器
        from rate_limiter import RateLimiter
        self._sim_rl = RateLimiter()

        self.setup_ui()

    # ── i18n helpers ──────────────────────────────────────────────────────────

    def T(self, key: str, *args) -> str:
        """Return translated string for the current language."""
        val = get_strings(self.lang).get(key, key)
        return val.format(*args) if args else val

    def _on_lang_change(self, event=None):
        chosen   = self.lang_var.get()
        new_lang = LANG_CODE_MAP.get(chosen, 'en')
        if new_lang == self.lang:
            return
        self.lang = new_lang
        _save_lang(new_lang)
        self.root.title(self.T('app_title'))
        self._rebuild_tabs()

    def _rebuild_tabs(self):
        """Clear and rebuild all tab contents after a language change."""
        for tab in (self.tab_viz, self.tab_sim, self.tab_diag,
                    self.tab_ingest, self.tab_admin, self.tab_manual):
            for w in tab.winfo_children():
                w.destroy()
        self.notebook.tab(self.tab_viz,    text=self.T('tab_viz'))
        self.notebook.tab(self.tab_sim,    text=self.T('tab_sim'))
        self.notebook.tab(self.tab_diag,   text=self.T('tab_diag'))
        self.notebook.tab(self.tab_ingest, text=self.T('tab_ingest'))
        self.notebook.tab(self.tab_admin,  text=self.T('tab_admin'))
        self.notebook.tab(self.tab_manual, text=self.T('tab_manual'))
        self.setup_viz_tab()
        self.setup_sim_tab()
        self.setup_diag_tab()
        self.setup_ingest_tab()
        self.setup_admin_tab()
        self.setup_manual_tab()

    # ── Preset helpers ────────────────────────────────────────────────────────

    def load_presets(self):
        try:
            with open('models/user_configs.json', 'r') as f:
                self.user_presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.user_presets = []

    def save_preset(self):
        current_selection = self.sim_model_var.get()
        is_existing_preset = any(p['name'] == current_selection for p in self.user_presets)
        name = None
        if is_existing_preset:
            choice = messagebox.askyesnocancel(
                self.T('save_preset_dlg'),
                self.T('overwrite_preset', current_selection),
            )
            if choice is True:
                name = current_selection
            elif choice is False:
                name = simpledialog.askstring(self.T('save_config_dlg'), self.T('new_preset_prompt'))
            else:
                return
        else:
            name = simpledialog.askstring(self.T('save_config_dlg'), self.T('preset_name_prompt'))
        if not name:
            return
        try:
            specs = {f: float(self.entries[f].get()) for f in EDITABLE_COLS}
            m_val = float(self.bass_m_entry.get())
            p_val = float(self.bass_p_entry.get())
            new_preset = {
                'name': name, 'specs': specs,
                'bass_m': m_val, 'bass_p': p_val,
                'timestamp': pd.Timestamp.now().isoformat(),
            }
            self.user_presets = [p for p in self.user_presets if p['name'] != name]
            self.user_presets.append(new_preset)
            if len(self.user_presets) > 10:
                self.user_presets.pop(0)
            with open('models/user_configs.json', 'w') as f:
                json.dump(self.user_presets, f, indent=4)
            self.update_product_dropdown()
            messagebox.showinfo(self.T('success'), self.T('preset_saved_msg', name))
        except Exception as e:
            logger.exception("Failed to save preset")
            messagebox.showerror("Error", self.T('err_save_preset', e))

    def export_json(self):
        if not hasattr(self, 'last_sim_data') or self.last_sim_data is None:
            messagebox.showwarning(self.T('warning'), self.T('err_no_sim_data'))
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title=self.T('export_json'),
        )
        if file_path:
            try:
                df_export = self.last_sim_data.copy()
                if 'date' in df_export.columns and pd.api.types.is_datetime64_any_dtype(df_export['date']):
                    df_export['date'] = df_export['date'].dt.strftime('%Y-%m-%d')
                df_export.to_json(file_path, orient='records', indent=4, force_ascii=False)
                messagebox.showinfo(self.T('success'), self.T('export_success', file_path))
            except Exception as e:
                logger.exception("Failed to export simulation results")
                messagebox.showerror("Error", self.T('err_export', e))

    def delete_preset(self):
        selected = self.sim_model_var.get()
        preset_to_delete = next((p for p in self.user_presets if p['name'] == selected), None)
        if not preset_to_delete:
            messagebox.showwarning(self.T('warning'), self.T('sel_preset_warn'))
            return
        if not messagebox.askyesno(self.T('confirm_delete'), self.T('delete_confirm_msg', selected)):
            return
        self.user_presets = [p for p in self.user_presets if p['name'] != selected]
        try:
            with open('models/user_configs.json', 'w') as f:
                json.dump(self.user_presets, f, indent=4)
            self.sim_combo.current(0)
            self.load_product_features(None)
            self.update_product_dropdown()
            messagebox.showinfo(self.T('success'), self.T('preset_deleted_msg', selected))
        except Exception as e:
            logger.exception("Failed to delete preset")
            messagebox.showerror("Error", self.T('err_del_preset', e))

    def update_product_dropdown(self):
        all_models = sorted(self.full_df['model_key'].unique())
        values = ["== NEW PRODUCT LAUNCH =="]
        if self.user_presets:
            values.append("== USER PRESETS ==")
            values.extend([p['name'] for p in self.user_presets])
        values.append("== HISTORICAL MODELS ==")
        values.extend(all_models)
        self.sim_combo['values'] = values

    # ── UI setup ──────────────────────────────────────────────────────────────

    def setup_ui(self):
        # Top bar with language selector
        top_bar = ttk.Frame(self.root)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=3)
        ttk.Label(top_bar, text=self.T('language') + ':').pack(side=tk.RIGHT, padx=(0, 2))
        self.lang_var = tk.StringVar(value=LANG_DISPLAY_MAP.get(self.lang, 'English'))
        lang_combo = ttk.Combobox(
            top_bar, textvariable=self.lang_var,
            values=LANG_OPTIONS, width=8, state='readonly',
        )
        lang_combo.pack(side=tk.RIGHT, padx=(0, 6))
        lang_combo.bind('<<ComboboxSelected>>', self._on_lang_change)

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.tab_viz    = ttk.Frame(self.notebook)
        self.tab_sim    = ttk.Frame(self.notebook)
        self.tab_diag   = ttk.Frame(self.notebook)
        self.tab_ingest = ttk.Frame(self.notebook)
        self.tab_admin  = ttk.Frame(self.notebook)
        self.tab_manual = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_viz,    text=self.T('tab_viz'))
        self.notebook.add(self.tab_sim,    text=self.T('tab_sim'))
        self.notebook.add(self.tab_diag,   text=self.T('tab_diag'))
        self.notebook.add(self.tab_ingest, text=self.T('tab_ingest'))
        self.notebook.add(self.tab_admin,  text=self.T('tab_admin'))
        self.notebook.add(self.tab_manual, text=self.T('tab_manual'))
        self.notebook.pack(expand=1, fill="both")

        self.setup_viz_tab()
        self.setup_sim_tab()
        self.setup_diag_tab()
        self.setup_ingest_tab()
        self.setup_admin_tab()
        self.setup_manual_tab()

    # ── Viz tab ───────────────────────────────────────────────────────────────

    def setup_viz_tab(self):
        control_frame = ttk.Frame(self.tab_viz)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Label(control_frame, text=self.T('select_model')).pack(side=tk.LEFT)
        model_keys = sorted(self.test_results['model_key'].unique())
        self.viz_model_var = tk.StringVar(value=model_keys[0])
        model_dropdown = ttk.Combobox(control_frame, textvariable=self.viz_model_var, values=model_keys)
        model_dropdown.pack(side=tk.LEFT, padx=5)
        model_dropdown.bind("<<ComboboxSelected>>", self.update_plot)
        self.plot_frame = ttk.Frame(self.tab_viz)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.update_plot()

    def update_plot(self, event=None):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        selected_model = self.viz_model_var.get()
        subset = self.test_results[self.test_results['model_key'] == selected_model].sort_values('date')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(subset['date'], subset['actual'], marker='o', label=self.T('actual'), alpha=0.7)
        for x, y in zip(subset['date'], subset['actual']):
            ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=8, color='blue')
        ax.plot(subset['date'], subset['predicted'], marker='x', linestyle='--',
                label=self.T('predicted'), alpha=0.7)
        for x, y in zip(subset['date'], subset['predicted']):
            ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                        xytext=(0, -10), ha='center', fontsize=8, color='orange')
        ax.set_title(self.T('actual_vs_pred', selected_model))
        ax.set_xlabel(self.T('date_axis'))
        ax.set_ylabel(self.T('daily_sales_vol'))
        ax.legend()
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Diag tab ──────────────────────────────────────────────────────────────

    def setup_diag_tab(self):
        actuals    = self.test_results['actual']
        errors     = self.test_results['error']
        abs_errors = self.test_results['abs_error']
        mae   = abs_errors.mean()
        rmse  = np.sqrt((errors ** 2).mean())
        wmape = abs_errors.sum() / actuals.sum() if actuals.sum() != 0 else 0
        baseline_rmse = np.sqrt(((actuals - actuals.mean()) ** 2).mean())

        metrics_frame = ttk.LabelFrame(self.tab_diag, text=self.T('performance_metrics'))
        metrics_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        for i, (label, value) in enumerate([
            ("WMAPE",                 f"{wmape:.2%}"),
            ("MAE",                   f"{mae:.2f}"),
            ("RMSE",                  f"{rmse:.2f}"),
            (self.T('baseline_rmse'), f"{baseline_rmse:.2f}"),
        ]):
            frame = ttk.Frame(metrics_frame)
            frame.grid(row=0, column=i, padx=20, pady=10, sticky="ew")
            ttk.Label(frame, text=label, font=("Helvetica", 10, "bold")).pack()
            ttk.Label(frame, text=value, font=("Helvetica", 14, "bold"),
                      foreground="#007acc").pack()
        for i in range(4):
            metrics_frame.columnconfigure(i, weight=1)

        plot_container = ttk.Frame(self.tab_diag)
        plot_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        ctl_frame = ttk.Frame(plot_container)
        ctl_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(ctl_frame, text=self.T('importance_metric')).pack(side=tk.LEFT)
        self.imp_type_var = tk.StringVar(value="gain")
        ttk.Radiobutton(ctl_frame, text=self.T('gain_impact'),  variable=self.imp_type_var,
                        value="gain",   command=self.update_imp_plot).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(ctl_frame, text=self.T('weight_freq'), variable=self.imp_type_var,
                        value="weight", command=self.update_imp_plot).pack(side=tk.LEFT, padx=10)
        self.imp_plot_frame = ttk.Frame(plot_container)
        self.imp_plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        booster = self.model.get_booster()
        self.imp_data = {
            'weight': booster.get_score(importance_type='gain'),
            'gain':   booster.get_score(importance_type='weight'),
        }
        self.update_imp_plot()

    def update_imp_plot(self):
        for widget in self.imp_plot_frame.winfo_children():
            widget.destroy()
        imp_type = self.imp_type_var.get()
        scores   = self.imp_data.get(imp_type, {})
        if not scores:
            ttk.Label(self.imp_plot_frame, text=self.T('no_imp_data')).pack()
            return
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
        feats = [x[0] for x in sorted_items][::-1]
        vals  = [x[1] for x in sorted_items][::-1]
        suffix = self.T('gain_suffix') if imp_type == 'gain' else self.T('weight_suffix')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feats, vals, color='#4c72b0' if imp_type == 'gain' else '#55a868')
        ax.set_title(self.T('feat_imp_title', suffix))
        ax.set_xlabel(self.T('value_axis'))
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.imp_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Sim tab ───────────────────────────────────────────────────────────────

    def setup_sim_tab(self):
        # ── 可滚动左侧控制面板 ────────────────────────────────────────────────
        left_outer = ttk.Frame(self.tab_sim, width=410)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
        left_outer.pack_propagate(False)

        _scroll_cv = tk.Canvas(left_outer, highlightthickness=0)
        _scrollbar = ttk.Scrollbar(left_outer, orient='vertical', command=_scroll_cv.yview)
        _scroll_cv.configure(yscrollcommand=_scrollbar.set)
        _scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        _scroll_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(_scroll_cv)
        _win_id = _scroll_cv.create_window((0, 0), window=left_panel, anchor='nw')

        left_panel.bind('<Configure>',
                        lambda e: _scroll_cv.configure(scrollregion=_scroll_cv.bbox('all')))
        _scroll_cv.bind('<Configure>',
                        lambda e: _scroll_cv.itemconfig(_win_id, width=e.width))
        # 鼠标滚轮绑定（仅在鼠标进入左侧面板时激活，避免影响右侧图表）
        left_outer.bind('<Enter>',
                        lambda e: _scroll_cv.bind_all('<MouseWheel>',
                            lambda ev: _scroll_cv.yview_scroll(int(-1*(ev.delta/120)), 'units')))
        left_outer.bind('<Leave>', lambda e: _scroll_cv.unbind_all('<MouseWheel>'))

        right_panel = ttk.Frame(self.tab_sim)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        col_labels = get_col_labels(self.lang)

        # 1. Product selection
        prod_frame = ttk.LabelFrame(left_panel, text=self.T('select_product'))
        prod_frame.pack(fill=tk.X, pady=5)
        self.sim_model_var = tk.StringVar()
        self.sim_combo = ttk.Combobox(prod_frame, textvariable=self.sim_model_var)
        self.sim_combo.pack(fill=tk.X, padx=5, pady=5)
        self.sim_combo.bind("<<ComboboxSelected>>", self.load_product_features)
        self.update_product_dropdown()
        self.info_label = ttk.Label(prod_frame, text=self.T('select_prod_hint'),
                                    foreground="gray", wraplength=300)
        self.info_label.pack(fill=tk.X, padx=5, pady=5)

        date_frame = ttk.Frame(prod_frame)
        date_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(date_frame, text=self.T('launch_date')).pack(side=tk.LEFT)
        self.date_entry = ttk.Entry(date_frame)
        self.date_entry.insert(0, pd.Timestamp.now().strftime('%Y-%m-%d'))
        self.date_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 2. Config input
        input_frame = ttk.LabelFrame(left_panel, text=self.T('modify_config'))
        input_frame.pack(fill=tk.X, pady=5)
        self.entries = {}
        for i, feat in enumerate(EDITABLE_COLS):
            ttk.Label(input_frame, text=col_labels.get(feat, feat)).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = ttk.Entry(input_frame)
            entry.insert(0, "0")
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries[feat] = entry

        # 3. Bass model
        bass_frame = ttk.LabelFrame(left_panel, text=self.T('bass_section'))
        bass_frame.pack(fill=tk.X, pady=5)
        ttk.Label(bass_frame, text=self.T('market_potential')).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.bass_m_entry = ttk.Entry(bass_frame)
        self.bass_m_entry.insert(0, "100000")
        self.bass_m_entry.grid(row=0, column=1, padx=5, pady=2)
        avg_p_ref = f"{self.bass_engine.avg_p:.4f}" if hasattr(self, 'bass_engine') else "0.05"
        ttk.Label(bass_frame, text=self.T('innovation_coeff')).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.bass_p_entry = ttk.Entry(bass_frame)
        self.bass_p_entry.insert(0, avg_p_ref)
        self.bass_p_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(bass_frame, text=self.T('avg_ref', avg_p_ref),
                  foreground="gray").grid(row=1, column=2, sticky=tk.W)
        ref_text = self.T('hist_params_ref')
        if hasattr(self, 'bass_engine') and self.bass_engine.params:
            for series, vals in self.bass_engine.params.items():
                p_val, q_val, m_val = vals
                ref_text += f"• {series}: p={p_val:.4f}, m={m_val/1000:.1f}k\n"
        else:
            ref_text += self.T('no_hist_data')
        ttk.Label(bass_frame, text=ref_text, foreground="#555",
                  font=("Helvetica", 8), justify=tk.LEFT).grid(
            row=3, column=0, columnspan=3, pady=5, sticky=tk.W)
        ttk.Label(bass_frame, text=self.T('bass_note'),
                  foreground="gray", font=("", 8)).grid(row=4, column=0, columnspan=3)

        # 4. Forecast settings
        sim_set_frame = ttk.LabelFrame(left_panel, text=self.T('forecast_settings'))
        sim_set_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sim_set_frame, text=self.T('duration_label')).pack(anchor=tk.W, padx=5)
        dur_labels = get_duration_labels(self.lang)
        self.duration_var   = tk.StringVar(value=dur_labels[0])
        self.duration_combo = ttk.Combobox(sim_set_frame, textvariable=self.duration_var,
                                           values=dur_labels, state="readonly")
        self.duration_combo.pack(fill=tk.X, padx=5, pady=5)
        self.duration_combo.current(0)
        btn_frame = ttk.Frame(sim_set_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text=self.T('save_preset'),
                   command=self.save_preset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_frame, text=self.T('delete_preset'),
                   command=self.delete_preset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.run_btn = ttk.Button(sim_set_frame, text=self.T('run_simulation'),
                                  command=self.run_simulation_series)
        self.run_btn.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(sim_set_frame, text=self.T('export_json'),
                   command=self.export_json).pack(fill=tk.X, padx=5, pady=5)

        # Right panel
        self.sim_plot_frame = ttk.Frame(right_panel)
        self.sim_plot_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.sim_plot_frame, text=self.T('sim_placeholder'),
                  font=("Helvetica", 14)).pack(pady=50)

        if self.sim_combo['values']:
            self.sim_combo.current(0)
            self.load_product_features(None)

    def load_product_features(self, event):
        selected_model = self.sim_model_var.get()
        preset_data = next((p for p in self.user_presets if p['name'] == selected_model), None)

        if selected_model == "== NEW PRODUCT LAUNCH ==" or preset_data:
            self.bass_m_entry.config(state='normal')
            self.bass_p_entry.config(state='normal')
            try:
                launch_dt = pd.to_datetime(self.date_entry.get()).normalize()
            except Exception:
                launch_dt = pd.Timestamp.now().normalize()
                self.date_entry.delete(0, tk.END)
                self.date_entry.insert(0, launch_dt.strftime('%Y-%m-%d'))
            self.current_context = {
                'launch_date':    launch_dt,
                'last_date':      launch_dt - timedelta(days=1),
                'last_7d_sales':  [0.0] * 7,
                'original_specs': self.median_specs.copy(),
                'is_new_launch':  True,
            }
            if preset_data:
                self.info_label.config(
                    text=self.T('preset_info', selected_model, launch_dt.date()))
                specs = preset_data.get('specs', {})
                for feat in EDITABLE_COLS:
                    if feat in self.entries:
                        self.entries[feat].delete(0, tk.END)
                        self.entries[feat].insert(0, str(specs.get(feat, 0)))
                self.bass_m_entry.delete(0, tk.END)
                self.bass_m_entry.insert(0, str(preset_data.get('bass_m', 100000)))
                self.bass_p_entry.delete(0, tk.END)
                self.bass_p_entry.insert(0, str(preset_data.get('bass_p', 0.05)))
            else:
                self.info_label.config(
                    text=self.T('new_launch_info', launch_dt.date()))
                for feat in EDITABLE_COLS:
                    if feat in self.entries:
                        self.entries[feat].delete(0, tk.END)
                        self.entries[feat].insert(0, str(self.median_specs.get(feat, 0)))
                self.bass_m_entry.delete(0, tk.END)
                self.bass_m_entry.insert(0, "100000")
                self.bass_p_entry.delete(0, tk.END)
                self.bass_p_entry.insert(0, str(round(self.bass_engine.avg_p, 4)))
            return

        self.bass_m_entry.config(state='disabled')
        self.bass_p_entry.config(state='disabled')
        subset = self.full_df[self.full_df['model_key'] == selected_model]
        if subset.empty:
            messagebox.showwarning(self.T('warning'), self.T('err_no_data', selected_model))
            return
        latest_row  = subset.iloc[-1]
        real_launch = subset['date'].min()
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, real_launch.strftime('%Y-%m-%d'))
        self.current_context = {
            'launch_date':    real_launch,
            'last_date':      subset['date'].max(),
            'last_7d_sales':  subset['daily_sales'].tail(7).tolist(),
            'original_specs': latest_row[EDITABLE_COLS].to_dict(),
            'is_new_launch':  False,
        }
        self.info_label.config(
            text=self.T('loaded_info', selected_model,
                        self.current_context['last_date'].date(), real_launch.date()))
        for feat in EDITABLE_COLS:
            if feat in self.entries:
                self.entries[feat].delete(0, tk.END)
                self.entries[feat].insert(0, str(latest_row.get(feat, 0)))

    def run_simulation_series(self):
        if not hasattr(self, 'current_context'):
            messagebox.showwarning(self.T('warning'), self.T('err_no_product'))
            return

        # ── 限速检查 ──────────────────────────────────────────────────────────
        allowed, reason = self._sim_rl.check()
        if not allowed:
            remaining = self._sim_rl.remaining()
            messagebox.showwarning(
                self.T('warning'),
                f"{reason}\n（本窗口期内剩余可用次数：{remaining}）",
            )
            return

        # 显示加载状态，防止重复点击
        if hasattr(self, 'run_btn'):
            self.run_btn.config(state='disabled', text=self.T('running'))
            self.root.update_idletasks()

        try:
            self._run_simulation_inner()
            self._sim_rl.record()   # 成功执行后才记录
        finally:
            if hasattr(self, 'run_btn'):
                self.run_btn.config(state='normal', text=self.T('run_simulation'))

    # ── M3: 拆分后的三个私有辅助方法 ─────────────────────────────────────────

    _SPEC_BOUNDS = {
        'current_price':   (1,       9_999_999),
        'battery_mah':     (500,     30_000),
        'ram_gb':          (1,       256),
        'storage_gb':      (8,       4_096),
        'refresh_rate_hz': (1,       500),
        'main_camera_mp':  (1,       500),
        'charging_w':      (1,       500),
        'screen_res':      (100,     16_000),
        'ip_rating':       (0,       99),
    }

    def _parse_sim_inputs(self):
        """解析并校验模拟所需的用户输入。
        返回 (static_feats, bass_params, end_day_offset, agg_mode)，校验失败返回 None。"""
        dur_idx = self.duration_combo.current()
        if dur_idx < 0:
            dur_idx = 0
        _, end_day_offset, agg_mode = DURATION_OPTIONS[dur_idx]

        try:
            static_feats = {f: float(self.entries[f].get()) for f in EDITABLE_COLS}
            m_potential  = float(self.bass_m_entry.get())
            p_user       = float(self.bass_p_entry.get())
        except Exception:
            messagebox.showerror("Error", self.T('err_invalid_input'))
            return None

        bound_errors = []
        for col, (lo, hi) in self._SPEC_BOUNDS.items():
            v = static_feats.get(col)
            if v is not None and not (lo <= v <= hi):
                bound_errors.append(f'{col}: 有效范围 {lo}–{hi}，当前值 {v}')
        if not (100 <= m_potential <= 10_000_000):
            bound_errors.append(
                f'Bass m (Market Potential): 有效范围 100–10,000,000，当前值 {m_potential}')
        if not (0.001 <= p_user <= 1.0):
            bound_errors.append(
                f'Bass p (Innovation Coeff): 有效范围 0.001–1.0，当前值 {p_user}')
        if bound_errors:
            messagebox.showerror(
                "Error", self.T('err_invalid_input') + '\n\n' + '\n'.join(bound_errors))
            return None

        return static_feats, {'m': m_potential, 'p': p_user}, end_day_offset, agg_mode

    def _save_sim_to_db(self, static_feats, bass_params, end_day_offset, agg_mode, plot_data):
        """将模拟结果写入数据库（失败时静默记录日志，不影响界面）。"""
        try:
            from db import init_db, save_simulation_result
            init_db()
            save_simulation_result(
                {
                    'product_name':  self.sim_model_var.get(),
                    'launch_date':   str(self.current_context.get('launch_date', ''))[:10],
                    'duration_days': end_day_offset,
                    'agg_mode':      agg_mode,
                    'user_specs':    static_feats,
                    'bass_params':   bass_params,
                },
                plot_data, source='gui',
            )
        except Exception:
            logger.warning("Failed to save simulation result to DB", exc_info=True)

    def _render_sim_chart(self, plot_data, x_col, y_col, title, agg_mode):
        """在模拟图表区域绘制 matplotlib 图表。"""
        for widget in self.sim_plot_frame.winfo_children():
            widget.destroy()
        plt.close('all')

        is_monthly = 'month' in agg_mode
        fig, ax = plt.subplots(figsize=(8, 5))
        if is_monthly:
            bars = ax.bar(plot_data[x_col], plot_data[y_col],
                          color='skyblue', edgecolor='black')
            for bar, val in zip(bars, plot_data[y_col]):
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.plot(plot_data[x_col], plot_data[y_col],
                    marker='o', linestyle='-', color='green')
            step = 1 if len(plot_data) <= 30 else 5
            for i, (x, y) in enumerate(zip(plot_data[x_col], plot_data[y_col])):
                if i % step == 0:
                    ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                                xytext=(0, 10), ha='center', fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(self.T('time_axis'))
        ax.set_ylabel(self.T('pred_sales_vol'))
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.sim_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run_simulation_inner(self):
        """模拟入口：协调输入解析、执行模拟、持久化、图表渲染。"""
        # 新产品模式：将日期选择器的值注入 context
        if self.current_context.get('is_new_launch', False):
            try:
                new_launch_dt = pd.to_datetime(self.date_entry.get()).normalize()
                self.current_context['launch_date'] = new_launch_dt
                self.current_context['last_date']   = new_launch_dt - timedelta(days=1)
            except Exception:
                pass

        parsed = self._parse_sim_inputs()
        if parsed is None:
            return
        static_feats, bass_params, end_day_offset, agg_mode = parsed
        self.current_context['model_key'] = self.sim_model_var.get()

        try:
            sim_results = self.simulator.run_simulation(
                context=self.current_context,
                user_specs=static_feats,
                bass_params=bass_params,
                duration_days=end_day_offset,
            )
            plot_data, x_col, y_col = self.simulator.aggregate_results(
                sim_results, agg_mode, self.current_context['launch_date']
            )
            is_monthly = 'month' in agg_mode
            title = (self.T('forecast_mo_title', static_feats['current_price']) if is_monthly
                     else self.T('forecast_day_title', end_day_offset, static_feats['current_price']))

            self.last_sim_data = plot_data
            self._save_sim_to_db(static_feats, bass_params, end_day_offset, agg_mode, plot_data)
            self._render_sim_chart(plot_data, x_col, y_col, title, agg_mode)

        except Exception as e:
            logger.exception("Simulation failed")
            messagebox.showerror(self.T('err_sim_title'), str(e))

    # ── Data import tab ───────────────────────────────────────────────────────

    def setup_ingest_tab(self):
        # ── 未登录时显示锁定提示 ──────────────────────────────────────────────
        if not self.admin_logged_in:
            lock_frame = ttk.Frame(self.tab_ingest, padding=40)
            lock_frame.place(relx=0.5, rely=0.5, anchor='center')
            ttk.Label(lock_frame, text=self.T('ingest_locked_title'),
                      font=('', 15, 'bold')).pack(pady=(0, 12))
            ttk.Label(lock_frame, text=self.T('ingest_locked_hint'),
                      foreground='#666666', justify='center').pack(pady=(0, 18))
            ttk.Button(lock_frame, text=self.T('ingest_locked_btn'),
                       command=lambda: self.notebook.select(self.tab_admin)
                       ).pack()
            return

        root_frame = ttk.Frame(self.tab_ingest, padding=20)
        root_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root_frame, text=self.T('ingest_title'),
                  font=('', 14, 'bold')).pack(anchor='w')
        ttk.Label(root_frame, text=self.T('ingest_hint'),
                  foreground='#666666').pack(anchor='w', pady=(2, 12))

        file_row = ttk.Frame(root_frame)
        file_row.pack(fill=tk.X, pady=(0, 12))
        file_row.columnconfigure(0, weight=1)
        file_row.columnconfigure(1, weight=1)

        self._specs_drop_label = self._make_drop_zone(
            file_row, row=0, col=0,
            title=self.T('specs_file_title'),
            hint=self.T('specs_file_hint'),
            path_var=self._specs_path,
            is_sales=False,
        )
        self._sales_drop_label = self._make_drop_zone(
            file_row, row=0, col=1,
            title=self.T('sales_file_title'),
            hint=self.T('sales_file_hint'),
            path_var=self._sales_path,
            is_sales=True,
        )

        btn_frame = ttk.Frame(root_frame)
        btn_frame.pack(pady=8)
        self._ingest_btn = ttk.Button(
            btn_frame, text=self.T('start_import'),
            command=self._start_ingest, width=20,
        )
        self._ingest_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(
            btn_frame, text=self.T('clear_log'),
            command=lambda: self._log_text.delete('1.0', tk.END),
        ).pack(side=tk.LEFT, padx=6)

        ttk.Label(root_frame, text=self.T('import_log'),
                  font=('', 10, 'bold')).pack(anchor='w')
        log_frame = ttk.Frame(root_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self._log_text = tk.Text(
            log_frame, height=12, state='normal',
            bg='#1E1E1E', fg='#D4D4D4',
            font=('Consolas', 9), relief='flat',
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # 规格文件支持的扩展名（需要 Excel Sheet 结构）
    _SPECS_EXTS = ('.xlsx', '.xls', '.xlsm', '.xlsb', '.ods', '.odf', '.odt')
    # 销量文件额外支持 CSV
    _SALES_EXTS = _SPECS_EXTS + ('.csv',)

    def _make_drop_zone(self, parent, row, col, title, hint, path_var, is_sales=False):
        wrapper = ttk.LabelFrame(parent, text=title, padding=10)
        wrapper.grid(row=row, column=col, padx=8, sticky='nsew')

        zone = tk.Frame(wrapper, bg='#EBF5FB', height=80, relief='groove', bd=2, cursor='hand2')
        zone.pack(fill=tk.X, pady=(0, 8))
        zone.pack_propagate(False)

        file_label = tk.Label(
            zone,
            text=self.T('drop_zone_text'),
            bg='#EBF5FB', fg='#1A5276',
            font=('', 10),
        )
        file_label.pack(expand=True)

        for widget in (zone, file_label):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind('<<Drop>>',
                            lambda e, pv=path_var, lb=file_label, s=is_sales:
                                self._handle_drop(e, pv, lb, s))

        ttk.Label(wrapper, text=hint, foreground='#888888', font=('', 8)).pack(anchor='w')
        ttk.Button(
            wrapper, text=self.T('browse_file'),
            command=lambda pv=path_var, lb=file_label, s=is_sales:
                self._browse_file(pv, lb, s),
        ).pack(anchor='w', pady=(4, 0))

        return file_label

    @staticmethod
    def _resolve_and_check(raw_path: str):
        """Resolve path, verify it exists and is a regular file.
        Returns the resolved path string, or raises ValueError with a safe message."""
        try:
            p = pathlib.Path(raw_path).resolve(strict=True)
        except (OSError, ValueError):
            raise ValueError(f'路径无效或文件不存在：{os.path.basename(raw_path)}')
        if not p.is_file():
            raise ValueError(f'所选路径不是文件：{p.name}')
        return str(p)

    def _handle_drop(self, event, path_var: tk.StringVar, label: tk.Label, is_sales: bool):
        raw = _parse_dnd_path(event.data)
        allowed = self._SALES_EXTS if is_sales else self._SPECS_EXTS
        if not raw.lower().endswith(allowed):
            ext_str = '  '.join(allowed)
            messagebox.showwarning(self.T('file_type_err'),
                                   self.T('file_type_msg') + f'\n({ext_str})')
            return
        try:
            path = self._resolve_and_check(raw)
        except ValueError as ve:
            messagebox.showwarning(self.T('file_type_err'), str(ve))
            return
        path_var.set(path)
        label.config(text=self.T('file_ok_fmt', os.path.basename(path)),
                     fg='#1D8348', bg='#EAFAF1')

    def _browse_file(self, path_var: tk.StringVar, label: tk.Label, is_sales: bool):
        if is_sales:
            title    = self.T('select_sales_dlg')
            ftype    = self.T('sales_filter')
            pattern  = "*.xlsx *.xls *.xlsm *.xlsb *.ods *.csv"
        else:
            title    = self.T('select_specs_dlg')
            ftype    = self.T('specs_filter')
            pattern  = "*.xlsx *.xls *.xlsm *.xlsb *.ods"
        raw = filedialog.askopenfilename(
            title=title,
            filetypes=[
                (ftype, pattern),
                (self.T('all_files_filter'), "*.*"),
            ],
        )
        if raw:
            try:
                path = self._resolve_and_check(raw)
            except ValueError as ve:
                messagebox.showwarning(self.T('file_type_err'), str(ve))
                return
            path_var.set(path)
            label.config(text=self.T('file_ok_fmt', os.path.basename(path)),
                         fg='#1D8348', bg='#EAFAF1')

    def _start_ingest(self):
        specs = self._specs_path.get()
        sales = self._sales_path.get()
        if not specs or not sales:
            messagebox.showwarning(self.T('no_file_title'), self.T('no_file_msg'))
            return
        self._ingest_btn.config(state='disabled', text=self.T('ingest_running'))
        self._log_text.delete('1.0', tk.END)
        t = threading.Thread(target=self._ingest_worker, args=(specs, sales), daemon=True)
        t.start()
        self.root.after(100, self._poll_log)

    def _ingest_worker(self, specs_path: str, sales_path: str):
        with _STDOUT_LOCK:
            old_stdout = sys.stdout
            sys.stdout = _QueueWriter(self._log_queue)
        try:
            # H3: validate file signatures before processing
            for path, label in [(specs_path, 'Specs'), (sales_path, 'Sales')]:
                if path:
                    suffix = os.path.splitext(path)[1].lower()
                    if not _validate_file_magic(path, suffix):
                        self._log_queue.put(
                            f'[Error] {label} 文件签名不匹配："{os.path.basename(path)}" '
                            f'不是有效的 {suffix} 文件（可能已损坏或被伪装）。')
                        return
            from data_processor_v2 import ingest_all
            ingest_all(specs_path, sales_path)
        except Exception as exc:
            self._log_queue.put(self.T('ingest_err_fmt', exc))
        finally:
            with _STDOUT_LOCK:
                sys.stdout = old_stdout
            self._log_queue.put('__DONE__')

    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if msg == '__DONE__':
                    self._ingest_btn.config(state='normal', text=self.T('start_import'))
                    self._log_append(self.T('ingest_done'))
                    return
                self._log_append(msg)
        except _queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _log_append(self, text: str):
        self._log_text.insert(tk.END, text + '\n')
        self._log_text.see(tk.END)

    # ── Admin tab ─────────────────────────────────────────────────────────────

    def setup_admin_tab(self):
        """构建管理员标签页：未登录时显示登录表单，已登录时显示管理面板。"""
        for w in self.tab_admin.winfo_children():
            w.destroy()

        if not self.admin_logged_in:
            self._build_admin_login()
        else:
            self._build_admin_panel()

    def _build_admin_login(self):
        """登录表单（或首次运行密码设置表单）。"""
        outer = ttk.Frame(self.tab_admin)
        outer.place(relx=0.5, rely=0.4, anchor='center')

        stored = _load_admin_hash()
        first_run = (stored is None)

        title = '初始密码设置 / Initial Setup' if first_run else self.T('admin_login_title')
        ttk.Label(outer, text=title,
                  font=('', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 16))

        if first_run:
            ttk.Label(outer, text='未检测到管理员密码，请设置初始密码：',
                      foreground='#8B4513').grid(
                row=1, column=0, columnspan=2, pady=(0, 8))

        ttk.Label(outer, text=self.T('admin_pw_label')).grid(
            row=2, column=0, sticky='e', padx=(0, 8), pady=6)
        pw_var = tk.StringVar()
        pw_entry = ttk.Entry(outer, textvariable=pw_var, show='*', width=22)
        pw_entry.grid(row=2, column=1, pady=6)
        pw_entry.focus_set()

        cf_var = tk.StringVar()
        cf_entry = None
        if first_run:
            ttk.Label(outer, text='确认密码 / Confirm:').grid(
                row=3, column=0, sticky='e', padx=(0, 8), pady=6)
            cf_entry = ttk.Entry(outer, textvariable=cf_var, show='*', width=22)
            cf_entry.grid(row=3, column=1, pady=6)

        msg_var = tk.StringVar()
        ttk.Label(outer, textvariable=msg_var, foreground='red').grid(
            row=4, column=0, columnspan=2, pady=(0, 8))

        def _do_login(e=None):
            pw = pw_var.get()
            if not pw:
                msg_var.set('密码不能为空。')
                return
            if first_run:
                if pw != cf_var.get():
                    msg_var.set(self.T('admin_pw_mismatch'))
                    pw_var.set(''); cf_var.set('')
                    return
                _save_admin_hash(_hash_pw(pw))
                self.admin_logged_in = True
                self.setup_admin_tab()
                for w in self.tab_ingest.winfo_children():
                    w.destroy()
                self.setup_ingest_tab()
            else:
                if _verify_pw(pw, stored):
                    # 如果是旧 SHA-256 格式，自动升级到 PBKDF2
                    if ':' not in stored:
                        _save_admin_hash(_hash_pw(pw))
                    self.admin_logged_in = True
                    self.setup_admin_tab()
                    for w in self.tab_ingest.winfo_children():
                        w.destroy()
                    self.setup_ingest_tab()
                else:
                    msg_var.set(self.T('admin_wrong_pw'))
                    pw_var.set('')

        btn_text = '设置密码 / Set Password' if first_run else self.T('admin_login_btn')
        ttk.Button(outer, text=btn_text,
                   command=_do_login).grid(row=5, column=0, columnspan=2, pady=4, ipadx=10)
        pw_entry.bind('<Return>', _do_login)
        if cf_entry:
            cf_entry.bind('<Return>', _do_login)

    def _build_admin_panel(self):
        """已登录后的管理面板（带滚动）。"""
        # 滚动容器
        cv = tk.Canvas(self.tab_admin, highlightthickness=0)
        sb = ttk.Scrollbar(self.tab_admin, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        panel = ttk.Frame(cv, padding=20)
        win_id = cv.create_window((0, 0), window=panel, anchor='nw')
        panel.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        cv.bind('<Configure>', lambda e: cv.itemconfig(win_id, width=e.width))
        cv.bind_all('<MouseWheel>', lambda e: cv.yview_scroll(int(-1*(e.delta/120)), 'units'))

        # ── 顶栏：登录状态 + 退出 ─────────────────────────────────────────
        top = ttk.Frame(panel)
        top.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(top, text=self.T('admin_logged_in_as'),
                  font=('', 11, 'bold'), foreground='#1D8348').pack(side=tk.LEFT)

        def _logout():
            self.admin_logged_in = False
            # 刷新管理员 tab 和数据导入 tab（重新显示锁定画面）
            self.setup_admin_tab()
            for w in self.tab_ingest.winfo_children():
                w.destroy()
            self.setup_ingest_tab()

        ttk.Button(top, text=self.T('admin_logout_btn'),
                   command=_logout).pack(side=tk.RIGHT)
        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=6)

        # ── 修改密码 ───────────────────────────────────────────────────────
        pw_sec = ttk.LabelFrame(panel, text=self.T('admin_change_pw_sec'), padding=10)
        pw_sec.pack(fill=tk.X, pady=8)
        ttk.Label(pw_sec, text=self.T('admin_new_pw')).grid(
            row=0, column=0, sticky='e', padx=(0, 8), pady=4)
        new_pw_var = tk.StringVar()
        ttk.Entry(pw_sec, textvariable=new_pw_var, show='*', width=20).grid(
            row=0, column=1, pady=4)
        ttk.Label(pw_sec, text=self.T('admin_confirm_pw')).grid(
            row=1, column=0, sticky='e', padx=(0, 8), pady=4)
        cf_pw_var = tk.StringVar()
        ttk.Entry(pw_sec, textvariable=cf_pw_var, show='*', width=20).grid(
            row=1, column=1, pady=4)
        pw_msg = tk.StringVar()
        ttk.Label(pw_sec, textvariable=pw_msg, foreground='green').grid(
            row=2, column=0, columnspan=3, pady=(0, 4))

        def _change_pw():
            np_, cp_ = new_pw_var.get(), cf_pw_var.get()
            if not np_:
                pw_msg.set(self.T('admin_pw_empty')); return
            if np_ != cp_:
                pw_msg.set(self.T('admin_pw_mismatch')); return
            _save_admin_hash(_hash_pw(np_))
            new_pw_var.set(''); cf_pw_var.set('')
            pw_msg.set(self.T('admin_pw_changed'))

        ttk.Button(pw_sec, text=self.T('admin_change_pw_btn'),
                   command=_change_pw).grid(row=0, column=2, rowspan=2, padx=10)

        # ── 训练记录 ───────────────────────────────────────────────────────
        self._build_admin_history_section(
            panel,
            section_title=self.T('admin_train_sec'),
            fetch_fn=self._admin_fetch_training,
            del_fn=self._admin_del_training,
            clear_fn=self._admin_clear_training,
            attr='_admin_train_tree',
        )

        # ── 模拟历史 ───────────────────────────────────────────────────────
        self._build_admin_history_section(
            panel,
            section_title=self.T('admin_sim_sec'),
            fetch_fn=self._admin_fetch_simulation,
            del_fn=self._admin_del_simulation,
            clear_fn=self._admin_clear_simulation,
            attr='_admin_sim_tree',
        )

        # ── 重新训练 ───────────────────────────────────────────────────────
        rt_sec = ttk.LabelFrame(panel, text=self.T('admin_retrain_sec'), padding=10)
        rt_sec.pack(fill=tk.X, pady=8)
        rt_log = tk.Text(rt_sec, height=8, state='disabled',
                         bg='#1E1E1E', fg='#D4D4D4',
                         font=('Consolas', 9), relief='flat')
        rt_log.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        def _retrain():
            rt_btn.config(state='disabled', text=self.T('admin_retrain_running'))
            rt_log.config(state='normal'); rt_log.delete('1.0', tk.END)
            rt_log.config(state='disabled')

            def _run():
                try:
                    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'train_daily.py')
                    proc = subprocess.Popen(
                        [sys.executable, script],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, cwd=os.path.dirname(os.path.dirname(script)),
                    )
                    for line in proc.stdout:
                        rt_log.config(state='normal')
                        rt_log.insert(tk.END, line)
                        rt_log.see(tk.END)
                        rt_log.config(state='disabled')
                    proc.wait()
                    rt_log.config(state='normal')
                    rt_log.insert(tk.END, '\n' + self.T('admin_retrain_done') + '\n')
                    rt_log.see(tk.END)
                    rt_log.config(state='disabled')
                except Exception as e:
                    rt_log.config(state='normal')
                    rt_log.insert(tk.END, self.T('admin_retrain_err', e) + '\n')
                    rt_log.config(state='disabled')
                finally:
                    rt_btn.config(state='normal', text=self.T('admin_retrain_btn'))

            threading.Thread(target=_run, daemon=True).start()

        rt_btn = ttk.Button(rt_sec, text=self.T('admin_retrain_btn'), command=_retrain)
        rt_btn.pack(anchor='w', pady=(0, 6))

    def _build_admin_history_section(self, parent, section_title,
                                     fetch_fn, del_fn, clear_fn, attr):
        """通用历史记录区块：表格 + 刷新/删除/清空按钮。"""
        sec = ttk.LabelFrame(parent, text=section_title, padding=10)
        sec.pack(fill=tk.X, pady=8)

        # 按钮行
        btn_row = ttk.Frame(sec)
        btn_row.pack(fill=tk.X, pady=(0, 6))

        # Treeview
        tree_frame = ttk.Frame(sec)
        tree_frame.pack(fill=tk.BOTH)
        tree = ttk.Treeview(tree_frame, height=6, selectmode='browse')
        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        setattr(self, attr, tree)

        def _refresh():
            fetch_fn(tree)

        def _del_sel():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning(self.T('warning'), self.T('admin_del_sel_warn'))
                return
            row_id = tree.item(sel[0])['values'][0]
            del_fn(int(row_id))
            _refresh()
            messagebox.showinfo(self.T('success'), self.T('admin_deleted_fmt', 1))

        def _clear_all():
            msg = (self.T('admin_clear_train_msg') if 'train' in attr
                   else self.T('admin_clear_sim_msg'))
            if not messagebox.askyesno(self.T('admin_confirm_clear'), msg):
                return
            n = clear_fn()
            _refresh()
            messagebox.showinfo(self.T('success'), self.T('admin_deleted_fmt', n))

        ttk.Button(btn_row, text=self.T('admin_refresh_btn'),
                   command=_refresh).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text=self.T('admin_del_sel_btn'),
                   command=_del_sel).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text=self.T('admin_clear_all_btn'),
                   command=_clear_all).pack(side=tk.LEFT)

        _refresh()  # 初始加载

    # ── Admin data helpers ────────────────────────────────────────────────────

    def _admin_fetch_training(self, tree):
        try:
            from db import get_training_history
            df = get_training_history(limit=50)
        except Exception:
            df = pd.DataFrame()
        tree.delete(*tree.get_children())
        cols = list(df.columns) if not df.empty else ['id', 'run_timestamp', 'wmape', 'mae', 'rmse']
        tree['columns'] = cols
        tree['show'] = 'headings'
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(80, len(c) * 9), anchor='center')
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=list(row))

    def _admin_fetch_simulation(self, tree):
        try:
            from db import get_simulation_history
            df = get_simulation_history(limit=50)
        except Exception:
            df = pd.DataFrame()
        tree.delete(*tree.get_children())
        cols = list(df.columns) if not df.empty else ['id', 'run_timestamp', 'product_name']
        tree['columns'] = cols
        tree['show'] = 'headings'
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(80, len(c) * 9), anchor='center')
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=list(row))

    def _admin_del_training(self, run_id: int):
        try:
            from db import delete_training_run
            delete_training_run(run_id)
        except Exception:
            pass

    def _admin_del_simulation(self, sim_id: int):
        try:
            from db import delete_simulation_result
            delete_simulation_result(sim_id)
        except Exception:
            pass

    def _admin_clear_training(self) -> int:
        try:
            from db import delete_all_training_runs
            return delete_all_training_runs()
        except Exception:
            return 0

    def _admin_clear_simulation(self) -> int:
        try:
            from db import delete_all_simulation_results
            return delete_all_simulation_results()
        except Exception:
            return 0

    # ── Manual tab ────────────────────────────────────────────────────────────

    def setup_manual_tab(self):
        """Build the User Manual tab with a scrollable, formatted markdown viewer."""
        from tkinter.scrolledtext import ScrolledText

        frame = ttk.Frame(self.tab_manual, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        txt = ScrolledText(frame, wrap=tk.WORD, state='normal',
                           font=('', 10), padx=12, pady=8,
                           relief='flat', borderwidth=0)
        txt.pack(fill=tk.BOTH, expand=True)

        # Configure display tags
        txt.tag_configure('h1',     font=('', 15, 'bold'), spacing1=14, spacing3=6,
                          foreground='#1a1a2e')
        txt.tag_configure('h2',     font=('', 13, 'bold'), spacing1=10, spacing3=4,
                          foreground='#16213e')
        txt.tag_configure('h3',     font=('', 11, 'bold'), spacing1=8,  spacing3=2,
                          foreground='#0f3460')
        txt.tag_configure('code',   font=('Courier', 9),   background='#f4f4f4',
                          foreground='#333333')
        txt.tag_configure('table',  font=('Courier', 9),   foreground='#444444')
        txt.tag_configure('bullet', lmargin1=20, lmargin2=32)
        txt.tag_configure('hr',     foreground='#cccccc')
        txt.tag_configure('normal', font=('', 10))

        manual_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_manual.md')
        try:
            with open(manual_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.exception("Failed to load user manual")
            txt.insert(tk.END, f'Error loading manual: {e}')
            txt.config(state='disabled')
            return

        # Split Chinese / English sections
        en_marker = '# Part 2: English User Guide'
        if en_marker in content:
            zh_part, en_part = content.split(en_marker, 1)
            en_part = en_marker + en_part
        else:
            zh_part = en_part = content

        text = zh_part if self.lang == 'zh' else en_part

        in_code = False
        for line in text.split('\n'):
            stripped = line.rstrip()
            if stripped.startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                txt.insert(tk.END, stripped + '\n', 'code')
            elif stripped.startswith('# ') and not stripped.startswith('##'):
                txt.insert(tk.END, stripped[2:] + '\n', 'h1')
            elif stripped.startswith('## ') and not stripped.startswith('###'):
                txt.insert(tk.END, stripped[3:] + '\n', 'h2')
            elif stripped.startswith('### '):
                txt.insert(tk.END, stripped[4:] + '\n', 'h3')
            elif stripped.startswith(('- ', '* ', '+ ')):
                txt.insert(tk.END, '  •  ' + stripped[2:] + '\n', 'bullet')
            elif stripped.startswith('|'):
                txt.insert(tk.END, stripped + '\n', 'table')
            elif stripped in ('---', '---\n'):
                txt.insert(tk.END, '─' * 80 + '\n', 'hr')
            else:
                txt.insert(tk.END, stripped + '\n', 'normal')

        txt.config(state='disabled')

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gui_app.log', encoding='utf-8'),
        ]
    )
    root = TkinterDnD.Tk()
    app = SalesPredictorApp(root)
    root.mainloop()
