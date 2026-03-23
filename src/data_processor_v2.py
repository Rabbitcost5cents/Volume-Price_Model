import re
import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 通用辅助函数
# ---------------------------------------------------------------------------

def _normalize_name(name):
    """将型号名称归一化为小写字母数字串（用于 model_key 生成）。"""
    if not isinstance(name, str):
        name = str(name)
    name = name.lower().replace('series', '').replace('5g', '')
    name = re.sub(r'(\d+)g', r'\1', name)
    return re.sub(r'[^a-z0-9]', '', name)


# ---------------------------------------------------------------------------
# 文件读取辅助函数
# ---------------------------------------------------------------------------

def _detect_sheet(file_path, preferred=('512GB', '512', 'Specs', 'Sheet1')):
    """
    自动检测 Excel 文件中最合适的 Sheet。
    按 preferred 顺序查找（大小写不敏感），都找不到时返回第一个 Sheet。
    """
    try:
        xl = pd.ExcelFile(file_path)
        names = xl.sheet_names
        xl.close()
        # 精确匹配
        for p in preferred:
            if p in names:
                return p
        # 大小写不敏感匹配
        for p in preferred:
            for n in names:
                if p.lower() in n.lower():
                    return n
        return names[0]
    except Exception:
        return 0  # fallback: first sheet by index


def _read_excel_any(file_path: str, **kwargs) -> pd.DataFrame:
    """
    支持 .xlsx / .xls / .xlsm / .xlsb / .ods / .csv 的统一读取。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        # CSV 时忽略 Excel 专属参数
        csv_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ('sheet_name', 'engine')}
        return pd.read_csv(file_path, **csv_kwargs)
    return pd.read_excel(file_path, **kwargs)


def clean_specs(file_path):
    """
    读取并清洗规格表（支持 xlsx/xls/xlsm/ods）。
    自动检测包含规格数据的 Sheet（优先 '512GB'，找不到则用第一个 Sheet）。
    """
    try:
        sheet = _detect_sheet(file_path, preferred=('512GB', '512', 'Specs', 'Sheet1'))
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)
        
        # 转置：第 0 行变为列
        df_t = df.T
        df_t.columns = df_t.iloc[0]
        df_t = df_t[1:]
        df_t.columns = df_t.columns.str.strip().str.upper()
        
        cols_map = {
            'PRODUCT MODEL': 'model_raw',
            'ORIGINAL PRICE': 'price_base',
            'RAM': 'ram_gb',
            'ROM': 'storage_gb',
            'BATTERY': 'battery_mah',
            'SCREEN REFRESH RATE': 'refresh_rate_hz',
            'REAR CAMERA': 'main_camera_mp',
            'CHARGING TYPE': 'charging_w',
            'SCREEN RESOLUTION': 'screen_res',
            'IP RATING': 'ip_rating'
        }
        
        # D1: 校验必要列是否存在
        if 'PRODUCT MODEL' not in df_t.columns:
            logger.error(
                f"Required column 'PRODUCT MODEL' missing from specs sheet. "
                f"Available columns: {df_t.columns.tolist()}"
            )
            return pd.DataFrame()
        optional_missing = [c for c in cols_map if c != 'PRODUCT MODEL' and c not in df_t.columns]
        if optional_missing:
            logger.warning(f"Optional spec columns not found (will default to 0): {optional_missing}")

        existing_cols = [c for c in cols_map.keys() if c in df_t.columns]
        df_clean = df_t[existing_cols].copy()
        df_clean = df_clean.rename(columns=cols_map)

        def extract_num(val):
            if isinstance(val, (int, float)): return float(val)
            match = re.search(r'(\d+(\.\d+)?)', str(val))
            return float(match.group(1)) if match else 0.0

        numeric_cols = ['price_base', 'ram_gb', 'storage_gb', 'battery_mah', 
                        'refresh_rate_hz', 'main_camera_mp', 'charging_w']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(extract_num)
            else:
                df_clean[col] = 0.0
                
        def clean_res(val):
            # R1: 替换裸 except，记录解析失败原因
            if not isinstance(val, str): return 0
            try:
                parts = val.lower().split('x')
                if len(parts) < 2: parts = val.split('*')
                if len(parts) >= 2:
                    return extract_num(parts[0]) * extract_num(parts[1])
            except (ValueError, TypeError) as e:
                logger.debug(f"screen_res parse failed for {val!r}: {e}")
            return 0.0

        if 'screen_res' in df_clean.columns:
            df_clean['screen_res'] = df_clean['screen_res'].apply(clean_res)
        else:
            df_clean['screen_res'] = 0.0
            
        def clean_ip(val):
            if not isinstance(val, str): return 0
            val = val.upper().replace('IP', '').strip()
            match = re.search(r'(\d+)', val)
            return float(match.group(1)) if match else 0.0
            
        if 'ip_rating' in df_clean.columns:
            df_clean['ip_rating'] = df_clean['ip_rating'].apply(clean_ip)
        else:
            df_clean['ip_rating'] = 0.0
                
        df_clean['model_key'] = df_clean['model_raw'].apply(_normalize_name)
        
        return df_clean
    except Exception as e:
        logger.error("clean_specs failed", exc_info=True)
        return pd.DataFrame()

def process_daily_sales(file_path):
    """
    解析销量文件（xlsx / xls / xlsm / ods / csv 均可）。
    结构：标题行（型号、总生命周期销量、日期...），然后是数据行。
    标题行识别：第 0 列值为 'model'（大小写不敏感）。
    CSV 格式：第一行即为标题，第 0 列名须含 'model'（大小写不敏感）。
    """
    ext = os.path.splitext(file_path)[1].lower()

    # CSV：结构更规整，直接用列名读取
    if ext == '.csv':
        return _process_daily_sales_csv(file_path)

    df_raw = _read_excel_any(file_path, header=None)

    # 找到所有标题行（第 0 列值含 'model'，大小写不敏感）
    header_indices = df_raw.index[
        df_raw[0].astype(str).str.strip().str.lower() == 'model'
    ].tolist()
    
    if not header_indices:
        raise ValueError("未能找到任何包含 'Model' 的标题行")
        
    all_blocks = []
    
    for start_idx in header_indices:
        # 获取标题
        headers = df_raw.iloc[start_idx].values
        
        # 确定数据块的结束行 (end_idx)
        end_idx = start_idx + 1
        while end_idx < len(df_raw):
            val = str(df_raw.iloc[end_idx, 0]).strip()
            if val == 'Model' or val == 'nan' or val == '':
                break
            end_idx += 1
            
        if end_idx > start_idx + 1:
            block_data = df_raw.iloc[start_idx+1:end_idx].copy()
            block_data.columns = headers
            
            # 逆透视 (Melt) 处理
            # ID 变量：'Model', 'Total Lifetime Sales' (第 1 列)
            # 值变量：所有日期列 (从索引 2 开始)
            id_vars = [headers[0], headers[1]]
            value_vars = headers[2:]
            
            # 过滤有效的日期列（排除 NaN）
            valid_value_vars = [v for v in value_vars if pd.notna(v)]
            
            melted = block_data.melt(
                id_vars=id_vars,
                value_vars=valid_value_vars,
                var_name='date',
                value_name='daily_sales'
            )
            
            melted.rename(columns={headers[0]: 'model_raw'}, inplace=True)
            # 此文件中没有 'current_price'，稍后将使用规格中的基准价格
            
            all_blocks.append(melted)
            
    if not all_blocks:
        return pd.DataFrame()
        
    df_long = pd.concat(all_blocks, ignore_index=True)
    
    # 数据清洗
    df_long['daily_sales'] = pd.to_numeric(df_long['daily_sales'], errors='coerce').fillna(0)
    df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
    df_long = df_long.dropna(subset=['date'])

    # D2: 过滤负值销量（可能是退货记录，不应参与训练）
    neg_mask = df_long['daily_sales'] < 0
    if neg_mask.any():
        logger.warning(f"Filtered {neg_mask.sum()} rows with negative daily_sales values")
        df_long = df_long[~neg_mask]

    # D2: 过滤未来日期（数据录入错误）
    today = pd.Timestamp.now().normalize()
    future_mask = df_long['date'] > today
    if future_mask.any():
        logger.warning(
            f"Filtered {future_mask.sum()} rows with future dates "
            f"(latest: {df_long.loc[future_mask, 'date'].max().date()})"
        )
        df_long = df_long[~future_mask]

    df_long['model_key'] = df_long['model_raw'].apply(_normalize_name)

    return df_long

def _process_daily_sales_csv(file_path):
    """
    处理 CSV 格式的销量文件。
    期望列：第 0 列为型号名（列名含 'model'），第 1 列为总销量（可选），
    其余列为日期（YYYY-MM-DD 或 Excel 日期序号）。
    """
    try:
        df = pd.read_csv(file_path)
        # 找到型号列（第一个列名含 'model' 的列）
        model_col = next(
            (c for c in df.columns if 'model' in str(c).lower()), df.columns[0]
        )
        # 猜测日期列：除 model_col 和可能的 'total' 列外，尝试解析为日期
        non_date_cols = {model_col}
        for c in df.columns[1:3]:  # 前两列里找 total 列
            if 'total' in str(c).lower() or 'lifetime' in str(c).lower():
                non_date_cols.add(c)
        value_vars = [c for c in df.columns if c not in non_date_cols]
        melted = df.melt(id_vars=[model_col], value_vars=value_vars,
                         var_name='date', value_name='daily_sales')
        melted.rename(columns={model_col: 'model_raw'}, inplace=True)
        melted['daily_sales'] = pd.to_numeric(melted['daily_sales'], errors='coerce').fillna(0)
        melted['date'] = pd.to_datetime(melted['date'], errors='coerce')
        melted = melted.dropna(subset=['date'])

        # D2: 过滤负值和未来日期
        neg_mask = melted['daily_sales'] < 0
        if neg_mask.any():
            logger.warning(f"CSV: filtered {neg_mask.sum()} rows with negative daily_sales")
            melted = melted[~neg_mask]
        today = pd.Timestamp.now().normalize()
        future_mask = melted['date'] > today
        if future_mask.any():
            logger.warning(f"CSV: filtered {future_mask.sum()} rows with future dates")
            melted = melted[~future_mask]

        melted['model_key'] = melted['model_raw'].apply(_normalize_name)
        return melted
    except Exception as e:
        logger.error("CSV sales parse failed", exc_info=True)
        return pd.DataFrame()


def add_ph_features(df):
    """添加菲律宾特定的节假日和促销标志。"""
    from config_loader import cfg
    df = df.copy()

    def is_payday(date):
        d = date.day
        if d == 15 or d == 30 or d == 31: return 1
        if date.month == 2 and d == 29: return 1
        return 0

    df['is_payday'] = df['date'].apply(is_payday)
    df['is_double_digit'] = (df['date'].dt.month == df['date'].dt.day).astype(int)

    holidays_dt = pd.to_datetime(cfg.get('holidays', []))
    df['is_holiday'] = df['date'].isin(holidays_dt).astype(int)

    return df

def get_series_from_model(model_key):
    """根据型号 ID 识别系列。"""
    if 'v30' in model_key: return 'V30'
    if 'v40' in model_key: return 'V40'
    if 'v50' in model_key: return 'V50'
    if 'v60' in model_key: return 'V60'
    return 'Unknown'

def apply_price_history(df, price_history_df=None):
    """
    应用价格变更历史。
    若提供 price_history_df 则从 DataFrame 动态读取，否则使用硬编码逻辑。
    """
    df = df.copy()
    if price_history_df is not None and not price_history_df.empty:
        for _, ph in price_history_df.iterrows():
            mask = (
                df['model_key'].str.contains(ph['model_key'], case=False, regex=False)
            ) & (
                df['date'] >= ph['effective_from']
            )
            df.loc[mask, 'current_price'] += ph['price_delta']
    else:
        # 回退：硬编码价格历史
        mask_v50 = (df['model_key'].str.contains('v50', case=False)) & \
                   (df['date'] >= '2025-06-14')
        df.loc[mask_v50, 'current_price'] -= 2000
    return df


def get_integrated_dataset(file_path):
    """
    集成规格和销量数据，并执行特征工程。
    优先从 SQLite 数据库读取，数据库不存在时回退到 Excel 文件。
    """
    # 尝试从数据库加载
    _use_db = False
    price_history_df = None
    try:
        from db import db_exists, load_specs_df, load_daily_sales_df, load_price_history_df
        _use_db = db_exists()
    except ImportError:
        pass

    if _use_db:
        logger.info("Loading data from database...")
        specs = load_specs_df()
        sales = load_daily_sales_df()
        price_history_df = load_price_history_df()
    else:
        logger.info("Database unavailable, falling back to Excel...")
        logger.info("Processing specs (raw_data.xlsx)...")
        specs = clean_specs('data/raw_data.xlsx')

        logger.info("Processing daily sales (Lifetime file)...")
        sales_file = 'data/V30, V40, V50, V60 LIFETIME SALES.xlsx'
        sales = process_daily_sales(sales_file)

        # 数据清洗：排除低端 8+256 型号
        if not sales.empty:
            exclude_mask = (sales['model_raw'].str.contains('8+256', na=False)) & \
                           (sales['model_raw'].str.contains('V40|V50|V60', case=False, na=False))
            sales = sales[~exclude_mask].copy()

    # 合并数据
    merged = pd.merge(sales, specs, on='model_key', how='inner', suffixes=('', '_specs'))

    # 处理缺失的 current_price
    if 'current_price' not in merged.columns:
        merged['current_price'] = merged['price_base']

    # 应用价格变更历史
    merged = apply_price_history(merged, price_history_df)
    
    # --- 特征：价格与发布价格比（相对价格） ---
    # R2: 防止 price_base 为零导致 inf，先替换再计算
    zero_base = merged['price_base'] <= 0
    if zero_base.any():
        logger.warning(
            f"Found {zero_base.sum()} rows with zero/negative price_base; "
            f"replacing with 1.0 for ratio calculation"
        )
        merged.loc[zero_base, 'price_base'] = 1.0
    merged['price_to_launch_ratio'] = merged['current_price'] / merged['price_base']
    merged['price_to_launch_ratio'] = (
        merged['price_to_launch_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    )
    
    # 特征工程
    merged['dow'] = merged['date'].dt.dayofweek
    merged['is_weekend'] = merged['dow'].isin([5, 6]).astype(int)
    merged = add_ph_features(merged)
    
    merged = merged.sort_values(by=['model_key', 'date'])
    
    # --- 智能发布过滤逻辑 ---
    def filter_pre_launch(group):
        # R3: 正确处理全零销量情况，避免返回意外空 DataFrame
        key = group['model_key'].iloc[0] if not group.empty else 'unknown'
        peak_sales = group['daily_sales'].max()
        threshold = max(50, peak_sales * 0.2)
        real_launch_data = group[group['daily_sales'] >= threshold]
        if real_launch_data.empty:
            positive = group[group['daily_sales'] > 0]
            if positive.empty:
                logger.warning(
                    f"Model {key}: all sales are zero, cannot determine launch date; keeping all rows"
                )
                return group
            return positive
        first_real_date = real_launch_data['date'].min()
        return group[group['date'] >= first_real_date]

    merged = merged.groupby('model_key', group_keys=False).apply(filter_pre_launch)
    
    launch_dates = merged.groupby('model_key')['date'].min().to_dict()

    # P2: 向量化替代逐行 apply(axis=1)，大数据集提速 10-100x
    merged['_launch_date'] = merged['model_key'].map(launch_dates)
    raw_days = (merged['date'] - merged['_launch_date']).dt.days.fillna(0)
    days_diff_vec = raw_days.clip(lower=0)

    merged['months_since_launch'] = days_diff_vec / 30.0
    merged['days_since_launch']   = days_diff_vec.astype(int)
    merged['is_launch_day']       = (days_diff_vec == 0).astype(int)
    merged['is_launch_week']      = (days_diff_vec < 7).astype(int)
    merged.drop(columns=['_launch_date'], inplace=True)

    # --- Bass 理论基线计算 ---
    from bass_engine import BassEngine
    logger.info("Computing Bass theoretical baseline...")
    bass_engine = BassEngine()
    bass_engine.train_on_sheet2('data/raw_data.xlsx') 
    
    def get_bass_val(row):
        series = get_series_from_model(row['model_key'])
        return bass_engine.calculate_theoretical_sales(series, row['months_since_launch'])
        
    # 计算系列级 Bass 理论值
    merged['bass_theoretical'] = merged.apply(get_bass_val, axis=1)
    
    # 计算目标残差（虽然现在主模型改用了乘法，但保留此字段以供诊断）
    merged['sales_residual'] = merged['daily_sales'] - merged['bass_theoretical']

    # 生成滞后特征
    merged['lag_1d'] = merged.groupby('model_key')['daily_sales'].shift(1).fillna(0)
    merged['rolling_7d_mean'] = (
        merged.groupby('model_key')['daily_sales']
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
        .fillna(0)
    )

    return merged


# ---------------------------------------------------------------------------
# 数据导入函数（Excel → 数据库）
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {'.xlsx', '.xls', '.xlsm', '.xlsb', '.ods', '.odf', '.odt'}
SUPPORTED_SALES_EXTS = SUPPORTED_EXTS | {'.csv'}


def _check_ext(file_path: str, allowed: set, label: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed:
        raise ValueError(
            f"{label} 不支持的文件格式 '{ext}'。"
            f"支持的格式：{', '.join(sorted(allowed))}"
        )


def ingest_specs(file_path='data/raw_data.xlsx') -> int:
    """
    从 Excel 读取规格数据并写入数据库。
    支持格式：xlsx / xls / xlsm / xlsb / ods。
    自动检测含规格的 Sheet（优先 '512GB'）。
    已存在的 model_key 会被覆盖（支持规格修正）。
    返回写入行数。
    """
    _check_ext(file_path, SUPPORTED_EXTS,
               '规格文件（Specs）只能是 Excel 格式，因为需要读取特定 Sheet')
    from db import init_db, upsert_specs
    init_db()
    df = clean_specs(file_path)
    if df.empty:
        logger.warning(
            f"No specs data read from {file_path}. "
            f"Ensure the file has a specs sheet (e.g. '512GB') with a 'PRODUCT MODEL' column."
        )
        return 0
    count = upsert_specs(df)
    logger.info(f"Specs ingested: {count} rows")
    return count


def ingest_daily_sales(file_path='data/V30, V40, V50, V60 LIFETIME SALES.xlsx') -> int:
    """
    从文件读取每日销量并写入数据库。
    支持格式：xlsx / xls / xlsm / xlsb / ods / csv。
    Excel：通过 '第 0 列 == Model' 定位标题行（大小写不敏感）。
    CSV：第一行为标题，第 0 列列名须含 'model'。
    (model_key, date) 已存在的记录会被覆盖（支持数据修正）。
    返回写入行数。
    """
    _check_ext(file_path, SUPPORTED_SALES_EXTS, '销量文件（Sales）')
    from db import init_db, upsert_daily_sales
    init_db()
    df = process_daily_sales(file_path)
    if df.empty:
        logger.warning(
            f"No sales data read from {file_path}. "
            f"Excel: ensure a header row exists where column 0 is 'Model'. "
            f"CSV: ensure column 0 name contains 'model', remaining columns are dates."
        )
        return 0
    # 排除低端 8+256 型号（与训练数据过滤逻辑保持一致）
    exclude_mask = (
        df['model_raw'].str.contains('8+256', na=False) &
        df['model_raw'].str.contains('V40|V50|V60', case=False, na=False)
    )
    df = df[~exclude_mask].copy()
    count = upsert_daily_sales(df)
    logger.info(f"Daily sales ingested: {count} rows")
    return count


def ingest_all(specs_file='data/raw_data.xlsx',
               sales_file='data/V30, V40, V50, V60 LIFETIME SALES.xlsx'):
    """
    一步完成规格和销量数据导入。
    支持传入新文件路径，用于追加增量数据。

    用法：
        python data_processor_v2.py                          # 使用默认文件
        python data_processor_v2.py raw_data.xlsx sales.xlsx # 指定文件
    """
    logger.info("=== Starting data ingestion ===")
    specs_count = ingest_specs(specs_file)
    sales_count = ingest_daily_sales(sales_file)
    logger.info(f"=== Done: {specs_count} specs rows, {sales_count} sales rows ===")
    return specs_count, sales_count


if __name__ == '__main__':
    import sys
    import os

    # 切换到项目根目录，确保相对路径可用
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(_root)

    _specs = sys.argv[1] if len(sys.argv) > 1 else 'data/raw_data.xlsx'
    _sales = sys.argv[2] if len(sys.argv) > 2 else 'data/V30, V40, V50, V60 LIFETIME SALES.xlsx'
    ingest_all(_specs, _sales)