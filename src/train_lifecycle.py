import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def extract_lifecycle_data(file_path):
    """
    从 Sheet2 提取月度生命周期数据。
    返回一个 DataFrame: ['series', 'month_index', 'monthly_sales']
    """
    # 读取标题行以识别系列位置
    df_raw = pd.read_excel(file_path, sheet_name='Sheet2', header=None)
    
    # 第 0 行包含系列标题（大约）
    # 第 1 行包含月份标题 (LM, M, M+1...)
    # 找到 "Grand Total" 行（基于检查为第 15 行）
    
    # 定位 "Grand Total" 行索引
    grand_total_idx = df_raw.index[df_raw[1].astype(str).str.strip().str.lower() == 'grand total'].tolist()
    if not grand_total_idx:
        raise ValueError("Could not find 'Grand Total' row in Sheet2")
    
    gt_row_idx = grand_total_idx[0]
    print(f"Grand Total Row found at index: {gt_row_idx}")
    
    # 扫描第 0 行以获取系列标题
    series_map = {} # {列索引: '系列名称'}
    row0 = df_raw.iloc[0]
    for c in range(len(row0)):
        val = str(row0[c])
        if 'SERIES' in val:
            series_name = val.split('SERIES')[0].strip() # "V40", "V50"
            series_map[c] = series_name
            
    print(f"Found Series Blocks: {series_map}")
    
    lifecycle_data = []
    
    # 对于每个系列，查找列 M 到 M+9
    # 子标题在第 1 行。
    row1 = df_raw.iloc[1]
    
    for start_col, series_name in series_map.items():
        # 在 start_col 之后的列中查找 'M'
        # 我们预期一个大约 12 列的数据块
        
        # 扫描 'M', 'M+1'...
        # 我们想要索引 0 到 9，代表 M 到 M+9
        
        found_m = False
        m_start_col = -1
        
        # 在 start_col 周围局部搜索（例如，在接下来的 5 列内）
        for c in range(start_col, min(start_col + 5, len(df_raw.columns))):
            if str(row1[c]).strip() == 'M':
                m_start_col = c
                found_m = True
                break
        
        if not found_m:
            print(f"Warning: Could not find 'M' column for {series_name}")
            continue
            
        # 提取 M 到 M+9（10 列）
        for month_offset in range(10): # 0 到 9
            col_idx = m_start_col + month_offset
            if col_idx >= len(df_raw.columns): break
            
            # 标签：M, M+1...
            col_label = str(row1[col_idx]).strip()
            
            # 来自 Grand Total 行的值
            sales_val = df_raw.iloc[gt_row_idx, col_idx]
            
            # 清洗数字
            try:
                sales_val = float(sales_val)
            except:
                sales_val = 0.0
                
            lifecycle_data.append({
                'series': series_name,
                'month_index': month_offset + 1, # 从 1 开始的索引（第 1 到 10 个月）
                'monthly_sales': sales_val,
                'label': col_label
            })
            
    return pd.DataFrame(lifecycle_data)

def train_lifecycle_model():
    file_path = 'data/raw_data.xlsx'
    print("Extracting lifecycle data...")
    df = extract_lifecycle_data(file_path)
    
    print("\nExtracted Data:")
    print(df)
    
    if df.empty:
        print("No data extracted.")
        return

    # 训练一个简单的模型：销量 ~ 系列 + 月份索引
    # 我们想看看月份索引如何影响销量。
    
    # 对系列进行编码
    df['series_code'] = df['series'].astype('category').cat.codes
    
    # 准备 X, y
    X = df[['series_code', 'month_index']]
    y = df['monthly_sales']
    
    # 训练
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    
    # 预测以观察曲线
    df['predicted'] = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, df['predicted']))
    print(f"\nModel RMSE: {rmse:.2f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    for series in df['series'].unique():
        subset = df[df['series'] == series]
        plt.plot(subset['month_index'], subset['monthly_sales'], marker='o', label=f'{series} Actual')
        plt.plot(subset['month_index'], subset['predicted'], linestyle='--', label=f'{series} Predicted')
        
    plt.title("Product Lifecycle Sales Curve (Sheet2 Total Data)")
    plt.xlabel("Month Since Launch (M, M+1...)")
    plt.ylabel("Total Monthly Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_lifecycle_model()