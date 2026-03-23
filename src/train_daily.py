import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import logging
from data_processor_v2 import get_integrated_dataset, get_series_from_model
from bass_engine import BassEngine, bass_S
import json
import os
import shutil

logger = logging.getLogger(__name__)


def _backup_model_file(path: str):
    """Create a .bak copy of *path* if it exists, before overwriting."""
    if os.path.isfile(path):
        shutil.copy2(path, path + '.bak')

# M4: get_series_from_model 已从 data_processor_v2 导入，无需重复定义

def train_daily_model():
    file_path = 'data/raw_data.xlsx'
    os.makedirs('models', exist_ok=True)
    
    # 1. 加载集成数据
    logger.info("Loading integrated daily sales data...")
    df = get_integrated_dataset(file_path)
    
    # 2. 准备特征和目标变量
    # 目标变量现在是 LIFT（实际值 / Bass 理论值）
    df['sales_lift'] = df['daily_sales'] / (df['bass_theoretical'] + 1.0)
    target = 'sales_lift'
    
    feature_cols = [
        'current_price', 'price_to_launch_ratio',
        'ram_gb', 'storage_gb', 'battery_mah',
        'refresh_rate_hz', 'main_camera_mp', 'charging_w', 
        'screen_res', 'ip_rating', 
        'dow', 'is_weekend',
        'is_payday', 'is_double_digit', 'is_holiday',
        'lag_1d', 'rolling_7d_mean',
        'months_since_launch', 'days_since_launch', 'is_launch_day', 'is_launch_week'
    ]
    
    # D3: 记录缺失率，使用列中位数填充（避免 fillna(0) 把"缺失"混同为"零值"）
    missing = df[feature_cols].isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        missing_pct = (missing / len(df) * 100).round(1)
        for col, pct in missing_pct.items():
            logger.warning(f"Feature '{col}' has {pct}% missing values — filling with column median")

    # 计算训练集中位数（仅用训练数据，防止数据泄露）
    df_sorted = df.sort_values(by='date').copy()
    split_date = df_sorted['date'].quantile(0.8)
    train_df = df_sorted[df_sorted['date'] < split_date]
    test_df  = df_sorted[df_sorted['date'] >= split_date]

    # 以训练集中位数填充所有分割后的特征（含 0 值语义的标志列无需填充，默认已为 0）
    col_medians = train_df[feature_cols].median()
    X = df[feature_cols].fillna(col_medians).fillna(0)  # 双重兜底：中位数本身也为 NaN 时用 0
    y = df[target]

    # 3. 基于时间的划分（df_sorted 已在上方计算）

    X_train = train_df[feature_cols].fillna(col_medians).fillna(0)
    y_train = train_df[target]
    X_test = test_df[feature_cols].fillna(col_medians).fillna(0)
    y_test = test_df[target] 
    y_test_actuals = test_df['daily_sales'].values 
    
    logger.info(f"Time-based Split Cutoff: {split_date}")
    logger.info(f"Training set: {X_train.shape}, test set: {X_test.shape}")
    
    # 4. 训练 XGBoost 预测 LIFT
    logger.info("Training XGBoost (Target: Sales Lift)...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 5. 设置推理用的 Bass 引擎
    bass_engine = BassEngine()
    bass_engine.train_on_sheet2(file_path)
    
    # 6. 评估循环 (Base * Lift)
    logger.info("Running Base * Lift Prediction...")
    final_preds = []
    
    for idx, row in test_df.iterrows():
        months_sl = row['months_since_launch']
        model_key = row['model_key']
        series = get_series_from_model(model_key)
        
        # 1. 计算基准 (Bass)
        base_val = bass_engine.calculate_theoretical_sales(series, months_sl)
            
        # 2. 计算 Lift (XGBoost)
        input_row = X_test.loc[[idx]]
        pred_lift = max(0, model.predict(input_row)[0])
        
        # 3. 最终预测
        final_sales = base_val * pred_lift
        final_preds.append(final_sales)
        
    final_preds = np.array(final_preds)
    
    # 7. 评估
    rmse = np.sqrt(mean_squared_error(y_test_actuals, final_preds))
    r2 = r2_score(y_test_actuals, final_preds)
    
    logger.info(f"Model Evaluation (Base * Lift): RMSE={rmse:.2f} (Daily Units), R2={r2:.2f}")
    
    # --- 详细误差分析 ---
    results = pd.DataFrame({
        'date': test_df['date'].values,
        'model_key': test_df['model_key'].values,
        'actual': y_test_actuals,
        'predicted': final_preds
    })
    results['error'] = results['actual'] - results['predicted']
    results['abs_error'] = results['error'].abs()
    
    logger.info("\n--- Top 10 Worst Predictions ---\n" +
                results.sort_values(by='abs_error', ascending=False)[
                    ['date', 'model_key', 'actual', 'predicted', 'error', 'abs_error']
                ].head(10).to_string(index=False))
    logger.info("\n--- MAE by Model ---\n" +
                results.groupby('model_key')['abs_error'].mean().sort_values(ascending=False).to_string())

    # --- 保存人工制品（先备份已有文件） ---
    logger.info("Saving model and results to 'models/' directory...")
    _backup_model_file('models/xgb_model.json')
    model.save_model('models/xgb_model.json')

    _backup_model_file('models/feature_cols.json')
    with open('models/feature_cols.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f)

    results.to_csv('models/test_results.csv', index=False)
    logger.info("Artifacts saved.")

    # --- 将训练记录写入数据库 ---
    wmape = results['abs_error'].sum() / results['actual'].sum() if results['actual'].sum() != 0 else 0
    mae = results['abs_error'].mean()
    metrics = {
        'split_date': split_date,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'wmape': float(wmape),
        'mae': float(mae),
        'rmse': float(rmse),
        'notes': ''
    }
    try:
        from db import init_db, save_training_run
        init_db()
        save_training_run(metrics, results)
    except Exception as e:
        logger.warning(f"Failed to save training run to database (training unaffected): {e}")

    # 6. 特征重要性 (仅限 XGB)
    logger.info("Displaying Feature Importance (Gain)...")
    try:
        booster = model.get_booster()
        importance = booster.get_score(importance_type='gain')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        lines = [f"  {feat}: {score:.4f}" for feat, score in sorted_importance]
        logger.info("Feature Importance (Gain):\n" + "\n".join(lines))
    except Exception as e:
        logger.warning(f"Feature importance display failed (model saved successfully): {e}")

    # --- 训练冷启动模型（无滞后特征） ---
    logger.info("=== Training Cold Start Model (No Lags) ===")
    cold_start_cols = [c for c in feature_cols if c not in ['lag_1d', 'rolling_7d_mean']]
    
    # 过滤：在发布的前 14 天（爆发 + 早期尾部）进行训练，并排除 V30。
    train_df_cold = train_df[
        (train_df['days_since_launch'] <= 14) & 
        (~train_df['model_key'].str.contains('v30', case=False))
    ].copy()
    
    # D3: 与主模型保持一致，使用训练集中位数填充而非 0
    cold_medians = train_df_cold[cold_start_cols].median()
    X_train_cold = train_df_cold[cold_start_cols].fillna(cold_medians).fillna(0)
    # 目标变量现在是 LIFT，与主模型一致
    y_train_cold = train_df_cold['sales_lift']
    
    cold_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    cold_model.fit(X_train_cold, y_train_cold)
    
    _backup_model_file('models/xgb_cold_start.json')
    cold_model.save_model('models/xgb_cold_start.json')
    _backup_model_file('models/cold_start_cols.json')
    with open('models/cold_start_cols.json', 'w') as f:
        json.dump(cold_start_cols, f)
    logger.info("Cold Start Model saved to models/xgb_cold_start.json")

if __name__ == "__main__":
    train_daily_model()
