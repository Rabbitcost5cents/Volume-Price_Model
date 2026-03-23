import json
import os

# config.json 相对于本文件所在目录（src/）的上一级（项目根目录）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'config.json')

DEFAULT_CONFIG = {
    "elasticity": {
        "cold_price": -2.0,
        "main_price": -3.0,
        "battery": 0.5,
        "ip_rating": 0.2,
        "ram": 0.3,
        "storage": 0.15
    },
    "bass_boosts": {
        "p_mult": 1.2,
        "q_mult": 1.2,
        "launch_pulse": 3.0,
        "pulse_duration": 7.0
    },
    "seasonality": {
        "weekend": 1.15,
        "payday": 1.20,
        "holiday": 1.30
    },
    "logic_thresholds": {
        "cutoff_days": 7,
        "transition_window": 7,
        "min_lift": 0.8,
        "bass_floor": 0.4,
        "launch_cushion_start": 1.5,
        "launch_cushion_end": 1.0,
        "launch_cushion_days": 10
    },
    "baselines": {
        "price": 26000,
        "price_main": 25000,
        "battery": 6000,
        "battery_lock": 5000,
        "ip_rating": 68,
        "ip_lock": 54,
        "ram": 12,
        "storage": 256
    }
}

def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 中存在的键覆盖 base，缺失的键保留 base 默认值。"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path=None):
    """
    从 JSON 文件加载配置，对用户配置与 DEFAULT_CONFIG 执行深度合并。
    - 用户配置中存在的键：覆盖默认值。
    - 用户配置中缺失的键：自动回退到 DEFAULT_CONFIG，不会导致 KeyError。
    - 文件不存在或解析失败：直接使用 DEFAULT_CONFIG 并打印警告。
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"Warning: Config file '{config_path}' not found. Using defaults.")
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        if not isinstance(user_config, dict):
            raise ValueError("config.json 顶层必须是 JSON 对象（字典）。")
        merged = _deep_merge(DEFAULT_CONFIG, user_config)
        return merged
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        return DEFAULT_CONFIG

# 全局实例，供其他模块直接导入使用
cfg = load_config()
