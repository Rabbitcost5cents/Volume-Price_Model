"""
auth.py — 公共密码哈希与校验工具
供 gui_app.py 和 streamlit_app.py 共同使用，避免重复定义。

哈希格式：PBKDF2-HMAC-SHA256，32 字节随机盐，260,000 次迭代。
存储格式：'salt_hex:dk_hex'（冒号分隔的十六进制字符串）。
兼容性：同时支持验证旧版无盐 SHA-256 哈希（用于遗留 admin.json 自动升级）。
"""
import hashlib
import hmac
import os
from typing import Optional


def hash_pw(pw: str) -> str:
    """对密码进行 PBKDF2-HMAC-SHA256 哈希，返回 'salt_hex:dk_hex' 格式字符串。"""
    salt = os.urandom(32)
    dk = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 260_000)
    return salt.hex() + ':' + dk.hex()


def verify_pw(pw: str, stored: Optional[str]) -> bool:
    """
    验证明文密码与存储的哈希是否匹配。

    支持两种格式：
      - 新格式 'salt_hex:dk_hex'：PBKDF2-HMAC-SHA256（推荐）
      - 旧格式（无冒号）：无盐 SHA-256（仅用于向后兼容，登录成功后应升级）

    使用 hmac.compare_digest 进行常量时间比较，防止时序攻击。
    """
    if not stored:
        return False
    try:
        if ':' in stored:
            salt_hex, dk_hex = stored.split(':', 1)
            actual = hashlib.pbkdf2_hmac(
                'sha256', pw.encode('utf-8'), bytes.fromhex(salt_hex), 260_000
            )
            return hmac.compare_digest(actual, bytes.fromhex(dk_hex))
        # 旧版无盐 SHA-256（遗留兼容，调用方应在验证成功后升级哈希）
        return hmac.compare_digest(
            hashlib.sha256(pw.encode('utf-8')).hexdigest(), stored
        )
    except Exception:
        return False
