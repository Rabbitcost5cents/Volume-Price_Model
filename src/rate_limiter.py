"""
rate_limiter.py — 滑动窗口 + 最小冷却限速工具
同时适用于 Tkinter GUI（实例持有）和 Streamlit（session_state 存储）。
"""
import time
from typing import Tuple

# 默认参数（可在调用方覆盖）
DEFAULT_COOLDOWN = 3        # 两次运行之间的最小间隔（秒）
DEFAULT_MAX_CALLS = 20      # 滑动窗口内允许的最大调用次数
DEFAULT_WINDOW = 300        # 滑动窗口大小（秒）


class RateLimiter:
    """
    滑动窗口限速器，线程安全（基于 GIL），可序列化为 list 存入 session_state。

    用法（GUI）：
        self._rl = RateLimiter()
        allowed, msg = self._rl.check()
        if allowed:
            self._rl.record()

    用法（Streamlit）：
        if 'rl_timestamps' not in st.session_state:
            st.session_state.rl_timestamps = []
        rl = RateLimiter.from_list(st.session_state.rl_timestamps)
        allowed, msg = rl.check()
        if allowed:
            rl.record()
            st.session_state.rl_timestamps = rl.to_list()
    """

    def __init__(
        self,
        cooldown: float = DEFAULT_COOLDOWN,
        max_calls: int = DEFAULT_MAX_CALLS,
        window: float = DEFAULT_WINDOW,
    ):
        self.cooldown  = cooldown
        self.max_calls = max_calls
        self.window    = window
        self._timestamps: list = []

    # ── 序列化支持（用于 Streamlit session_state）─────────────────────────────

    @classmethod
    def from_list(cls, timestamps: list, **kwargs) -> "RateLimiter":
        rl = cls(**kwargs)
        rl._timestamps = list(timestamps)
        return rl

    def to_list(self) -> list:
        return list(self._timestamps)

    # ── 核心逻辑 ───────────────────────────────────────────────────────────────

    def _purge(self, now: float):
        """移除窗口之外的旧记录。"""
        self._timestamps = [t for t in self._timestamps if now - t < self.window]

    def check(self) -> Tuple[bool, str]:
        """
        检查当前是否允许执行。
        返回 (allowed: bool, reason_message: str)。
        message 为空字符串表示允许。
        """
        now = time.time()
        self._purge(now)

        # 1. 最小冷却检查
        if self._timestamps:
            elapsed = now - self._timestamps[-1]
            if elapsed < self.cooldown:
                wait = self.cooldown - elapsed
                return False, f"请稍候 {wait:.1f} 秒后再试。"

        # 2. 滑动窗口频次检查
        if len(self._timestamps) >= self.max_calls:
            oldest = self._timestamps[0]
            wait   = int(self.window - (now - oldest)) + 1
            return False, (
                f"操作过于频繁：{self.window / 60:.0f} 分钟内已运行 "
                f"{self.max_calls} 次，请 {wait} 秒后重试。"
            )

        return True, ""

    def record(self):
        """记录一次成功执行的时间戳（应在 check() 返回 True 后调用）。"""
        self._timestamps.append(time.time())
        self._purge(time.time())  # 顺带清理，防列表无限增长

    # ── 状态查询 ───────────────────────────────────────────────────────────────

    def remaining(self) -> int:
        """返回当前窗口内的剩余可用次数。"""
        now = time.time()
        self._purge(now)
        return max(0, self.max_calls - len(self._timestamps))

    def next_available_in(self) -> float:
        """返回距下次可执行还需等待的秒数（0 表示立即可用）。"""
        if not self._timestamps:
            return 0.0
        now     = time.time()
        elapsed = now - self._timestamps[-1]
        return max(0.0, self.cooldown - elapsed)
