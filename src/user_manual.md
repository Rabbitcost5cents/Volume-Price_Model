# 销量-价格模型操作手册
# Sales-Price Model User Manual

> 本文档提供中英双语说明。中文版见第一部分，English version starts at [Part 2](#part-2-english-user-guide)。

---

# 第一部分：中文操作指南

本系统结合 **Bass 扩散模型（生命周期基准）** 与 **XGBoost（日销量波动预测）**，能够根据手机的硬件配置、价格和发布时间生成高精度的日级销量预测。系统提供两种操作界面：

- **桌面 GUI**（Tkinter）：离线运行，交互响应快。
- **Streamlit Web 应用**：浏览器访问，支持更丰富的图表与历史记录查询。

---

## 1. 环境准备与启动

### 1.1 前置条件

确认项目根目录下存在以下文件，否则需先完成数据导入和模型训练：

```
data/sales.db              ← SQLite 数据库（由数据导入步骤生成）
models/xgb_model.json      ← 主模型
models/xgb_cold_start.json ← 冷启动模型
models/feature_cols.json
models/cold_start_cols.json
models/test_results.csv
```

### 1.2 启动桌面 GUI

```bash
cd C:\Users\Rabbit\PycharmProjects\Volume-Price_Model
python src/gui_app.py
```

启动时窗口立即出现并显示加载进度，无需等待黑屏。

### 1.3 启动 Streamlit Web 应用

```bash
cd C:\Users\Rabbit\PycharmProjects\Volume-Price_Model
streamlit run src/streamlit_app.py
```

浏览器自动打开 `http://localhost:8501`。

> **⚠️ 首次启动需完成初始设置**：若 `models/admin.json` 不存在，Streamlit 会在任何页面渲染前显示**初始安全设置**页面，要求同时设置应用访问密码和管理员密码（均不能为空）。设置完成后方可正常使用。
>
> **特殊情形：如果已先通过桌面 GUI 设置了管理员密码**，则 `admin.json` 已存在但其中不含应用访问密码（`app_hash`）。此时 Streamlit 会显示**补充设置**页面，提示"管理员密码已由桌面客户端配置"，仅需补充填写应用访问密码（两次确认）即可完成设置。详见第 2.5 节。
>
> 后续访问需输入应用访问密码（见第 2 节）。

### 1.4 语言切换

两个界面均支持 **English / 中文** 切换：

- **GUI**：顶部工具栏右侧下拉菜单，选择语言后界面即时刷新，设置自动保存。
- **Streamlit**：左侧边栏顶部下拉菜单，选择后页面重新渲染。

---

## 2. 访问控制与安全

### 2.1 双层密码体系

系统采用两个独立密码保护不同功能层级：

```
应用访问密码
  └─ 所有用户进入 Streamlit 时必须输入
  └─ 控制只读页面的访问（模拟器、可视化、诊断、历史）

管理员密码
  └─ 在管理员页面单独输入
  └─ 解锁：数据导入、历史记录删除、模型重训、密码管理
```

> **无默认密码**：系统不预置任何默认密码。首次启动时（`models/admin.json` 不存在），Streamlit 会强制引导完成初始密码设置；GUI 在首次进入管理员选项卡时显示初始密码设置表单。设置完成前无法使用任何功能。

### 2.2 GUI — 管理员选项卡

在 GUI 顶部选项卡中切换至 **"🔐 管理员"（Admin）**：

**未登录状态**：若尚未设置过密码，显示**初始密码设置**表单（输入密码 + 确认密码）；已设置则显示常规登录框。

**已登录后可使用以下功能**：

| 功能区 | 说明 |
|--------|------|
| **修改管理员密码** | 输入新密码两次确认后生效，密码以 PBKDF2-HMAC-SHA256（含随机盐）加密存储 |
| **训练记录** | 查看最近 50 条训练历史；可选行删除或一键清空全部 |
| **模拟历史** | 查看最近 50 条模拟记录；可选行删除或一键清空全部 |
| **重新训练模型** | 点击按钮后台运行 `train_daily.py`，训练日志实时输出到内嵌控制台 |

> 管理员登录后，**"Data Import"** 选项卡自动解锁；退出登录后重新锁定。

### 2.3 Streamlit — 管理员页面

在左侧边栏导航中选择 **"🔐 Admin / 管理员"** 页面：

**未登录状态**：显示管理员密码输入框（与应用访问密码不同）。

**已登录后可使用以下功能**：

| 功能区 | 说明 |
|--------|------|
| **修改管理员密码** | 独立展开面板，修改后立即生效 |
| **修改应用访问密码** | 同页面右侧展开面板，控制所有用户的入口密码 |
| **训练记录** | 数据表展示 + ID 下拉选择删除 + 清空全部（含二次确认） |
| **模拟历史** | 同上 |
| **重新训练模型** | 运行训练脚本，完成后自动清除资源缓存，下次访问即加载新模型 |

**退出登录**：点击侧边栏底部 **"🚪 退出登录"** 按钮，同时清除应用登录和管理员登录状态。

### 2.5 跨客户端初始化场景

当桌面 GUI 与 Streamlit 混合使用时，有以下初始化顺序说明：

**场景 A（推荐）：先启动 Streamlit**

Streamlit 首次启动时一次性设置应用访问密码 + 管理员密码，生成完整的 `admin.json`。之后打开 GUI，管理员选项卡直接显示常规登录框（密码已就绪）。

**场景 B：先启动桌面 GUI**

GUI 仅设置管理员密码，`admin.json` 中不含 `app_hash`（应用访问密码）。此时打开 Streamlit，会看到：

> **"检测到管理员密码已由桌面客户端配置，请在此设置应用访问密码。"**

仅需填写应用访问密码（两次确认）即可完成 Streamlit 的访问控制设置。

**场景 C：重置密码**

若需要完全重置，删除 `models/admin.json` 后重启任一客户端，均会触发完整的初始设置流程。

---

### 2.6 密码存储位置

密码哈希值存储在 `models/admin.json`：

```json
{
  "app_hash":       "（salt_hex:dk_hex 格式，PBKDF2-HMAC-SHA256）",
  "password_hash":  "（salt_hex:dk_hex 格式，PBKDF2-HMAC-SHA256）"
}
```

- 密码明文不保存在任何位置；源代码中不存在默认密码。
- 每次修改密码时，系统生成新的随机 32 字节盐值，防止彩虹表攻击。
- 若 `admin.json` 不存在，系统会在首次启动时引导创建，**不会自动回退到任何预设密码**。
- 若存在旧版（SHA-256 无盐）哈希，首次登录成功后自动升级为 PBKDF2 格式。

---

## 3. 数据导入

> **⚠️ 权限要求**：数据导入功能仅限管理员使用。请先登录管理员账户（见第 2 节）。

在开始预测之前，需要将原始数据导入 SQLite 数据库。

### 3.1 通过界面导入（推荐）

管理员登录后，**GUI** 的 **"Data Import"** 选项卡 / **Streamlit** 的 **"Data Import"** 页面解锁，提供拖放或浏览上传功能：

| 文件类型 | 用途 | 支持格式 |
|----------|------|----------|
| Specs File（规格文件） | 手机硬件规格表（含价格、电池、内存等） | `.xlsx` `.xls` `.xlsm` `.xlsb` `.ods` |
| Sales File（销量文件） | 每日销量历史记录 | `.xlsx` `.xls` `.xlsm` `.xlsb` `.ods` `.csv` |

操作步骤：
1. 将文件拖放到对应区域，或点击 **"Browse"** 选择文件。
2. 文件路径显示后，点击 **"Import to Database"**。
3. 等待导入完成提示，即可进行模拟。

> **注意**：重复导入同一型号数据会自动覆盖（Upsert），不会产生重复记录。
>
> **文件安全校验**：系统在导入前会校验文件的 Magic Bytes（文件头签名），确保文件格式与扩展名一致。将其他格式文件改名为 `.xlsx` 等后上传将被拒绝并提示"文件签名不匹配"。
>
> **数据质量自动过滤**：导入时系统会自动过滤以下异常记录，并在日志中记录过滤数量：
> - **负数销量**（daily_sales < 0）：视为数据录入错误，自动剔除。
> - **未来日期**（date > 今日）：视为无效预测数据，自动剔除。

### 3.2 通过命令行导入

```bash
# 使用默认文件路径
python src/data_processor_v2.py

# 指定文件路径
python src/data_processor_v2.py path/to/raw_data.xlsx path/to/sales.xlsx
```

### 3.3 规格文件格式要求

- Excel 文件需包含规格 Sheet（优先识别名为 `512GB`、`512`、`Specs`、`Sheet1` 的 Sheet）。
- Sheet 内数据**横向排列**（每列为一款产品，行为规格项）。
- 必须包含列名：`PRODUCT MODEL`、`ORIGINAL PRICE`、`RAM`、`ROM`、`BATTERY` 等。

### 3.4 销量文件格式要求

**Excel 格式**：
- 每个产品组的第一行为标题行，第 0 列值为 `Model`（大小写不敏感）。
- 第 1 列为累计总销量（可选），第 2 列起为日期列，值为当日销量。

**CSV 格式**：
- 第一行为标题，第 0 列列名须包含 `model`。
- 其余列为日期（`YYYY-MM-DD` 格式），值为当日销量。

---

## 4. 模型训练

导入数据后，如需更新预测模型，有两种方式：

**方式一：通过管理员面板（推荐）**

在 GUI 管理员选项卡或 Streamlit 管理员页面的 **"重新训练模型"** 区域，点击按钮即可在界面内完成训练并查看日志。

**方式二：命令行**

```bash
cd C:\Users\Rabbit\PycharmProjects\Volume-Price_Model
python src/train_daily.py
```

训练完成后：
- 模型文件写入 `models/` 目录（覆盖前自动备份为 `.bak` 文件，如 `xgb_model.json.bak`）。
- 训练指标（WMAPE、MAE、RMSE）和测试集预测结果自动写入数据库。
- Bass 模型参数缓存到 `models/bass_params.json`，下次启动无需重新拟合；系统加载缓存时会对参数值域进行完整性校验，异常时自动回退到重新拟合。

> **何时需要重新训练**：新增了销量历史数据、更改了规格数据、或希望更新模型精度时。
>
> **模型恢复**：若新训练的模型质量不理想，可将 `models/` 目录下对应的 `.bak` 文件重命名（去掉 `.bak`）还原上一版本。

---

## 5. 销量模拟器使用指南

### 第一步：选择产品或场景

在产品下拉菜单中选择：

- **== NEW PRODUCT LAUNCH ==**：预测从未上市的新机型，系统加载历史中位数配置作为起点。
- **--- USER PRESETS ---**（下方列表）：加载之前保存的配置方案（如"V70激进版"）。
- **--- HISTORICAL MODELS ---**（下方列表）：加载已上市手机的最后状态，用于复盘或预测剩余生命周期。

### 第二步：设定发布日期

在 **"Launch Date"** 输入框填写上市时间，格式 `YYYY-MM-DD`（如 `2026-06-01`）。

> 发布日期影响"哪天是周末/发薪日/节假日"，直接改变预测曲线的波动形态。

### 第三步：调整硬件配置与价格

在 **"Specs & Price"** 区域修改各项参数（GUI 左侧面板支持上下滚动）：

| 参数 | 说明 | 有效范围 | 弹性参考 |
|------|------|---------|---------|
| Current Price | 当前售价（最核心参数） | 1 – 9,999,999 | 价格弹性约 -2.5 ~ -4.0 |
| Battery (mAh) | 电池容量 | 500 – 30,000 | 正弹性约 +0.5 |
| RAM (GB) | 运行内存 | 1 – 256 | 正弹性约 +0.3 |
| Storage (GB) | 存储空间 | 8 – 4,096 | 正弹性约 +0.15 |
| IP Rating | 防水等级 | 0 – 99 | 正弹性约 +0.2 |
| Refresh Rate (Hz) | 屏幕刷新率 | 1 – 500 | — |
| Main Camera (MP) | 主摄像头像素 | 1 – 500 | — |
| Charging (W) | 充电功率 | 1 – 500 | — |

> **参数超出范围**：点击"Run Simulation"后若任一参数超出上表有效范围，系统将拒绝执行并列出具体违规项，不会产生无效预测结果。

### 第四步：设定 Bass 模型参数

在 **"Bass Parameters"** 区域控制产品整体热度：

| 参数 | 说明 | 有效范围 | 推荐值 |
|------|------|---------|--------|
| Market Potential (m) | 全生命周期预计总销量 | 100 – 10,000,000 | V40/V50 约 7~8 万；旗舰 10 万+ |
| Innovation Coefficient (p) | 首发爆发力 | 0.001 – 1.0 | 稳健型 0.03~0.05；爆发型 0.15（如 V60） |

> **Bass 参数参考**：界面中提供同系列历史参数作为参考，可直接对照设置。

### 第五步：选择预测范围并运行

| 时长选项 | 适用场景 |
|----------|----------|
| 1 Week (Daily) | 观察首发 7 天每日细节 |
| 1 Month (Daily) | 第一个月完整每日曲线 |
| 6 Months (Daily) | 半年每日曲线 |
| 6 Months (Rolling Monthly) | **推荐**，最符合业务汇报需求 |
| 6 Months (Monthly) | 月度汇总 |

点击 **"Run Simulation"**，右侧图表即时生成预测曲线。

> **运行频率限制**：为保护服务器资源，每次运行需间隔至少 **3 秒**，且每 **5 分钟** 内最多运行 **20 次**。触发限制时系统会提示等待时间。

### 第六步：解读预测结果

- **首发日 (Day 0)**：通常为最高点，受 $p$ 值、价格和首发缓冲因子共同影响。
- **衰减期 (Day 1–14)**：销量从首发峰值平滑回落，Day 1 通常降至首发日的 60% 左右。
- **稳定期 (Day 14+)**：由主模型控制，价格弹性约 -4.0，周末/发薪日/节假日有周期性波动。
- **保底机制**：销量不低于 Bass 理论基准线的 80%，防止预测值过低。

---

## 6. 预设管理

### 保存预设

配置满意后点击 **"Save Preset"**：
- 若当前已选中预设：系统询问是"覆盖"还是"另存为新预设"。
- 预设保存至 `models/user_configs.json`，重启后依然存在。

### 删除预设

选中不需要的预设，点击 **"Delete Preset"** 即可。

---

## 7. 历史记录（Streamlit 专属）

Streamlit 界面的 **"History"** 页面提供：
- 最近 50 次模拟预测的汇总表（产品名、发布日期、时长、来源）。
- 最近 20 次模型训练记录（训练日期、WMAPE、MAE、RMSE）。

> 管理员可在 **"🔐 Admin"** 页面对历史记录进行删除管理。

---

## 8. 故障排查

### Q1：启动报错 "model file not found"
**原因**：`models/` 目录下缺少模型文件。
**解决**：先运行 `python src/train_daily.py` 训练模型，或通过管理员面板重新训练。

### Q2：运行模拟报错 "Invalid numeric input"
**原因**：输入框中含非数字字符（如 "26,000" 或空值）。
**解决**：确保所有输入框填写纯数字（如 `26000`）。

### Q3：数据导入后 "No data found"
**原因**：文件格式或 Sheet 结构与预期不符。
**解决**：
- 规格文件：确认 Sheet 名包含 `512GB` / `Specs`，且含 `PRODUCT MODEL` 列。
- 销量文件（Excel）：确认存在第 0 列值为 `Model` 的标题行。
- 销量文件（CSV）：确认第 0 列名包含 `model`。

### Q4：预测曲线是一条直线
**原因**：预测时段内无周末/发薪日，或参数设置过于保守。
**解决**：增加预测时长（至少 1 个月），或检查发布日期是否正确。

### Q5：修改配置后销量变化不明显
**原因**：修改幅度太小，或处于冷启动期（弹性系数较小）。
**解决**：尝试大幅调整价格（如涨价 5000），应能看到显著变化。

### Q6：Streamlit 页面显示访问密码输入框
**原因**：系统已启用访问控制，所有用户需验证身份后才能使用。
**解决**：输入应用访问密码。若忘记密码，请联系管理员在管理员页面重置。

### Q7：点击"Run Simulation"后显示"操作过于频繁"
**原因**：触发了运行频率限制（最小间隔 3 秒 / 每 5 分钟最多 20 次）。
**解决**：按提示等待相应时间后重试。

### Q8：数据导入选项卡被锁定
**原因**：该功能需要管理员权限，请先完成管理员登录。
**解决**：切换至 **"🔐 管理员"** 选项卡（GUI）或页面（Streamlit），输入管理员密码登录后即可解锁。

### Q9：首次启动显示"初始安全设置"页面
**原因**：`models/admin.json` 不存在，系统需要设置初始密码。
**解决**：按提示填写应用访问密码和管理员密码（两项均需两次确认）。设置完成后系统自动跳转到正常界面。此步骤只需完成一次。

### Q10：文件上传提示"文件签名不匹配"
**原因**：上传的文件扩展名与文件实际格式不符（如将 CSV 改名为 `.xlsx`），或文件已损坏。
**解决**：确保上传文件未经格式转换，直接使用软件原生导出的文件。若文件来自 Excel，请直接另存为 `.xlsx` 格式。

### Q11：模拟参数提示"超出有效范围"
**原因**：某项参数值超出系统允许的边界（如价格为负数、Bass m 超过 10,000,000 等）。
**解决**：按提示检查并修正对应字段的数值，参见第 5 节参数表格中的"有效范围"列。

### Q12：先用 GUI 设置密码后，打开 Streamlit 时显示应用密码设置页面
**原因**：桌面 GUI 仅设置管理员密码，`admin.json` 中缺少应用访问密码（`app_hash`）。Streamlit 检测到此状态后显示**补充设置**页面。
**解决**：按页面提示填写应用访问密码（两次确认），提交后系统自动跳转到正常访问界面。此操作只需执行一次。详见第 2.5 节。

### Q13：GUI 程序异常崩溃，如何查看详细错误信息？
**解决**：GUI 运行时会将详细日志（含完整堆栈跟踪）写入 `gui_app.log` 文件（位于项目根目录）。将该文件内容提供给开发者有助于定位问题。

---

## 9. 项目文件结构

```
Volume-Price_Model/
├── config.json              ← 全局参数（弹性系数、Bass 约束、节假日列表）
├── data/
│   └── sales.db             ← SQLite 数据库（规格、销量、训练历史、模拟历史）
├── models/
│   ├── xgb_model.json           ← 主 XGBoost 模型
│   ├── xgb_model.json.bak       ← 上一版主模型备份（每次训练自动生成）
│   ├── xgb_cold_start.json      ← 冷启动模型（发布前 14 天）
│   ├── xgb_cold_start.json.bak  ← 上一版冷启动模型备份
│   ├── feature_cols.json        ← 主模型特征列表
│   ├── feature_cols.json.bak    ← 备份
│   ├── cold_start_cols.json     ← 冷启动模型特征列表
│   ├── cold_start_cols.json.bak ← 备份
│   ├── test_results.csv         ← 最近一次训练的测试集结果
│   ├── bass_params.json         ← Bass 参数缓存（自动生成，加速启动）
│   ├── admin.json               ← 应用访问密码和管理员密码的 PBKDF2 哈希值
│   └── user_configs.json        ← 用户保存的预设配置
└── src/
    ├── gui_app.py           ← 桌面 GUI 主程序
    ├── streamlit_app.py     ← Streamlit Web 应用主程序
    ├── app.py               ← 核心模拟引擎（SalesSimulator）
    ├── auth.py              ← 密码哈希与校验工具（GUI 和 Streamlit 共用）
    ├── bass_engine.py       ← Bass 扩散模型拟合与预测
    ├── rate_limiter.py      ← 模拟运行频率限速工具
    ├── data_processor_v2.py ← 数据清洗、特征工程、数据导入
    ├── db.py                ← SQLite 数据库工具模块
    ├── train_daily.py       ← 主模型训练脚本
    ├── train_lifecycle.py   ← 生命周期数据提取脚本
    ├── config_loader.py     ← config.json 加载器
    ├── i18n.py              ← 国际化字符串（英文/中文）
    └── migrate_excel.py     ← 一次性 Excel → 数据库迁移脚本
```

## 10. 关键配置参考（`config.json`）

| 配置节 | 键名 | 说明 |
|--------|------|------|
| `holidays` | 日期列表 | 菲律宾公众假期 2021–2026（法定节假日 + 特别非工作日） |
| `bass_fit` | `conservative_series` | 使用更严格市场潜量上限的系列名称（默认：`["V60"]`） |
| `bass_fit` | `m_upper_conservative` | 保守系列的市场潜量上限倍率（默认：1.5×） |
| `bass_fit` | `m_upper_default` | 默认市场潜量上限倍率（默认：3.0×） |
| `elasticity` | `cold_price` | 冷启动模型的价格弹性（默认：−2.0） |
| `elasticity` | `main_price` | 主模型的价格弹性（默认：−3.0） |
| `seasonality` | `weekend` | 周末销量倍率（默认：1.15） |
| `seasonality` | `payday` | 发薪日（15 日、30 日、31 日）销量倍率（默认：1.20） |
| `seasonality` | `holiday` | 公众假期销量倍率（默认：1.30） |
| `logic_thresholds` | `cutoff_days` | 从冷启动模型切换至主模型的天数阈值（默认：7 天） |

---

> **部分覆盖支持**：`config.json` 现在支持只写需要修改的字段，系统会将用户配置与内置默认值深度合并，缺失的键自动使用默认值，不会因字段缺失引发运行时错误。

---

**文档版本**：v4.1
**最后更新**：2026-03-23

---
---

# Part 2: English User Guide

This system combines a **Bass Diffusion Model (lifecycle baseline)** with **XGBoost (daily sales fluctuation prediction)** to generate high-accuracy daily sales forecasts based on a smartphone's hardware specifications, price, and launch date. Two interfaces are available:

- **Desktop GUI** (Tkinter): runs offline, fast and responsive.
- **Streamlit Web App**: browser-based, richer charts and simulation history.

---

## 1. Setup & Launch

### 1.1 Prerequisites

Ensure the following files exist before running the application. If they are missing, complete the data import and model training steps first.

```
data/sales.db              ← SQLite database (created during data import)
models/xgb_model.json      ← Main prediction model
models/xgb_cold_start.json ← Cold-start model
models/feature_cols.json
models/cold_start_cols.json
models/test_results.csv
```

### 1.2 Launch the Desktop GUI

```bash
cd C:\Users\Rabbit\PycharmProjects\Volume-Price_Model
python src/gui_app.py
```

A loading splash screen appears immediately on startup — no blank wait period.

### 1.3 Launch the Streamlit Web App

```bash
cd C:\Users\Rabbit\PycharmProjects\Volume-Price_Model
streamlit run src/streamlit_app.py
```

The browser will automatically open `http://localhost:8501`.

> **⚠️ First launch requires initial setup.** If `models/admin.json` does not exist, Streamlit displays an **Initial Security Setup** page before rendering anything else. You must set both the app access password and the admin password (neither may be empty) to proceed.
>
> **Special case: if the desktop GUI was used first to set an admin password**, `admin.json` already exists but contains no app access password (`app_hash`). In this case Streamlit shows a **Supplemental Setup** page, noting that the admin password was already configured by the desktop client, and only asks you to set the app access password (with confirmation). See Section 2.5 for details.
>
> Subsequent visits require the app access password (see Section 2).

### 1.4 Switching Language

Both interfaces support **English / 中文** switching:

- **GUI**: dropdown in the top toolbar — changes take effect immediately and are saved automatically.
- **Streamlit**: dropdown at the top of the left sidebar — page re-renders on change.

---

## 2. Access Control & Security

### 2.1 Two-Tier Password System

The system uses two independent passwords to protect different functional levels:

```
App Access Password
  └─ Required by all users to enter the Streamlit app
  └─ Controls access to read-only pages (Simulator, Viz, Diagnostics, History)

Admin Password
  └─ Entered separately on the Admin page
  └─ Unlocks: Data Import, record deletion, model retraining, password management
```

> **No default passwords.** The system ships with no pre-configured credentials. On first launch (when `models/admin.json` is absent), Streamlit forces you through an initial setup page; the GUI shows an initial setup form the first time you visit the Admin tab. Neither interface is usable until setup is complete.

### 2.2 GUI — Admin Tab

Switch to the **"🔐 Admin"** tab in the GUI notebook:

**When not logged in**: If no password has been set yet, an **initial password setup** form is shown (password + confirmation). Otherwise, the standard login form is displayed.

**After logging in, the following functions are available**:

| Section | Description |
|---------|-------------|
| **Change Admin Password** | Enter and confirm a new password; stored as PBKDF2-HMAC-SHA256 with a random salt |
| **Training Records** | View up to 50 training runs; delete individual rows or clear all |
| **Simulation History** | View up to 50 simulation runs; delete individual rows or clear all |
| **Retrain Model** | Runs `train_daily.py` in a background thread; live log output shown in the embedded console |

> After admin login, the **"Data Import"** tab unlocks automatically. It re-locks on logout.

### 2.3 Streamlit — Admin Page

Select **"🔐 Admin"** from the left sidebar navigation:

**When not logged in**: An admin password input form is displayed (separate from the app access password).

**After logging in, the following functions are available**:

| Section | Description |
|---------|-------------|
| **Change Admin Password** | Collapsible panel on the left |
| **Change App Access Password** | Collapsible panel on the right — controls the entry password for all users |
| **Training Records** | Table view + ID dropdown for individual deletion + clear all (with confirmation) |
| **Simulation History** | Same controls as training records |
| **Retrain Model** | Runs the training script; output displayed via `st.code`; cache cleared automatically on success |

**Logging out**: Click the **"🚪 Log Out"** button at the bottom of the sidebar to clear both the app session and the admin session simultaneously.

### 2.5 Cross-Client Initialization

When the desktop GUI and Streamlit are both in use, the following initialization scenarios apply:

**Scenario A (Recommended): Launch Streamlit first**

Streamlit's first-run setup collects both the app access password and the admin password in a single step, producing a complete `admin.json`. When the GUI is opened afterward, its Admin tab shows the standard login form immediately.

**Scenario B: Launch the desktop GUI first**

The GUI only configures the admin password. `admin.json` is written without an `app_hash` (app access password). When Streamlit is then opened, it displays:

> **"Admin password was already configured by the desktop client. Please set the app access password to continue."**

Simply fill in the app access password (with confirmation) to complete Streamlit's access-control setup.

**Scenario C: Full reset**

To reset all credentials, delete `models/admin.json` and restart either client. Both will trigger the complete initial setup flow.

---

### 2.6 Password Storage

Password hashes are stored in `models/admin.json`:

```json
{
  "app_hash":       "(salt_hex:dk_hex — PBKDF2-HMAC-SHA256)",
  "password_hash":  "(salt_hex:dk_hex — PBKDF2-HMAC-SHA256)"
}
```

- Password plaintext is never stored anywhere; no default passwords exist in the source code.
- A fresh 32-byte random salt is generated each time a password is changed, preventing rainbow-table attacks.
- If `admin.json` is missing, the system guides you through first-run setup — it does **not** fall back to any pre-set credentials.
- Legacy SHA-256 hashes (from a previous installation) are silently upgraded to PBKDF2 on the first successful login.

---

## 3. Data Import

> **⚠️ Admin access required.** The Data Import function is restricted to administrators. Log in via the Admin tab/page first (see Section 2).

Raw data must be imported into the SQLite database before running simulations.

### 3.1 Import via the Interface (Recommended)

After admin login, the **"Data Import"** tab (GUI) or **"Data Import"** page (Streamlit) unlocks and provides drag-and-drop or browse-to-upload functionality:

| File Type | Purpose | Supported Formats |
|-----------|---------|-------------------|
| Specs File | Smartphone hardware specs (price, battery, RAM, etc.) | `.xlsx` `.xls` `.xlsm` `.xlsb` `.ods` |
| Sales File | Daily sales history | `.xlsx` `.xls` `.xlsm` `.xlsb` `.ods` `.csv` |

Steps:
1. Drag and drop a file onto the target area, or click **"Browse"** to select one.
2. Once the path is shown, click **"Import to Database"**.
3. Wait for the completion message before running simulations.

> **Note**: Re-importing the same model key automatically overwrites the existing record (upsert). No duplicates are created.
>
> **File security validation**: Before processing, the system checks each file's magic bytes (file header signature) to verify the format matches its extension. Renaming a file of a different type to `.xlsx` will be rejected with a "file signature mismatch" error.
>
> **Automatic data quality filtering**: During import, the system silently removes the following invalid records and logs how many were dropped:
> - **Negative sales values** (`daily_sales < 0`): treated as data-entry errors.
> - **Future-dated records** (`date > today`): treated as invalid forecast placeholders.

### 3.2 Import via Command Line

```bash
# Use default file paths
python src/data_processor_v2.py

# Specify custom paths
python src/data_processor_v2.py path/to/raw_data.xlsx path/to/sales.xlsx
```

### 3.3 Specs File Format

- The file must be an Excel workbook containing a specs sheet (system auto-detects sheets named `512GB`, `512`, `Specs`, or `Sheet1`).
- Data is laid out **horizontally**: each column is one product, rows are spec fields.
- Required column names include: `PRODUCT MODEL`, `ORIGINAL PRICE`, `RAM`, `ROM`, `BATTERY`, etc.

### 3.4 Sales File Format

**Excel format**:
- Each product block starts with a header row where column 0 equals `Model` (case-insensitive).
- Column 1 is the cumulative lifetime sales (optional); columns 2 onward are date columns with daily sales values.

**CSV format**:
- First row is the header; column 0 name must contain `model`.
- Remaining columns are dates (`YYYY-MM-DD`), values are daily sales figures.

---

## 4. Model Training

After importing or updating data, retrain the model to reflect the latest information. Two options are available:

**Option 1: Via Admin Panel (Recommended)**

In the GUI Admin tab or Streamlit Admin page, use the **"Retrain Model"** section. Training output is streamed live to the embedded console; no terminal access is needed.

**Option 2: Command Line**

```bash
cd C:\Users\Rabbit\PycharmProjects\Volume-Price_Model
python src/train_daily.py
```

After training completes:
- Model files are written to the `models/` directory. The previous version of each file is automatically backed up as `<filename>.bak` (e.g. `xgb_model.json.bak`) before being overwritten.
- Training metrics (WMAPE, MAE, RMSE) and test-set predictions are automatically saved to the database.
- Bass model parameters are cached to `models/bass_params.json`. The cache is validated on load (value-range check on p, q, m); if validation fails the system silently re-fits from source data.

> **When to retrain**: after adding new sales history, updating specs data, or whenever you want to improve forecast accuracy.
>
> **Model rollback**: if the newly trained model performs worse, rename the corresponding `.bak` file (remove the `.bak` suffix) in the `models/` directory to restore the previous version.

---

## 5. Sales Simulator Guide

### Step 1 — Select a Product or Scenario

Use the product dropdown to choose:

- **== NEW PRODUCT LAUNCH ==**: Forecast a phone that has never been released. The system pre-fills median historical specs as a starting point.
- **--- USER PRESETS ---** (items below): Load a previously saved configuration (e.g. "V70 Aggressive").
- **--- HISTORICAL MODELS ---** (items below): Load the last-known state of a released phone to review past performance or predict the remainder of its lifecycle.

### Step 2 — Set the Launch Date

Enter the launch date in the **"Launch Date"** field using the format `YYYY-MM-DD` (e.g. `2026-06-01`).

> The launch date determines which days fall on weekends, paydays, and public holidays, directly shaping the shape of the forecast curve.

### Step 3 — Adjust Hardware Specs & Price

Edit parameters in the **"Specs & Price"** panel (the GUI left panel is scrollable if content overflows):

| Parameter | Description | Valid Range | Elasticity Reference |
|-----------|-------------|-------------|----------------------|
| Current Price | Selling price — the most critical parameter | 1 – 9,999,999 | Approx. −2.5 to −4.0 |
| Battery (mAh) | Capacity | 500 – 30,000 | Positive ~+0.5 |
| RAM (GB) | RAM | 1 – 256 | Positive ~+0.3 |
| Storage (GB) | Internal storage | 8 – 4,096 | Positive ~+0.15 |
| IP Rating | Water resistance | 0 – 99 | Positive ~+0.2 |
| Refresh Rate (Hz) | Screen refresh rate | 1 – 500 | — |
| Main Camera (MP) | Main camera resolution | 1 – 500 | — |
| Charging (W) | Charging power | 1 – 500 | — |

> **Out-of-range protection**: If any parameter falls outside its valid range when you click "Run Simulation", the system rejects the run and lists the offending fields. No invalid forecasts are produced.

### Step 4 — Configure Bass Model Parameters

Control the overall product lifecycle shape in the **"Bass Parameters"** panel:

| Parameter | Description | Valid Range | Recommended Values |
|-----------|-------------|-------------|---------------------|
| Market Potential (m) | Estimated total lifetime sales | 100 – 10,000,000 | V40/V50 ~70,000–80,000; flagship 100,000+ |
| Innovation Coefficient (p) | Launch-day burst strength | 0.001 – 1.0 | Steady: 0.03–0.05 · Burst (e.g. V60): 0.15 |

> **Reference values**: The panel displays historical Bass parameters for the same series — use these as benchmarks when setting your own values.

### Step 5 — Choose Forecast Duration and Run

| Duration Option | Best For |
|-----------------|----------|
| 1 Week (Daily) | Detailed first-week daily breakdown |
| 1 Month (Daily) | Complete daily curve for month one |
| 6 Months (Daily) | Six-month daily detail |
| 6 Months (Rolling Monthly) | **Recommended** — standard business reporting |
| 6 Months (Monthly) | Monthly aggregation |

Click **"Run Simulation"**. The forecast chart renders immediately on the right.

> **Rate Limiting**: To protect server resources, a minimum interval of **3 seconds** is enforced between runs, and no more than **20 runs per 5-minute window** are allowed. If the limit is reached, the system displays the remaining wait time.

### Step 6 — Interpreting the Results

- **Launch Day (Day 0)**: Typically the peak. Driven by $p$, price, and the launch cushion multiplier.
- **Decay Phase (Days 1–14)**: Sales taper smoothly from the launch peak. Day 1 is usually ~60% of Day 0.
- **Steady Phase (Day 14+)**: Governed by the main XGBoost model. Price elasticity ~−4.0; periodic spikes from weekends, paydays, and public holidays.
- **Floor Mechanism**: Predictions never fall below 80% of the Bass theoretical baseline, preventing unrealistically low forecasts.

---

## 6. Preset Management

### Saving a Preset

Once satisfied with a configuration, click **"Save Preset"**:
- If a preset is currently selected: the system asks whether to **overwrite** or **save as new**.
- Presets are stored in `models/user_configs.json` and persist across restarts.

### Deleting a Preset

Select an unwanted preset from the dropdown, then click **"Delete Preset"**.

---

## 7. History (Streamlit Only)

The **"History"** page in the Streamlit app provides:
- A summary table of the **50 most recent simulations** (product name, launch date, duration, source interface).
- A table of the **20 most recent training runs** (date, WMAPE, MAE, RMSE).

> Administrators can delete individual records or clear all history from the **"🔐 Admin"** page.

---

## 8. Troubleshooting

### Q1: App fails to start — "model file not found"
**Cause**: Model files are missing from the `models/` directory.
**Fix**: Run `python src/train_daily.py` to train and generate the model files, or use the Admin panel's Retrain function.

### Q2: Simulation error — "Invalid numeric input"
**Cause**: An input field contains non-numeric characters (e.g. "26,000" or empty).
**Fix**: Ensure all fields contain plain numbers (e.g. `26000`).

### Q3: Data import completes but shows "No data found"
**Cause**: File format or sheet structure does not match expectations.
**Fix**:
- Specs file: confirm the sheet name contains `512GB` / `Specs`, and that `PRODUCT MODEL` column exists.
- Sales file (Excel): confirm a header row where column 0 equals `Model`.
- Sales file (CSV): confirm column 0 name contains `model`.

### Q4: Forecast curve is a flat line
**Cause**: No weekends or paydays fall within the forecast window, or parameters are too conservative.
**Fix**: Extend the forecast duration to at least 1 month, or verify the launch date is set correctly.

### Q5: Changing specs has no visible effect
**Cause**: The change is too small, or the product is in the cold-start phase (lower elasticity).
**Fix**: Try a large price change (e.g. +5,000) — you should see a clear drop in forecast sales.

### Q6: Streamlit shows an access password prompt
**Cause**: App-level access control is enabled. All users must authenticate before viewing any page.
**Fix**: Enter the app access password. If you have forgotten it, ask the administrator to reset it via the Admin page.

### Q7: "Run Simulation" shows a rate-limit warning
**Cause**: The minimum 3-second interval or the 20-runs-per-5-minutes limit has been reached.
**Fix**: Wait the indicated number of seconds, then try again.

### Q8: The Data Import tab/page is locked
**Cause**: This function requires admin privileges.
**Fix**: Switch to the **"🔐 Admin"** tab (GUI) or page (Streamlit), enter the admin password, and log in to unlock it.

### Q9: First launch shows an "Initial Security Setup" page
**Cause**: `models/admin.json` does not exist — the system needs passwords to be configured before it can operate.
**Fix**: Fill in both the app access password and the admin password (each requires confirmation). After submitting, the system saves the hashes and redirects to the normal interface. This step only happens once.

### Q10: File upload rejected — "file signature mismatch"
**Cause**: The file's actual format does not match its extension (e.g. a CSV renamed to `.xlsx`), or the file is corrupted.
**Fix**: Use the file as originally exported by the source application. If using Excel, save as `.xlsx` directly from Excel rather than renaming another file type.

### Q11: Simulation rejected — "parameter out of valid range"
**Cause**: One or more spec or Bass parameters fall outside the allowed bounds.
**Fix**: Review the flagged fields shown in the error message and correct the values. Refer to the valid-range columns in the Step 3 and Step 4 tables in Section 5.

### Q12: Streamlit shows an app-access-password setup page after I already configured a password in the GUI
**Cause**: The desktop GUI only sets the admin password. `admin.json` is written without an app access password (`app_hash`). Streamlit detects this partial state and shows a **Supplemental Setup** page.
**Fix**: Enter and confirm the app access password as prompted. After submitting, Streamlit saves the hash and proceeds normally. This only needs to be done once. See Section 2.5 for full details.

### Q13: The GUI crashed — where can I find detailed error information?
**Fix**: The GUI writes detailed logs (including full stack traces) to `gui_app.log` in the project root directory. Providing this file to the developer will help diagnose the problem.

---

## 9. Project File Structure

```
Volume-Price_Model/
├── config.json              ← Global parameters (elasticity, Bass constraints, public holidays)
├── data/
│   └── sales.db             ← SQLite database (specs, sales, training history, sim history)
├── models/
│   ├── xgb_model.json           ← Main XGBoost model
│   ├── xgb_model.json.bak       ← Previous model backup (auto-created on each retrain)
│   ├── xgb_cold_start.json      ← Cold-start model (first 14 days post-launch)
│   ├── xgb_cold_start.json.bak  ← Previous cold-start model backup
│   ├── feature_cols.json        ← Main model feature list
│   ├── feature_cols.json.bak    ← Backup
│   ├── cold_start_cols.json     ← Cold-start model feature list
│   ├── cold_start_cols.json.bak ← Backup
│   ├── test_results.csv         ← Test-set predictions from the latest training run
│   ├── bass_params.json         ← Cached Bass parameters (auto-generated, speeds up startup)
│   ├── admin.json               ← PBKDF2-hashed app access password and admin password
│   └── user_configs.json        ← User-saved preset configurations
└── src/
    ├── gui_app.py           ← Desktop GUI main application
    ├── streamlit_app.py     ← Streamlit web app main application
    ├── app.py               ← Core simulation engine (SalesSimulator)
    ├── auth.py              ← Password hashing and verification (shared by GUI and Streamlit)
    ├── bass_engine.py       ← Bass diffusion model fitting and prediction
    ├── rate_limiter.py      ← Simulation run-rate limiter utility
    ├── data_processor_v2.py ← Data cleaning, feature engineering, data ingestion
    ├── db.py                ← SQLite database utility module
    ├── train_daily.py       ← Main model training script
    ├── train_lifecycle.py   ← Lifecycle data extraction script
    ├── config_loader.py     ← config.json loader
    ├── i18n.py              ← Internationalization strings (English / Chinese)
    └── migrate_excel.py     ← One-time Excel → database migration script
```

---

## 10. Key Configuration Reference (`config.json`)

| Section | Key | Description |
|---------|-----|-------------|
| `holidays` | list of dates | Philippine public holidays 2021–2026 (Regular + Special Non-Working Days) |
| `bass_fit` | `conservative_series` | Series names that use a tighter market potential cap (default: `["V60"]`) |
| `bass_fit` | `m_upper_conservative` | Market potential upper bound multiplier for conservative series (default: 1.5×) |
| `bass_fit` | `m_upper_default` | Default upper bound multiplier (default: 3.0×) |
| `elasticity` | `cold_price` | Price elasticity for the cold-start model (default: −2.0) |
| `elasticity` | `main_price` | Price elasticity for the main model (default: −3.0) |
| `seasonality` | `weekend` | Sales multiplier on weekends (default: 1.15) |
| `seasonality` | `payday` | Sales multiplier on paydays (15th, 30th, 31st; default: 1.20) |
| `seasonality` | `holiday` | Sales multiplier on public holidays (default: 1.30) |
| `logic_thresholds` | `cutoff_days` | Days since launch before switching from cold-start to main model (default: 7) |

> **Partial override support**: `config.json` now supports partial configurations. The system deep-merges your file with built-in defaults — only the keys you specify are overridden; missing keys automatically fall back to their default values, preventing `KeyError` crashes from incomplete configs.

---

**Document Version**: v4.1
**Last Updated**: 2026-03-23
