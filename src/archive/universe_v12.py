"""
涌积态宇宙模型 v12.0 — 塔蒂尼相位锁定：量子层验证架构
基于 v11.0 最终参数，引入微观量子非线性交叉项，验证"塔蒂尼音"假说。

═══════════════════════════════════════════════════════════════════
  理论背景：塔蒂尼音 → 涌积态量子层映射
═══════════════════════════════════════════════════════════════════

  在声学中，塔蒂尼音（差频）是非线性媒介对两股基频波的强制重新采样：
    两股波 f1, f2 → 非线性媒介 → 差频包络波 |f1 - f2|

  在涌积态量子层中，对应关系为：
    P 场（概率波函数）= 弥散的量子叠加态
    失真算子 D（乘性噪声）= 量子真空涨落（非线性媒介）
    P_A × P_B 乘积项 = 相位锁定的包络波（塔蒂尼音）
    C1 的诞生 = 波函数坍缩（相位锁定 → 定域化）
    有损算子 L 的记忆沉淀 = 时间箭头涌现（锁定的不可逆性）

═══════════════════════════════════════════════════════════════════
  v12.0 核心新增：塔蒂尼非线性交叉项（Tartini Nonlinear Cross-Term）
═══════════════════════════════════════════════════════════════════

  在 P 场的 PDE 更新中引入非线性自相互作用项：

    dP/dt = D_P·∇²P - A_{P→C1} + ... - β·P²

  其中 β·P² 是"塔蒂尼项"：
    - P² 模拟 P_A × P_B 的局部非线性耦合（两股概率波的干涉）
    - β 控制非线性强度（量子真空涨落的耦合系数）
    - 负号表示非线性耦合消耗 P 场能量（相位锁定的代价）
    - 被消耗的 P 场能量转化为 C1（波函数坍缩的产物）

  守恒性保证：
    dP/dt 中减去的 β·P² 必须等量加入 dC1/dt，确保总质量守恒：
    dC1/dt += β·P²（塔蒂尼项产生的 C1 增量）

═══════════════════════════════════════════════════════════════════
  验证目标
═══════════════════════════════════════════════════════════════════

  1. 相位锁定验证：β·P² 项是否导致 C1 的局部聚集（空间分化）？
     指标：C_var 是否在 β > 0 时显著高于 β = 0？

  2. 时间箭头验证：C1 的聚集是否不可逆？
     指标：一旦 C1 聚集形成，是否会自发散开回到 P？
     观测：C_var 的振荡是否具有方向性（单调爬升 vs 对称振荡）？

  3. 纠缠拓扑验证：C1 聚集是否与 I* 场（stock_state_C1）的梯度相关？
     指标：C1 的空间分布是否与 I_macro 的高梯度区域重合？

  4. 量子-经典过渡验证：β 从 0 增大时，系统是否从"均匀扩散"
     过渡到"局部坍缩"再到"宏观结构锁定"？

运行依赖：numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.ndimage import convolve, label, uniform_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings, pathlib
from datetime import datetime
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR  = pathlib.Path(__file__).parent
_SCRIPT_NAME = pathlib.Path(__file__).stem
_RUN_TIME    = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR   = _SCRIPT_DIR / _SCRIPT_NAME / _RUN_TIME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 一、参数配置
# ─────────────────────────────────────────────

class Config:
    # ── 网格 ──────────────────────────────────────────────────────────
    W, H   = 128, 128
    dt     = 0.05
    steps  = 24000   # 日常运行默认值

    # ── 扩散系数 ──────────────────────────────────────────────────────
    D_P    = 0.20
    D_C1   = 0.05
    D_C2   = 0.004

    # ── 传承失真 ──────────────────────────────────────────────────────
    k_noise   = 0.50
    blur_size = 5

    # ── 拓扑清洗 ──────────────────────────────────────────────────────
    I_max_capacity = 1.0
    D_decay        = 0.15
    gauss_size     = 5

    # ── 跃迁势垒与速率（铁律参数）────────────────────────────────────
    Theta_barrier  = 0.28
    Base_Rate_1    = 1.0
    Base_Rate_2    = 0.4
    Base_Rate_dec  = 1.2
    decay_C1       = 0.06
    decay_C2       = 0.008

    # ── 熵产自适应退化 ────────────────────────────────────────────────
    k_entropy = 0.01

    # ── 自适应调参 ────────────────────────────────────────────────────
    adaptive_tuning = True
    adapt_interval  = 50

    # ── 观测 ──────────────────────────────────────────────────────────
    render_every   = 20     # 每 20 帧生成一张图片
    struct_every   = 20
    phi_threshold  = None
    hd_window      = 300
    hd_drive_eps   = 5e-5

    # ── [v7.0 继承] 广义算子参数 ──────────────────────────────────────
    w_rate_P   = 0.1
    w_rate_C1  = 0.7
    w_rate_C2  = 1.5
    w_state_C1 = 0.15
    w_state_C2 = 0.25

    lam_rate_P   = 0.15
    lam_rate_C1  = 0.04
    lam_rate_C2  = 0.005
    lam_state_C1 = 0.005
    lam_state_C2 = 0.003

    kw_rate_P   = 0.001
    kw_rate_C1  = 0.005
    kw_rate_C2  = 0.0005
    kw_state_C1 = 0.05
    kw_state_C2 = 0.05

    k_norm_base   = 0.002
    kappa_inertia = 0.3

    # ── [v7.0 继承] 情境自适应遗忘率 ─────────────────────────────────
    alpha_lambda  = 1.5
    lam_min_ratio = 0.1

    # ── [v7.0 继承] 零残差拓扑热汇 H_sink ────────────────────────────
    theta_sink  = 0.03
    gamma_sink  = 0.02
    sink_window = 5

    # ── [v7.0 继承] C2 全局不稳定机制（回差法）──────────────────────
    k_instab         = 0.005
    theta_instab_low = 0.005
    theta_instab_high= 0.020

    # ── [v8.0 继承] 涌现度规 ─────────────────────────────────────────
    alpha_metric    = 0.3
    alpha_metric_C1 = 0.0

    # ── [v8.0 继承] 早期图灵种子扰动 ─────────────────────────────────
    seed_enabled    = True
    seed_fc2_thresh = 0.40
    seed_strength   = 0.002
    seed_freq       = 2

    # ── [v9.0 继承] 种子扰动自适应强度 ──────────────────────────────
    seed_adaptive       = True
    seed_cvar_low       = 0.005
    seed_cvar_high      = 0.030

    # ── [v9.0 继承] 低谷期自适应势垒 ─────────────────────────────────
    trough_adapt_enabled  = True
    trough_fc2_thresh     = 0.20
    trough_phi_thresh     = 25.0
    trough_decay_rate     = 0.9999
    trough_theta_min      = 0.18

    # ── [v10.0 继承] 通道二：低谷期失真增强 ──────────────────────────
    distortion_boost_enabled = True
    distortion_boost_max     = 0.20

    # ── [v11.0 继承] 振荡计数器 ──────────────────────────────────────
    osc_counter_enabled   = True
    osc_count_thresh      = 3
    osc_cvar_window       = 500
    osc_cvar_floor        = 0.03

    # ── [v11.0 继承] 双阈值扩展 ──────────────────────────────────────
    dual_thresh_enabled      = True
    trough_fc2_thresh_ext    = 0.28
    trough_phi_thresh_ext    = 20.0
    trough_decay_rate_ext    = 0.9999
    trough_ext_min_duration  = 2000
    trough_ext_cooldown      = 50000

    # ── [v12.0 新增] 塔蒂尼非线性交叉项 ─────────────────────────────
    # 物理意义：P 场的自相互作用，模拟量子真空涨落的非线性耦合
    # 数学形式：dP/dt -= beta_tartini × P²
    #           dC1/dt += beta_tartini × P²  （守恒：P 损失转化为 C1）
    # 铁律对齐：
    #   - 有损算子（L）：P² 项提取 P 场的"自干涉强度"作为坍缩驱动力
    #   - 失真算子（D）：非线性耦合打破 P 场的空间对称性，制造相位锁定
    # 验证目标：
    #   - beta=0：纯线性扩散（对照组，无塔蒂尼效应）
    #   - beta>0：非线性耦合激活，观察 C1 聚集和时间箭头涌现
    beta_tartini     = 0.001    # 日常默认（对照组）；验证用 0.001/0.005
    # 参数扫描建议：0.0 → 0.001 → 0.005 → 0.010 → 0.020
    # 注意：beta 过大会导致 P 场快速耗尽，需要配合 P 场的补充机制

    # ── [v12.0 新增] 量子观测量 ──────────────────────────────────────
    # 用于量化相位锁定程度的额外观测量
    # 局部相位锁定指数（Local Phase-Locking Index）：
    #   LPI = Var(C1) / (Var(P) + Var(C1) + eps)
    #   LPI → 0：P 场主导（量子叠加态，无坍缩）
    #   LPI → 1：C1 场主导（经典定域态，完全坍缩）
    compute_lpi = True   # 是否计算局部相位锁定指数


# ─────────────────────────────────────────────
# 二、自适应调参模块（继承 v11.0）
# ─────────────────────────────────────────────

class AdaptiveTuner:
    BOUNDS = {
        'lam_rate_C2':   (0.001, 0.015),
        'lam_rate_C1':   (0.005, 0.04),
        'Theta_barrier': (0.10,  0.40),
        'decay_C2':      (0.001, 0.020),
        'Base_Rate_2':   (0.2,   1.0),
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = []
        self._phi_net_buf = []
        self._c_var_buf   = []
        self._chaotic_cnt = 0
        self._turing_fc2_entry  = None
        self._turing_entry_tick = None
        self._last_phase = None
        self._buf_len    = 100

    def update(self, tick, phase, f_C2, f_C1, phi_net, c_var, delta_sig):
        cfg = self.cfg
        self._phi_net_buf.append(phi_net)
        self._c_var_buf.append(c_var)
        if len(self._phi_net_buf) > self._buf_len:
            self._phi_net_buf.pop(0)
            self._c_var_buf.pop(0)
        if phase == 'TURING_GROWTH' and self._last_phase != 'TURING_GROWTH':
            self._turing_fc2_entry  = f_C2
            self._turing_entry_tick = tick
        self._chaotic_cnt = self._chaotic_cnt + 1 if phase == 'CHAOTIC_EDGE' else 0
        self._last_phase  = phase
        interval = 20 if phase == 'TURING_GROWTH' else cfg.adapt_interval
        if tick % interval != 0 or tick == 0:
            return
        self._check_rules(tick, phase, f_C2, f_C1, phi_net, c_var, delta_sig)

    def _check_rules(self, tick, phase, f_C2, f_C1, phi_net, c_var, delta_sig):
        cfg = self.cfg
        if (phase == 'TURING_GROWTH' and len(self._phi_net_buf) >= 50
                and self._declining(self._phi_net_buf[-50:], 0.3)
                and self._declining(self._c_var_buf[-50:], 0.1)):
            self._apply('lam_rate_C2', cfg.lam_rate_C2,
                        max(self.BOUNDS['lam_rate_C2'][0], cfg.lam_rate_C2 * 0.80),
                        tick, '图灵期衰退预警：降低lam_rate_C2增强记忆')
        if (phase == 'TURING_GROWTH' and self._turing_entry_tick is not None
                and tick - self._turing_entry_tick > 200):
            elapsed = tick - self._turing_entry_tick
            rate = (f_C2 - (self._turing_fc2_entry or 0)) / elapsed
            if rate < 0.0001 and f_C2 < 0.05 and phi_net < 5.0:
                self._apply('Theta_barrier', cfg.Theta_barrier,
                            max(self.BOUNDS['Theta_barrier'][0], cfg.Theta_barrier * 0.90),
                            tick, f'C2积累过慢且Phi_net极低(rate={rate:.6f}/帧)，降低势垒')
        if (self._chaotic_cnt >= 300 and phi_net == 0.0 and f_C2 < 0.20):
            new = max(self.BOUNDS['lam_rate_C1'][0], cfg.lam_rate_C1 * 0.80)
            if new < cfg.lam_rate_C1 - 1e-6:
                self._apply('lam_rate_C1', cfg.lam_rate_C1, new, tick,
                            f'CHAOTIC_EDGE锁死{self._chaotic_cnt}帧，降低lam_rate_C1')
                self._chaotic_cnt = 0
        if (phase in ('UNIFORM_GROWTH', 'TURING_GROWTH') and f_C1 > 0.75
                and f_C2 < 0.05 and tick > 100):
            self._apply('Theta_barrier', cfg.Theta_barrier,
                        max(self.BOUNDS['Theta_barrier'][0], cfg.Theta_barrier * 0.85),
                        tick, f'C1过度积累(f_C1={f_C1:.3f})，降低势垒')
        if (phase == 'UNIFORM_GROWTH' and tick > 500 and c_var < 0.001
                and f_C2 > 0.1 and cfg.lam_rate_C1 < 0.03
                and (not self.log or tick - self.log[-1][0] >= 200)):
            self._apply('lam_rate_C2', cfg.lam_rate_C2,
                        min(self.BOUNDS['lam_rate_C2'][1], cfg.lam_rate_C2 * 1.10),
                        tick, f'C_var长期过低({c_var:.5f})，提高lam_rate_C2')
        if (self._chaotic_cnt >= 200 and phi_net == 0.0 and cfg.lam_rate_C2 > 0.007
                and (not self.log or tick - self.log[-1][0] >= 200)):
            self._apply('lam_rate_C2', cfg.lam_rate_C2,
                        max(0.005, cfg.lam_rate_C2 * 0.90),
                        tick, f'lam_rate_C2上调后仍CHAOTIC_EDGE，回调')

    def _declining(self, buf, threshold):
        if len(buf) < 10: return False
        mid = len(buf) // 2
        a, b = np.mean(buf[:mid]), np.mean(buf[mid:])
        return a > 1e-9 and b < a * (1 - threshold)

    def _apply(self, param, old, new, tick, reason):
        if abs(new - old) < 1e-9: return
        setattr(self.cfg, param, new)
        self.log.append((tick, param, old, new, reason))
        print(f'  [AdaptTune] tick={tick:05d}  {param}: {old:.6f} → {new:.6f}')
        print(f'              原因: {reason}')

    def summary(self):
        if not self.log:
            return '_本次运行未触发任何自适应调参_\n'
        lines = ['| tick | 参数 | 原值 | 新值 | 触发原因 |', '|---|---|---|---|---|']
        for tick, param, old, new, reason in self.log:
            lines.append(f'| {tick} | {param} | {old:.6f} | {new:.6f} | {reason} |')
        return '\n'.join(lines) + '\n'


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def make_kernel(size, kind='blur'):
    k = np.ones((size, size), dtype=np.float32)
    if kind == 'gauss':
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                k[i, j] = np.exp(-((i-cx)**2 + (j-cy)**2) / (2*(size/4)**2))
    k /= k.sum()
    return k

LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                               [1,-4, 1],
                               [0, 1, 0]], dtype=np.float32)

def lap(field):
    return convolve(field, LAPLACIAN_KERNEL, mode='wrap')

def spatial_noise(shape):
    return np.random.randn(*shape).astype(np.float32)

def softsign(x, cap):
    return x / (1.0 + np.abs(x) / cap)

def div_diff(field, D_field):
    """守恒的非均匀扩散：∇·(D(x,y)·∇C)，周期边界，sum=0"""
    D_xp = 0.5 * (D_field + np.roll(D_field, -1, axis=1))
    D_xm = 0.5 * (D_field + np.roll(D_field,  1, axis=1))
    D_yp = 0.5 * (D_field + np.roll(D_field, -1, axis=0))
    D_ym = 0.5 * (D_field + np.roll(D_field,  1, axis=0))
    flux_xp = D_xp * (np.roll(field, -1, axis=1) - field)
    flux_xm = D_xm * (field - np.roll(field,  1, axis=1))
    flux_yp = D_yp * (np.roll(field, -1, axis=0) - field)
    flux_ym = D_ym * (field - np.roll(field,  1, axis=0))
    return (flux_xp - flux_xm + flux_yp - flux_ym).astype(np.float32)

def gini(arr):
    if len(arr) == 0: return 0.0
    arr = np.sort(arr.astype(float))
    n = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * (idx * arr).sum()) / (n * arr.sum() + 1e-9) - (n + 1) / n


# ─────────────────────────────────────────────
# 三、物理层（含塔蒂尼量子层）
# ─────────────────────────────────────────────

class PhysicsLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        shape = (cfg.H, cfg.W)

        rng = np.random.default_rng(42)
        self.P  = rng.uniform(0.2, 1.0, shape).astype(np.float32)
        self.C1 = np.zeros(shape, np.float32)
        self.C2 = np.zeros(shape, np.float32)

        self.stock_rate_P   = np.zeros(shape, np.float32)
        self.stock_rate_C1  = np.zeros(shape, np.float32)
        self.stock_rate_C2  = np.zeros(shape, np.float32)
        self.stock_state_C1 = np.zeros(shape, np.float32)
        self.stock_state_C2 = np.zeros(shape, np.float32)

        self._I_macro_prev = np.zeros(shape, np.float32)

        self.P_prev  = np.maximum(self.P + cfg.D_P * lap(self.P) * cfg.dt, 0.0)
        self.C1_prev = self.C1.copy()
        self.C2_prev = self.C2.copy()

        self.blur_k  = make_kernel(cfg.blur_size, 'blur')
        self.gauss_k = make_kernel(cfg.gauss_size, 'gauss')
        self._seed_template = self._make_seed_template(cfg)

        self.I_stock         = np.zeros(shape, np.float32)
        self.I_increment     = np.zeros(shape, np.float32)
        self.I_drive         = np.zeros(shape, np.float32)
        self.A_C1_to_C2      = np.zeros(shape, np.float32)
        self.A_C2_to_C1      = np.zeros(shape, np.float32)
        self._I_norm_vis      = np.zeros(shape, np.float32)
        self.sink_mask       = np.zeros(shape, np.float32)
        self.metric_vis      = np.ones(shape, np.float32)
        self.rate_stock_mean = 0.0
        self._instab_active  = False

        # v11.0 继承
        self._trough_active  = False
        self.k_noise_eff     = cfg.k_noise
        self._cvar_window_buf  = []
        self._cvar_above_mean  = False
        self._osc_count        = 0
        self._osc_confirmed    = False
        self.osc_count_vis     = 0
        self.dual_thresh_active = False
        self._ext_below_count  = 0
        self._ext_cooldown_remaining = 0

        # [v12.0 新增] 塔蒂尼量子层观测量
        self.A_tartini       = np.zeros(shape, np.float32)  # 塔蒂尼项（P²）
        self.lpi             = 0.0   # 局部相位锁定指数（Local Phase-Locking Index）
        self.tartini_total   = 0.0   # 累积塔蒂尼通量（量化相位锁定总量）

    def _make_seed_template(self, cfg: Config):
        H, W = cfg.H, cfg.W
        x = np.linspace(0, 2 * np.pi * cfg.seed_freq, W, endpoint=False)
        y = np.linspace(0, 2 * np.pi * cfg.seed_freq, H, endpoint=False)
        xx, yy = np.meshgrid(x, y)
        template = (np.sin(xx) + np.sin(yy) +
                    np.sin(xx + yy) * 0.5 +
                    np.sin(xx - yy) * 0.5).astype(np.float32)
        template /= (np.abs(template).max() + 1e-9)
        return template

    def step(self, tick: int = 0, c_var: float = 0.0, phase: str = 'CHAOTIC_EDGE'):
        cfg = self.cfg
        P, C1, C2 = self.P, self.C1, self.C2

        # ── 步骤0：情境遗忘率场 ──────────────────────────────────────
        if cfg.alpha_lambda > 0:
            gx = np.gradient(self._I_macro_prev, axis=1).astype(np.float32)
            gy = np.gradient(self._I_macro_prev, axis=0).astype(np.float32)
            grad_mag = np.sqrt(gx**2 + gy**2)
            ctx_factor = np.exp(-cfg.alpha_lambda * grad_mag /
                                (self._I_macro_prev + 1e-6)).astype(np.float32)
        else:
            ctx_factor = np.ones((cfg.H, cfg.W), np.float32)

        def lam_field(lam_base):
            return np.maximum(lam_base * ctx_factor, lam_base * cfg.lam_min_ratio)

        lam_rP  = lam_field(cfg.lam_rate_P)
        lam_rC1 = lam_field(cfg.lam_rate_C1)
        lam_rC2 = lam_field(cfg.lam_rate_C2)
        lam_sC1 = lam_field(cfg.lam_state_C1)
        lam_sC2 = lam_field(cfg.lam_state_C2)

        # ── 步骤1：特征张量提取 ──────────────────────────────────────
        chi_rate_P  = np.abs(P  - self.P_prev)
        chi_rate_C1 = np.abs(C1 - self.C1_prev)
        chi_rate_C2 = np.abs(C2 - self.C2_prev)
        chi_state_C1 = C1
        chi_state_C2 = C2

        # ── 步骤2：H_sink 极化检测 ───────────────────────────────────
        C2_lmean = uniform_filter(C2, size=cfg.sink_window, mode='wrap')
        C2_lm2   = uniform_filter(C2**2, size=cfg.sink_window, mode='wrap')
        C2_lvar  = np.maximum(C2_lm2 - C2_lmean**2, 0.0)
        raw_mask = (C2_lvar > cfg.theta_sink).astype(np.float32)
        sink_mask = convolve(raw_mask, make_kernel(3, 'gauss'), mode='wrap')
        sink_mask = np.clip(sink_mask, 0.0, 1.0)
        self.sink_mask = sink_mask

        chi_rate_C2_eff = chi_rate_C2 * (1.0 - sink_mask)
        chi_rate_C1_eff = chi_rate_C1 * (1.0 - sink_mask * 0.5)
        A_C2_sink = cfg.gamma_sink * sink_mask * C2

        # ── 步骤3：L 有损算子 ────────────────────────────────────────
        I_macro          = np.zeros_like(P)
        I_macro_rate_sum = np.zeros_like(P)

        features = [
            (chi_rate_P,    'stock_rate_P',   cfg.kw_rate_P,   lam_rP,  cfg.w_rate_P,   True),
            (chi_rate_C1_eff,'stock_rate_C1', cfg.kw_rate_C1,  lam_rC1, cfg.w_rate_C1,  True),
            (chi_rate_C2_eff,'stock_rate_C2', cfg.kw_rate_C2,  lam_rC2, cfg.w_rate_C2,  True),
            (chi_state_C1,  'stock_state_C1', cfg.kw_state_C1, lam_sC1, cfg.w_state_C1, False),
            (chi_state_C2,  'stock_state_C2', cfg.kw_state_C2, lam_sC2, cfg.w_state_C2, False),
        ]

        for chi, stock_name, k_w, lam_arr, weight, is_rate in features:
            stock = getattr(self, stock_name)
            p_write = chi / (chi + k_w)
            mask = (np.random.rand(*P.shape) < p_write).astype(np.float32)
            stock[:] = stock * (1.0 - lam_arr) + chi * mask * cfg.dt
            norm_stock = stock / (stock + cfg.k_norm_base)
            I_macro += weight * norm_stock
            if is_rate:
                I_macro_rate_sum += weight * norm_stock

        self._I_macro_prev = I_macro.copy()
        self.I_stock = (self.stock_rate_P + self.stock_rate_C1 + self.stock_rate_C2 +
                        self.stock_state_C1 + self.stock_state_C2)
        self.I_increment = (chi_rate_P * cfg.w_rate_P +
                            chi_rate_C1 * cfg.w_rate_C1 +
                            chi_rate_C2 * cfg.w_rate_C2)
        self.rate_stock_mean = float(
            (self.stock_rate_P + self.stock_rate_C1 + self.stock_rate_C2).mean())

        # ── 步骤4：D 失真算子（含通道二失真增强）────────────────────
        k_noise_eff = cfg.k_noise
        if cfg.distortion_boost_enabled and self._trough_active:
            M_total = P.sum() + C1.sum() + C2.sum()
            f_C2_now = C2.sum() / (M_total + 1e-9)
            eff_thresh = (cfg.trough_fc2_thresh_ext if self.dual_thresh_active
                          else cfg.trough_fc2_thresh)
            if f_C2_now < eff_thresh:
                boost_factor = 1.0 - f_C2_now / eff_thresh
                k_noise_eff = cfg.k_noise + cfg.distortion_boost_max * boost_factor
        self.k_noise_eff = k_noise_eff

        I_context   = convolve(I_macro, self.blur_k, mode='wrap')
        S_norm_C1   = self.stock_state_C1 / (self.stock_state_C1 + cfg.k_norm_base)
        S_norm_C2   = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        S_macro     = cfg.w_state_C1 * S_norm_C1 + cfg.w_state_C2 * S_norm_C2
        laplacian_S = lap(S_macro)
        noise       = spatial_noise(P.shape) * k_noise_eff * I_context
        I_distorted = I_context + cfg.kappa_inertia * laplacian_S + noise

        # ── 步骤5：柔性极化与死区清洗 ────────────────────────────────
        Omega_raw    = softsign(I_distorted, cfg.I_max_capacity)
        Omega_damped = (1.0 - cfg.D_decay) * convolve(Omega_raw, self.gauss_k, mode='wrap')
        I_drive      = np.where(np.abs(Omega_damped) < 1e-4, 0.0, Omega_damped)
        self.I_drive     = I_drive
        self._I_norm_vis = I_macro / (I_macro.max() + 1e-9)

        # ── 步骤6：涌现度规 ──────────────────────────────────────────
        S_C2_norm = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        S_C1_norm = self.stock_state_C1 / (self.stock_state_C1 + cfg.k_norm_base)
        metric_factor = np.maximum(0.1,
            1.0 - cfg.alpha_metric * S_C2_norm - cfg.alpha_metric_C1 * S_C1_norm)
        D_C1_eff = cfg.D_C1 * metric_factor
        D_C2_eff = cfg.D_C2 * metric_factor
        self.metric_vis = metric_factor

        # ── 步骤7：形态更新（能-信二元 PDE）─────────────────────────
        A_P_to_C1  = np.maximum(0,  I_drive) * cfg.Base_Rate_1 * P
        A_C1_to_P  = np.abs(np.minimum(0, I_drive)) * cfg.Base_Rate_1 * C1
        A_C1_to_C2 = np.maximum(0, I_drive - cfg.Theta_barrier) * cfg.Base_Rate_2 * C1
        A_C2_to_C1 = np.abs(np.minimum(0, I_drive)) * cfg.Base_Rate_dec * C2

        A_C1_decay = cfg.decay_C1 * C1
        decay_C2_dyn = cfg.decay_C2 + cfg.k_entropy * I_macro_rate_sum.mean()
        A_C2_decay   = decay_C2_dyn * C2

        self.A_C1_to_C2 = A_C1_to_C2
        self.A_C2_to_C1 = A_C2_to_C1

        # ── [v12.0 新增] 塔蒂尼非线性交叉项 ─────────────────────────
        # 物理意义：P 场的自相互作用 → 相位锁定 → 波函数坍缩
        # 数学形式：A_tartini = beta × P²
        # 守恒性：从 dP/dt 中减去，等量加入 dC1/dt
        # 铁律对齐：
        #   有损算子（L）：P² 提取 P 场的"自干涉强度"（宏观拓扑趋势）
        #   失真算子（D）：非线性耦合打破 P 场空间对称性（对称性破缺）
        if cfg.beta_tartini > 0:
            A_tartini = cfg.beta_tartini * P * P  # P² 项（守恒：P 损失 = C1 增益）
            A_tartini = np.maximum(A_tartini, 0.0)  # 非负（只能从 P 流向 C1）
        else:
            A_tartini = np.zeros_like(P)
        self.A_tartini = A_tartini
        self.tartini_total = float(A_tartini.sum())

        self.P_prev  = P.copy()
        self.C1_prev = C1.copy()
        self.C2_prev = C2.copy()

        # 记录更新前总质量（用于截断后守恒修正）
        M_before = float(P.sum() + C1.sum() + C2.sum())

        dP_dt  = (cfg.D_P  * lap(P)
                  - A_P_to_C1 + A_C1_to_P + A_C1_decay + A_C2_sink
                  - A_tartini)                          # [v12.0] 塔蒂尼项：P 损失

        dC1_dt = (div_diff(C1, D_C1_eff)
                  + A_P_to_C1 - A_C1_to_P - A_C1_to_C2 + A_C2_to_C1
                  - A_C1_decay + A_C2_decay
                  + A_tartini)                          # [v12.0] 塔蒂尼项：C1 增益

        dC2_dt = (div_diff(C2, D_C2_eff)
                  + A_C1_to_C2 - A_C2_to_C1 - A_C2_decay - A_C2_sink)

        # 全局回差法
        C2_var = float(C2.var())
        if C2_var < cfg.theta_instab_low:
            self._instab_active = True
        elif C2_var > cfg.theta_instab_high:
            self._instab_active = False
        if self._instab_active and C2.mean() > 1e-6:
            noise_instab = spatial_noise(C2.shape)
            A_instab = cfg.k_instab * C2 * noise_instab
            A_instab -= A_instab.sum() / A_instab.size
            dC2_dt = dC2_dt + A_instab

        self.P  = np.maximum(P  + dP_dt  * cfg.dt, 0.0)
        self.C1 = np.maximum(C1 + dC1_dt * cfg.dt, 0.0)
        self.C2 = np.maximum(C2 + dC2_dt * cfg.dt, 0.0)

        # 截断后守恒修正（继承 v11.0 T-007 修复）
        M_after = float(self.P.sum() + self.C1.sum() + self.C2.sum())
        if abs(M_after - M_before) > 0.001 and M_after > 1e-9:
            scale_factor = M_before / M_after
            self.P  *= scale_factor
            self.C1 *= scale_factor
            self.C2 *= scale_factor

        # ── [v12.0 新增] 局部相位锁定指数（LPI）────────────────────
        # LPI = Var(C1) / (Var(P) + Var(C1) + eps)
        # LPI → 0：P 场主导（量子叠加态）
        # LPI → 1：C1 场主导（经典定域态，相位锁定完成）
        if cfg.compute_lpi:
            var_P  = float(self.P.var())
            var_C1 = float(self.C1.var())
            self.lpi = var_C1 / (var_P + var_C1 + 1e-9)
        else:
            self.lpi = 0.0

        # ── 步骤8：自适应种子扰动（继承 v9.0）───────────────────────
        if cfg.seed_enabled:
            M_total = self.P.sum() + self.C1.sum() + self.C2.sum()
            f_C2_now = self.C2.sum() / (M_total + 1e-9)
            if f_C2_now < cfg.seed_fc2_thresh and self.C2.mean() > 1e-6:
                if cfg.seed_adaptive:
                    cvar_range = cfg.seed_cvar_high - cfg.seed_cvar_low
                    cvar_factor = max(0.0, (cfg.seed_cvar_high - c_var) / (cvar_range + 1e-9))
                    cvar_factor = min(1.0, cvar_factor)
                else:
                    cvar_factor = 1.0
                fc2_factor = 1.0 - f_C2_now / cfg.seed_fc2_thresh
                strength = cfg.seed_strength * fc2_factor * cvar_factor
                if strength > 1e-6:
                    C2_sum_before = self.C2.sum()
                    scale = 1.0 + self._seed_template * strength
                    scale = np.maximum(scale, 0.01)
                    C2_new = self.C2 * scale
                    self.C2 = C2_new * (C2_sum_before / (C2_new.sum() + 1e-9))

        # ── 步骤9：振荡计数器 + 双阈值扩展（继承 v11.0）────────────
        if phase == 'TURING_GROWTH':
            M_total = self.P.sum() + self.C1.sum() + self.C2.sum()
            f_C2_now = self.C2.sum() / (M_total + 1e-9)
            phi_net_now = float(self.A_C1_to_C2.sum() - self.A_C2_to_C1.sum())

            if cfg.osc_counter_enabled and c_var > cfg.osc_cvar_floor:
                self._cvar_window_buf.append(c_var)
                if len(self._cvar_window_buf) > cfg.osc_cvar_window:
                    self._cvar_window_buf.pop(0)
                if len(self._cvar_window_buf) >= cfg.osc_cvar_window // 2:
                    cvar_mean = float(np.mean(self._cvar_window_buf))
                    was_above = self._cvar_above_mean
                    is_above  = c_var > cvar_mean
                    if is_above and not was_above:
                        self._osc_count += 1
                    self._cvar_above_mean = is_above
                if self._osc_count >= cfg.osc_count_thresh:
                    self._osc_confirmed = True
            self.osc_count_vis = self._osc_count

            trough_condition_base = (cfg.trough_adapt_enabled
                                     and f_C2_now < cfg.trough_fc2_thresh
                                     and phi_net_now > cfg.trough_phi_thresh
                                     and cfg.Theta_barrier > cfg.trough_theta_min)

            if self._ext_cooldown_remaining > 0:
                self._ext_cooldown_remaining -= 1

            if (cfg.dual_thresh_enabled and self._osc_confirmed
                    and c_var > cfg.osc_cvar_floor
                    and f_C2_now < cfg.trough_fc2_thresh_ext
                    and phi_net_now > cfg.trough_phi_thresh_ext
                    and self._ext_cooldown_remaining == 0):
                self._ext_below_count += 1
            else:
                self._ext_below_count = 0

            trough_condition_ext = (self._ext_below_count >= cfg.trough_ext_min_duration
                                    and cfg.Theta_barrier > cfg.trough_theta_min)

            self.dual_thresh_active = trough_condition_ext and not trough_condition_base
            self._trough_active = trough_condition_base or trough_condition_ext

            if trough_condition_base:
                cfg.Theta_barrier = max(cfg.trough_theta_min,
                                        cfg.Theta_barrier * cfg.trough_decay_rate)
            elif trough_condition_ext:
                cfg.Theta_barrier = max(cfg.trough_theta_min,
                                        cfg.Theta_barrier * cfg.trough_decay_rate_ext)
                self._ext_below_count = 0
                self._ext_cooldown_remaining = cfg.trough_ext_cooldown
        else:
            self._cvar_window_buf.clear()
            self._cvar_above_mean = False
            self._ext_below_count = 0
            self._trough_active = False
            self.dual_thresh_active = False


# ─────────────────────────────────────────────
# 四、观测层（继承 v11.0，新增量子观测量）
# ─────────────────────────────────────────────

class ObservationLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = {k: [] for k in [
            'tick', 'M_total', 'f_C2', 'f_C1', 'f_P',
            'delta_sigma', 'Phi_net', 'C_var', 'H_C2',
            'n_clusters', 'size_gini', 'phase', 'Theta_barrier',
            'k_noise_eff', 'osc_count', 'dual_thresh_active',
            # [v12.0 新增] 量子观测量
            'lpi',            # 局部相位锁定指数
            'tartini_flux',   # 塔蒂尼通量（P→C1 的非线性转化率）
            'tartini_cumul',  # 累积塔蒂尼通量
        ]}
        self._hd_buf = []
        self._tartini_cumul = 0.0

    def observe(self, phys: PhysicsLayer, tick: int):
        cfg = self.cfg
        P, C1, C2 = phys.P, phys.C1, phys.C2
        eps = 1e-9

        M_P, M_C1, M_C2 = P.sum(), C1.sum(), C2.sum()
        M_total = M_P + M_C1 + M_C2
        f_C2 = M_C2 / (M_total + eps)
        f_C1 = M_C1 / (M_total + eps)
        f_P  = M_P  / (M_total + eps)

        sig_inc  = phys.I_increment.sum()
        sig_loss = cfg.lam_rate_C2 * phys.rate_stock_mean * P.size
        delta_sig = sig_inc - sig_loss

        phi_C2 = C2 / (P + C1 + C2 + eps)
        H_C2   = -(phi_C2 * np.log(phi_C2 + eps)).sum()
        C_var  = phi_C2.var()

        Phi_net = phys.A_C1_to_C2.sum() - phys.A_C2_to_C1.sum()
        phase   = self._classify(delta_sig, Phi_net, C_var, f_C2)

        n_cl, s_gini = 0, 0.0
        if tick % cfg.struct_every == 0:
            n_cl, s_gini = self._struct_stats(phi_C2)

        self._hd_buf.append((phys.I_increment.max(), abs(phys.I_drive).max()))
        if len(self._hd_buf) > cfg.hd_window:
            self._hd_buf.pop(0)

        # 累积塔蒂尼通量
        self._tartini_cumul += phys.tartini_total * cfg.dt

        h = self.history
        h['tick'].append(tick)
        h['M_total'].append(M_total)
        h['f_C2'].append(f_C2)
        h['f_C1'].append(f_C1)
        h['f_P'].append(f_P)
        h['delta_sigma'].append(delta_sig)
        h['Phi_net'].append(Phi_net)
        h['C_var'].append(C_var)
        h['H_C2'].append(H_C2)
        h['n_clusters'].append(n_cl)
        h['size_gini'].append(s_gini)
        h['phase'].append(phase)
        h['Theta_barrier'].append(cfg.Theta_barrier)
        h['k_noise_eff'].append(phys.k_noise_eff)
        h['osc_count'].append(phys.osc_count_vis)
        h['dual_thresh_active'].append(int(phys.dual_thresh_active))
        h['lpi'].append(phys.lpi)
        h['tartini_flux'].append(phys.tartini_total)
        h['tartini_cumul'].append(self._tartini_cumul)
        return phase, C_var

    @property
    def is_dead(self):
        if len(self._hd_buf) < self.cfg.hd_window:
            return False
        inc_vals   = [v[0] for v in self._hd_buf]
        drive_vals = [v[1] for v in self._hd_buf]
        return max(inc_vals) < 1e-4 and max(drive_vals) < self.cfg.hd_drive_eps

    def _classify(self, ds, pn, cv, fc2):
        if ds > 0 and pn > 0:
            return 'TURING_GROWTH' if cv > 0.01 else 'UNIFORM_GROWTH'
        elif ds > 0 and pn < 0:
            return 'RESTRUCTURING'
        elif ds < 0 and pn < 0:
            return 'COLLAPSE'
        elif ds < 0 and pn > 0 and fc2 > 0.2:
            return 'COARSENING'
        elif abs(ds) < 1e-3 and abs(pn) < 1e-3:
            return 'NEAR_EQUILIBRIUM'
        return 'CHAOTIC_EDGE'

    def _struct_stats(self, phi_C2):
        threshold = (phi_C2.mean() + 0.5 * phi_C2.std()
                     if self.cfg.phi_threshold is None else self.cfg.phi_threshold)
        binary = (phi_C2 > threshold).astype(int)
        labeled, n = label(binary)
        if n == 0:
            return 0, 0.0
        sizes = np.array([(labeled == i).sum() for i in range(1, n+1)])
        return n, gini(sizes)


# ─────────────────────────────────────────────
# 五、可视化层
# ─────────────────────────────────────────────

PHASE_COLORS = {
    'TURING_GROWTH':    '#2ecc71',
    'UNIFORM_GROWTH':   '#a8e6a3',
    'RESTRUCTURING':    '#e67e22',
    'COLLAPSE':         '#e74c3c',
    'COARSENING':       '#9b59b6',
    'NEAR_EQUILIBRIUM': '#95a5a6',
    'CHAOTIC_EDGE':     '#f1c40f',
}

def render(phys: PhysicsLayer, obs: ObservationLayer, tick: int):
    cfg = phys.cfg
    P, C1, C2 = phys.P, phys.C1, phys.C2
    eps = 1e-9
    phi_C2 = C2 / (P + C1 + C2 + eps)

    fig = plt.figure(figsize=(18, 10), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(2, 5, figure=fig,
                            hspace=0.35, wspace=0.3,
                            left=0.04, right=0.97,
                            top=0.92, bottom=0.08)

    # 上排：5 个空间场
    fields = [
        (P,                'hot',     None,            'P 场（概率波函数）\n量子叠加态'),
        (C1,               'Blues',   None,            'C1 场（坍缩定域态）\n相位锁定产物'),
        (phi_C2,           'viridis', None,            'φ_C2  高阶结构密度'),
        (phys.A_tartini,   'plasma',  None,            '[v12] 塔蒂尼通量 β·P²\n相位锁定强度'),
        (phys._I_norm_vis, 'cividis', None,            'I_macro  宏观因果场'),
    ]
    for col, (data, cmap, norm, title) in enumerate(fields):
        ax = fig.add_subplot(gs[0, col])
        kw = dict(cmap=cmap, interpolation='nearest', aspect='auto')
        if norm is not None:
            kw['norm'] = norm
        im = ax.imshow(data, **kw)
        ax.set_title(title, color='white', fontsize=7, pad=3)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(
            labelcolor='white', labelsize=5)

    h = obs.history
    ticks = h['tick']
    _n = len(ticks)
    if _n > 1000:
        _step = _n // 1000
        _idx  = list(range(0, _n, _step)) + [_n - 1]
        ticks_plot = [ticks[i] for i in _idx]
        h_plot = {k: [h[k][i] for i in _idx] for k in
                  ['f_C2', 'f_C1', 'f_P', 'C_var', 'Phi_net', 'Theta_barrier',
                   'lpi', 'tartini_flux', 'tartini_cumul', 'phase']}
    else:
        ticks_plot = ticks
        h_plot = h

    # 下排左：主时序曲线
    ax_ts = fig.add_subplot(gs[1, :3])
    ax_ts.set_facecolor('#1a1a1a')
    for key, color, lbl in [
        ('f_P',           '#3498db', 'f_P  概率波函数占比（量子叠加态）'),
        ('f_C1',          '#e67e22', 'f_C1  坍缩定域态占比'),
        ('f_C2',          '#2ecc71', 'f_C2  高阶结构占比'),
        ('C_var',         '#e74c3c', 'C_var  空间分化度'),
        ('Theta_barrier', '#f39c12', 'Θ  势垒'),
        ('lpi',           '#9b59b6', '[v12] LPI  相位锁定指数'),
    ]:
        ax_ts.plot(ticks_plot, h_plot[key], color=color, lw=1.0, label=lbl)

    if len(ticks_plot) > 1:
        phases_plot = h_plot['phase']
        seg_start, seg_phase = ticks_plot[0], phases_plot[0]
        for i in range(1, len(ticks_plot)):
            if phases_plot[i] != seg_phase:
                ax_ts.axvspan(seg_start, ticks_plot[i], alpha=0.10,
                              color=PHASE_COLORS.get(seg_phase, '#ffffff'))
                seg_start, seg_phase = ticks_plot[i], phases_plot[i]
        ax_ts.axvspan(seg_start, ticks_plot[-1], alpha=0.10,
                      color=PHASE_COLORS.get(seg_phase, '#ffffff'))

    ax_ts.axhline(0, color='white', lw=0.4, alpha=0.4)
    ax_ts.legend(loc='upper left', fontsize=6,
                 facecolor='#1a1a1a', labelcolor='white', framealpha=0.6)
    ax_ts.tick_params(colors='white', labelsize=7)
    ax_ts.set_xlabel('帧数 (tick)', color='white', fontsize=8)
    ax_ts.set_title('量子-经典演化时序  （背景色带 = 当前演化阶段）',
                    color='#aaaaaa', fontsize=8, pad=4)
    for spine in ax_ts.spines.values():
        spine.set_edgecolor('#444')

    # 下排右：塔蒂尼通量
    ax_q = fig.add_subplot(gs[1, 3])
    ax_q.set_facecolor('#1a1a1a')
    ax_q.plot(ticks_plot, h_plot['tartini_flux'], color='#e91e63', lw=1.0,
              label='塔蒂尼通量 β·P²')
    ax_q2 = ax_q.twinx()
    ax_q2.plot(ticks_plot, h_plot['tartini_cumul'], color='#ff9800', lw=1.0,
               linestyle='--', label='累积通量')
    ax_q2.tick_params(colors='#ff9800', labelsize=6)
    ax_q.legend(loc='upper left', fontsize=6,
                facecolor='#1a1a1a', labelcolor='white', framealpha=0.6)
    ax_q.tick_params(colors='white', labelsize=7)
    ax_q.set_title('[v12] 塔蒂尼相位锁定通量', color='#aaaaaa', fontsize=8, pad=4)
    for spine in ax_q.spines.values():
        spine.set_edgecolor('#444')

    # 下排最右：LPI 分布
    ax_lpi = fig.add_subplot(gs[1, 4])
    ax_lpi.set_facecolor('#1a1a1a')
    ax_lpi.plot(ticks_plot, h_plot['lpi'], color='#9b59b6', lw=1.0)
    ax_lpi.axhline(0.5, color='white', lw=0.5, alpha=0.5, linestyle='--')
    ax_lpi.set_ylim(0, 1)
    ax_lpi.tick_params(colors='white', labelsize=7)
    ax_lpi.set_title('[v12] LPI 相位锁定指数\n0=量子叠加  1=经典定域', color='#aaaaaa', fontsize=7, pad=4)
    for spine in ax_lpi.spines.values():
        spine.set_edgecolor('#444')

    phase_now = h['phase'][-1] if h['phase'] else '—'
    color_now = PHASE_COLORS.get(phase_now, 'white')
    lpi_now   = h['lpi'][-1] if h['lpi'] else 0.0
    tartini_now = h['tartini_flux'][-1] if h['tartini_flux'] else 0.0
    fig.suptitle(
        f'涌积态宇宙 v12.0（塔蒂尼量子层）   第 {tick:05d} 帧   '
        f'β={cfg.beta_tartini:.4f}   '
        f'LPI={lpi_now:.3f}   '
        f'塔蒂尼通量={tartini_now:.2e}   '
        f'f_P={h["f_P"][-1]:.3f}   f_C1={h["f_C1"][-1]:.3f}',
        color=color_now, fontsize=10, fontweight='bold'
    )

    plt.savefig(OUTPUT_DIR / f'frame_{tick:05d}.png',
                dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  [render] tick={tick:05d}  β={cfg.beta_tartini:.4f}  '
          f'LPI={lpi_now:.3f}  tartini={tartini_now:.2e}  '
          f'f_P={h["f_P"][-1]:.3f}')


# ─────────────────────────────────────────────
# 六、主循环
# ─────────────────────────────────────────────

def run():
    cfg   = Config()
    phys  = PhysicsLayer(cfg)
    obs   = ObservationLayer(cfg)
    tuner = AdaptiveTuner(cfg) if cfg.adaptive_tuning else None

    print("=" * 70)
    print("涌积态宇宙模型 v12.0 — 塔蒂尼相位锁定量子层验证")
    print(f"  网格: {cfg.W}×{cfg.H}   dt={cfg.dt}   最大帧数={cfg.steps}")
    print(f"  [v12.0新增] beta_tartini={cfg.beta_tartini}  "
          f"（0=对照组，>0=塔蒂尼非线性耦合激活）")
    print(f"  量子映射：P=概率波函数  C1=坍缩定域态  C2=退相干经典实在")
    print(f"  塔蒂尼项：dP/dt -= beta*P^2  dC1/dt += beta*P^2  （守恒）")
    print(f"  验证目标：LPI↑ = 相位锁定增强  tartini_cumul↑ = 坍缩总量")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    c_var_current = 0.0
    phase_current = 'CHAOTIC_EDGE'
    for tick in range(cfg.steps):
        phys.step(tick, c_var_current, phase_current)
        phase, c_var_current = obs.observe(phys, tick)
        phase_current = phase

        if tuner is not None:
            h = obs.history
            tuner.update(
                tick=tick, phase=phase,
                f_C2=h['f_C2'][-1], f_C1=h['f_C1'][-1],
                phi_net=h['Phi_net'][-1], c_var=h['C_var'][-1],
                delta_sig=h['delta_sigma'][-1],
            )

        if tick % cfg.render_every == 0:
            render(phys, obs, tick)

        if tick % 500 == 0 and tick > 0:
            h = obs.history
            pct = tick / cfg.steps * 100
            print(f'  [progress] {pct:5.1f}%  tick={tick:06d}  '
                  f'phase={h["phase"][-1]:<18s}  '
                  f'f_P={h["f_P"][-1]:.3f}  '
                  f'f_C1={h["f_C1"][-1]:.3f}  '
                  f'f_C2={h["f_C2"][-1]:.3f}  '
                  f'C_var={h["C_var"][-1]:.4f}  '
                  f'LPI={h["lpi"][-1]:.3f}  '
                  f'tartini={h["tartini_flux"][-1]:.2e}')

        if obs.is_dead:
            print(f"\n[HEAT DEATH] tick={tick}  系统进入热寂态，终止模拟。")
            render(phys, obs, tick)
            break

    print("\n模拟结束。")
    _print_summary(obs, tuner)


def _print_summary(obs: ObservationLayer, tuner=None):
    h = obs.history
    if not h['tick']:
        return
    print("\n── 量子层演化摘要 ────────────────────────────────")
    print(f"  总帧数:          {h['tick'][-1]}")
    print(f"  最终 f_P:        {h['f_P'][-1]:.4f}  （概率波函数残余）")
    print(f"  最终 f_C1:       {h['f_C1'][-1]:.4f}  （坍缩定域态）")
    print(f"  最终 f_C2:       {h['f_C2'][-1]:.4f}  （退相干经典实在）")
    print(f"  最终 LPI:        {h['lpi'][-1]:.4f}  （相位锁定指数）")
    print(f"  累积塔蒂尼通量:  {h['tartini_cumul'][-1]:.4f}  （总坍缩量）")
    print(f"  C_var 峰值:      {max(h['C_var']):.5f}")
    print(f"  最终阶段:        {h['phase'][-1]}")
    print(f"  最终 Theta:      {h['Theta_barrier'][-1]:.4f}")
    m0, m1 = h['M_total'][0], h['M_total'][-1]
    drift = abs(m1 - m0) / (m0 + 1e-9) * 100
    print(f"  质量守恒漂移:    {drift:.6f}%")
    if tuner and tuner.log:
        print(f"  自适应调参次数:  {len(tuner.log)}")
    print("──────────────────────────────────────────────────")
    _save_report(obs, tuner)


def _save_report(obs: ObservationLayer, tuner=None):
    h   = obs.history
    cfg = obs.cfg
    if not h['tick']:
        return

    ticks     = np.array(h['tick'])
    f_c2      = np.array(h['f_C2'])
    f_c1      = np.array(h['f_C1'])
    f_p       = np.array(h['f_P'])
    c_var     = np.array(h['C_var'])
    lpi_arr   = np.array(h['lpi'])
    tartini_f = np.array(h['tartini_flux'])
    tartini_c = np.array(h['tartini_cumul'])
    theta_arr = np.array(h['Theta_barrier'])
    phases    = h['phase']

    total_ticks = h['tick'][-1]
    m0, m1 = h['M_total'][0], h['M_total'][-1]
    drift = abs(m1 - m0) / (m0 + 1e-9) * 100

    cvar_peak_idx = int(np.argmax(c_var))
    lpi_peak_idx  = int(np.argmax(lpi_arr))

    milestones = []
    for pct in range(0, 101, 10):
        idx = min(int(pct / 100 * (len(ticks) - 1)), len(ticks) - 1)
        milestones.append({
            'tick': int(ticks[idx]),
            'f_P': float(f_p[idx]), 'f_C1': float(f_c1[idx]), 'f_C2': float(f_c2[idx]),
            'C_var': float(c_var[idx]), 'lpi': float(lpi_arr[idx]),
            'tartini_f': float(tartini_f[idx]), 'tartini_c': float(tartini_c[idx]),
            'theta': float(theta_arr[idx]), 'phase': phases[idx],
        })

    lines = []
    lines.append("# 涌积态宇宙模型 v12.0 — 塔蒂尼量子层验证报告")
    lines.append(f"\n**运行时间**：{_RUN_TIME}  ")
    lines.append(f"**输出目录**：`{OUTPUT_DIR}`\n")
    lines.append("---\n")
    lines.append("## 一、实验参数\n")
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 网格 | {cfg.W}×{cfg.H} |")
    lines.append(f"| dt / steps | {cfg.dt} / {cfg.steps} |")
    lines.append(f"| **[v12] beta_tartini** | **{cfg.beta_tartini}** |")
    lines.append(f"| Theta_barrier（初始） | 0.28 |")
    lines.append(f"| alpha_metric | {cfg.alpha_metric} |")
    lines.append(f"| adaptive_tuning | {cfg.adaptive_tuning} |")
    lines.append("\n---\n")
    lines.append("## 二、量子层验证摘要\n")
    lines.append("| 指标 | 值 | 量子意义 |")
    lines.append("|---|---|---|")
    lines.append(f"| 实际运行帧数 | {total_ticks} | — |")
    lines.append(f"| 最终 f_P | {f_p[-1]:.4f} | 概率波函数残余（量子叠加态比例） |")
    lines.append(f"| 最终 f_C1 | {f_c1[-1]:.4f} | 坍缩定域态比例 |")
    lines.append(f"| 最终 f_C2 | {f_c2[-1]:.4f} | 退相干经典实在比例 |")
    lines.append(f"| C_var 峰值 | {float(c_var[cvar_peak_idx]):.5f}（tick={int(ticks[cvar_peak_idx])}）| 空间分化度峰值 |")
    lines.append(f"| LPI 峰值 | {float(lpi_arr[lpi_peak_idx]):.4f}（tick={int(ticks[lpi_peak_idx])}）| 相位锁定指数峰值 |")
    lines.append(f"| 最终 LPI | {lpi_arr[-1]:.4f} | 最终相位锁定程度 |")
    lines.append(f"| 累积塔蒂尼通量 | {tartini_c[-1]:.4f} | 总坍缩量（P→C1 的非线性转化） |")
    lines.append(f"| 质量守恒漂移 | {drift:.6f}% | — |")
    lines.append(f"| 最终演化阶段 | {phases[-1]} | — |")
    lines.append(f"| 最终 Theta | {theta_arr[-1]:.4f} | — |")
    lines.append("\n---\n")
    lines.append("## 三、自适应调参日志\n")
    lines.append(tuner.summary() if tuner is not None else "_自适应调参未启用_\n")
    lines.append("\n---\n")
    lines.append("## 四、进度里程碑数据（每 10% 采样）\n")
    lines.append("| 进度 | tick | f_P | f_C1 | f_C2 | C_var | LPI | 塔蒂尼通量 | 累积通量 | Θ | 阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(milestones):
        lines.append(
            f"| {i*10}% | {m['tick']} | {m['f_P']:.4f} | {m['f_C1']:.4f} | {m['f_C2']:.4f} | "
            f"{m['C_var']:.5f} | {m['lpi']:.4f} | {m['tartini_f']:.4e} | "
            f"{m['tartini_c']:.4f} | {m['theta']:.4f} | {m['phase']} |"
        )
    lines.append("\n---\n")
    lines.append("## 五、全量时序数据（每帧）\n")
    lines.append("| tick | f_P | f_C1 | f_C2 | C_var | LPI | 塔蒂尼通量 | 累积通量 | Θ | 阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in range(len(ticks)):
        lines.append(
            f"| {int(ticks[i])} | {f_p[i]:.4f} | {f_c1[i]:.4f} | {f_c2[i]:.4f} | "
            f"{c_var[i]:.5f} | {lpi_arr[i]:.4f} | {tartini_f[i]:.4e} | "
            f"{tartini_c[i]:.4f} | {theta_arr[i]:.4f} | {phases[i]} |"
        )

    report_path = OUTPUT_DIR / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n  [报告] 已保存至 {report_path}")


if __name__ == '__main__':
    run()
