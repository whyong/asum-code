"""
涌积态宇宙模型 v9.0 — 自适应双稳态与循环相变架构
基于 v8.0（T-008 最终参数），针对三项已知问题的定向升级：

  1. 消除规则2触发（低谷期自然积累）
     v8.0 在低谷期（f_C2≈0.17，约74000帧）末尾规则2触发（C2积累过慢），
     通过降低 Theta_barrier 强制触发相变。
     v9.0 引入"低谷期自适应势垒"：在图灵期内 f_C2 持续低于阈值时，
     自动缓慢降低 Theta_barrier（而非规则2的突变式降低），
     让系统以更自然的方式从低谷态跨越到高位态。

  2. 种子扰动自适应强度（C_var 反馈控制）
     v8.0 的 seed_strength=0.002 是固定值。
     v9.0 引入自适应种子强度：C_var 低时增强种子（打破对称性），
     C_var 高时减弱种子（不干扰已形成的斑图），
     实现对称性破缺的精确控制。

  3. 双层涌现度规（C1 + C2 状态记忆共同参与）
     v8.0 只对 stock_state_C2 施加度规。
     v9.0 引入 alpha_metric_C1，让 C1 的状态记忆也参与空间弯曲，
     在黑暗年代（C1 大量积累时）形成更强的向心引力，
     加速从低谷态到高位态的相变。

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
    steps  = 24000   # 日常运行默认值（长程验证用 500000）

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
    render_every   = 999999  # 暂时关闭帧图片输出（恢复用 20）
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
    alpha_metric  = 0.3    # C2 状态记忆的空间弯曲强度

    # ── [v9.0 新增] 双层涌现度规（C1 状态记忆也参与空间弯曲）─────────
    # 黑暗年代 C1 大量积累时，C1 的遗迹引力加速相变
    # alpha_metric_C1=0 时退化为 v8.0（只有 C2 度规）
    alpha_metric_C1 = 0.0   # T-003: 完全禁用，做干净对照（原0.05）

    # ── [v8.0 继承] 早期图灵种子扰动 ─────────────────────────────────
    seed_enabled    = True
    seed_fc2_thresh = 0.40
    seed_strength   = 0.002   # 基础强度（自适应版本以此为上限）
    seed_freq       = 2

    # ── [v9.0 新增] 种子扰动自适应强度 ──────────────────────────────
    # C_var 低时增强种子（打破对称性），C_var 高时减弱（不干扰斑图）
    # seed_adaptive=False 时退化为 v8.0 固定强度
    seed_adaptive       = True
    seed_cvar_low       = 0.005   # C_var 低于此值时使用最大强度
    seed_cvar_high      = 0.030   # C_var 高于此值时强度降为零
    # 实际强度 = seed_strength * max(0, (seed_cvar_high - C_var) / (seed_cvar_high - seed_cvar_low))

    # ── [v9.0 新增] 低谷期自适应势垒（替代规则2的突变式降低）─────────
    # 在图灵期内 f_C2 持续低于阈值时，缓慢降低 Theta_barrier
    # 目标：让系统自然积累跃迁动力，无需规则2的外部干预
    trough_adapt_enabled  = True
    trough_fc2_thresh     = 0.20   # f_C2 低于此值时触发低谷期自适应
    trough_phi_thresh     = 25.0   # T-008: 降低（原40），低谷期Phi_net≈25~35，加phase条件后可精确触发
    trough_decay_rate     = 0.9999 # 每帧 Theta_barrier 乘以此系数（极缓慢降低）
    trough_theta_min      = 0.18   # Theta_barrier 的下界（防止过度降低）


# ─────────────────────────────────────────────
# 二、自适应调参模块（继承 v8.0，规则7/7b 禁用，规则2 弱化）
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

        # 规则1：图灵期衰退预警
        if (phase == 'TURING_GROWTH' and len(self._phi_net_buf) >= 50
                and self._declining(self._phi_net_buf[-50:], 0.3)
                and self._declining(self._c_var_buf[-50:], 0.1)):
            self._apply('lam_rate_C2', cfg.lam_rate_C2,
                        max(self.BOUNDS['lam_rate_C2'][0], cfg.lam_rate_C2 * 0.80),
                        tick, '图灵期衰退预警：降低lam_rate_C2增强记忆')

        # 规则2：C2 积累过慢（v9.0 弱化：仅在 Phi_net 极低时触发，避免低谷期误触发）
        # v8.0 中规则2在低谷期（Phi_net≈74）触发，v9.0 改为只在 Phi_net < 5 时触发
        # 低谷期的势垒降低由 trough_adapt 机制负责
        if (phase == 'TURING_GROWTH' and self._turing_entry_tick is not None
                and tick - self._turing_entry_tick > 200):
            elapsed = tick - self._turing_entry_tick
            rate = (f_C2 - (self._turing_fc2_entry or 0)) / elapsed
            if rate < 0.0001 and f_C2 < 0.05 and phi_net < 5.0:
                # 仅在 f_C2 极低且 Phi_net 极低时触发（真正的启动失败）
                self._apply('Theta_barrier', cfg.Theta_barrier,
                            max(self.BOUNDS['Theta_barrier'][0], cfg.Theta_barrier * 0.90),
                            tick, f'C2积累过慢且Phi_net极低(rate={rate:.6f}/帧)，降低势垒')

        # 规则3：CHAOTIC_EDGE 锁死
        if (self._chaotic_cnt >= 300 and phi_net == 0.0 and f_C2 < 0.20):
            new = max(self.BOUNDS['lam_rate_C1'][0], cfg.lam_rate_C1 * 0.80)
            if new < cfg.lam_rate_C1 - 1e-6:
                self._apply('lam_rate_C1', cfg.lam_rate_C1, new, tick,
                            f'CHAOTIC_EDGE锁死{self._chaotic_cnt}帧，降低lam_rate_C1')
                self._chaotic_cnt = 0

        # 规则4：C1 过度积累
        if (phase in ('UNIFORM_GROWTH', 'TURING_GROWTH') and f_C1 > 0.75
                and f_C2 < 0.05 and tick > 100):
            self._apply('Theta_barrier', cfg.Theta_barrier,
                        max(self.BOUNDS['Theta_barrier'][0], cfg.Theta_barrier * 0.85),
                        tick, f'C1过度积累(f_C1={f_C1:.3f})，降低势垒')

        # 规则5：C_var 长期过低
        if (phase == 'UNIFORM_GROWTH' and tick > 500 and c_var < 0.001
                and f_C2 > 0.1 and cfg.lam_rate_C1 < 0.03
                and (not self.log or tick - self.log[-1][0] >= 200)):
            self._apply('lam_rate_C2', cfg.lam_rate_C2,
                        min(self.BOUNDS['lam_rate_C2'][1], cfg.lam_rate_C2 * 1.10),
                        tick, f'C_var长期过低({c_var:.5f})，提高lam_rate_C2')

        # 规则6：lam_rate_C2 上调后仍崩溃
        if (self._chaotic_cnt >= 200 and phi_net == 0.0 and cfg.lam_rate_C2 > 0.007
                and (not self.log or tick - self.log[-1][0] >= 200)):
            self._apply('lam_rate_C2', cfg.lam_rate_C2,
                        max(0.005, cfg.lam_rate_C2 * 0.90),
                        tick, f'lam_rate_C2上调后仍CHAOTIC_EDGE，回调')

        # 规则7/7b：已禁用（v7.0 T-008 关键发现：规则7是图灵期杀手）

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
# 三、物理层
# ─────────────────────────────────────────────

class PhysicsLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        shape = (cfg.H, cfg.W)

        rng = np.random.default_rng(42)
        self.P  = rng.uniform(0.2, 1.0, shape).astype(np.float32)
        self.C1 = np.zeros(shape, np.float32)
        self.C2 = np.zeros(shape, np.float32)

        # 五轨存量场
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

        # 供观测层读取
        self.I_stock         = np.zeros(shape, np.float32)
        self.I_increment     = np.zeros(shape, np.float32)
        self.I_drive         = np.zeros(shape, np.float32)
        self.A_C1_to_C2      = np.zeros(shape, np.float32)
        self.A_C2_to_C1      = np.zeros(shape, np.float32)
        self._I_norm_vis      = np.zeros(shape, np.float32)
        self.sink_mask       = np.zeros(shape, np.float32)
        self.metric_vis      = np.ones(shape, np.float32)   # 供可视化
        self.rate_stock_mean = 0.0

        self._instab_active = False

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

        # ── 步骤4：D 失真算子 ────────────────────────────────────────
        I_context   = convolve(I_macro, self.blur_k, mode='wrap')
        S_norm_C1   = self.stock_state_C1 / (self.stock_state_C1 + cfg.k_norm_base)
        S_norm_C2   = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        S_macro     = cfg.w_state_C1 * S_norm_C1 + cfg.w_state_C2 * S_norm_C2
        laplacian_S = lap(S_macro)
        noise       = spatial_noise(P.shape) * cfg.k_noise * I_context
        I_distorted = I_context + cfg.kappa_inertia * laplacian_S + noise

        # ── 步骤5：柔性极化与死区清洗 ────────────────────────────────
        Omega_raw    = softsign(I_distorted, cfg.I_max_capacity)
        Omega_damped = (1.0 - cfg.D_decay) * convolve(Omega_raw, self.gauss_k, mode='wrap')
        I_drive      = np.where(np.abs(Omega_damped) < 1e-4, 0.0, Omega_damped)
        self.I_drive     = I_drive
        self._I_norm_vis = I_macro / (I_macro.max() + 1e-9)

        # ── 步骤6：[v9.0] 双层涌现度规 ──────────────────────────────
        # C2 状态记忆（继承 v8.0）+ C1 状态记忆（v9.0 新增）
        S_C2_norm = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        S_C1_norm = self.stock_state_C1 / (self.stock_state_C1 + cfg.k_norm_base)
        # 合并度规因子：C2 主导（alpha_metric），C1 辅助（alpha_metric_C1）
        metric_factor = np.maximum(0.1,
            1.0 - cfg.alpha_metric * S_C2_norm - cfg.alpha_metric_C1 * S_C1_norm)
        D_C1_eff = cfg.D_C1 * metric_factor
        D_C2_eff = cfg.D_C2 * metric_factor
        self.metric_vis = metric_factor  # 供可视化

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

        self.P_prev  = P.copy()
        self.C1_prev = C1.copy()
        self.C2_prev = C2.copy()

        dP_dt  = (cfg.D_P  * lap(P)
                  - A_P_to_C1 + A_C1_to_P + A_C1_decay + A_C2_sink)
        dC1_dt = (div_diff(C1, D_C1_eff)
                  + A_P_to_C1 - A_C1_to_P - A_C1_to_C2 + A_C2_to_C1
                  - A_C1_decay + A_C2_decay)
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
            A_instab -= A_instab.mean()
            dC2_dt = dC2_dt + A_instab

        self.P  = np.maximum(P  + dP_dt  * cfg.dt, 0.0)
        self.C1 = np.maximum(C1 + dC1_dt * cfg.dt, 0.0)
        self.C2 = np.maximum(C2 + dC2_dt * cfg.dt, 0.0)

        # ── 步骤8：[v9.0] 自适应种子扰动 ────────────────────────────
        if cfg.seed_enabled:
            M_total = self.P.sum() + self.C1.sum() + self.C2.sum()
            f_C2_now = self.C2.sum() / (M_total + 1e-9)
            if f_C2_now < cfg.seed_fc2_thresh and self.C2.mean() > 1e-6:
                # 自适应强度：C_var 低时增强，C_var 高时减弱
                if cfg.seed_adaptive:
                    cvar_range = cfg.seed_cvar_high - cfg.seed_cvar_low
                    cvar_factor = max(0.0, (cfg.seed_cvar_high - c_var) / (cvar_range + 1e-9))
                    cvar_factor = min(1.0, cvar_factor)
                else:
                    cvar_factor = 1.0
                # f_C2 线性衰减
                fc2_factor = 1.0 - f_C2_now / cfg.seed_fc2_thresh
                strength = cfg.seed_strength * fc2_factor * cvar_factor
                if strength > 1e-6:
                    C2_sum_before = self.C2.sum()
                    scale = 1.0 + self._seed_template * strength
                    scale = np.maximum(scale, 0.01)
                    C2_new = self.C2 * scale
                    self.C2 = C2_new * (C2_sum_before / (C2_new.sum() + 1e-9))

        # ── 步骤9：[v9.0] 低谷期自适应势垒 ──────────────────────────
        # 只在 TURING_GROWTH 期内触发，避免启动期误触发
        # 在图灵期内 f_C2 持续低且 Phi_net 足够高时，缓慢降低 Theta_barrier
        if cfg.trough_adapt_enabled and phase == 'TURING_GROWTH':
            M_total = self.P.sum() + self.C1.sum() + self.C2.sum()
            f_C2_now = self.C2.sum() / (M_total + 1e-9)
            phi_net_now = float(self.A_C1_to_C2.sum() - self.A_C2_to_C1.sum())
            if (f_C2_now < cfg.trough_fc2_thresh
                    and phi_net_now > cfg.trough_phi_thresh
                    and cfg.Theta_barrier > cfg.trough_theta_min):
                cfg.Theta_barrier = max(cfg.trough_theta_min,
                                        cfg.Theta_barrier * cfg.trough_decay_rate)


# ─────────────────────────────────────────────
# 四、观测层（继承 v8.0）
# ─────────────────────────────────────────────

class ObservationLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = {k: [] for k in [
            'tick', 'M_total', 'f_C2', 'f_C1',
            'delta_sigma', 'Phi_net', 'C_var', 'H_C2',
            'n_clusters', 'size_gini', 'phase', 'Theta_barrier'
        ]}
        self._hd_buf = []

    def observe(self, phys: PhysicsLayer, tick: int):
        cfg = self.cfg
        P, C1, C2 = phys.P, phys.C1, phys.C2
        eps = 1e-9

        M_P, M_C1, M_C2 = P.sum(), C1.sum(), C2.sum()
        M_total = M_P + M_C1 + M_C2
        f_C2 = M_C2 / (M_total + eps)
        f_C1 = M_C1 / (M_total + eps)

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

        h = self.history
        h['tick'].append(tick)
        h['M_total'].append(M_total)
        h['f_C2'].append(f_C2)
        h['f_C1'].append(f_C1)
        h['delta_sigma'].append(delta_sig)
        h['Phi_net'].append(Phi_net)
        h['C_var'].append(C_var)
        h['H_C2'].append(H_C2)
        h['n_clusters'].append(n_cl)
        h['size_gini'].append(s_gini)
        h['phase'].append(phase)
        h['Theta_barrier'].append(cfg.Theta_barrier)  # 记录势垒变化
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

    fig = plt.figure(figsize=(16, 9), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.35, wspace=0.3,
                            left=0.05, right=0.97,
                            top=0.92, bottom=0.08)

    # [v9.0] 第三格：双层度规场（C1+C2 共同弯曲）
    fields = [
        (phi_C2,           'viridis',  None,            'φ_C2  高阶结构密度\n蓝=低  黄=高'),
        (phys.I_drive,     'RdBu_r',   TwoSlopeNorm(0), 'I_drive  驱动场\n红=生长  蓝=退化  白=零'),
        (phys.metric_vis,  'plasma',   None,            '[v9] 双层度规\n亮=扩散正常  暗=时空弯曲锁定'),
        (phys._I_norm_vis, 'cividis',  None,            'I_macro归一化  宏观因果场\n深=弱  亮=强'),
    ]
    for col, (data, cmap, norm, title) in enumerate(fields):
        ax = fig.add_subplot(gs[0, col])
        kw = dict(cmap=cmap, interpolation='nearest', aspect='auto')
        if norm is not None:
            kw['norm'] = norm
        im = ax.imshow(data, **kw)
        ax.set_title(title, color='white', fontsize=8, pad=3)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(
            labelcolor='white', labelsize=6)

    h = obs.history
    ticks = h['tick']
    _n = len(ticks)
    if _n > 1000:
        _step = _n // 1000
        _idx  = list(range(0, _n, _step)) + [_n - 1]
        ticks_plot = [ticks[i] for i in _idx]
        h_plot = {k: [h[k][i] for i in _idx] for k in
                  ['f_C2', 'f_C1', 'delta_sigma', 'Phi_net',
                   'n_clusters', 'size_gini', 'phase', 'Theta_barrier']}
    else:
        ticks_plot = ticks
        h_plot = h

    ax_ts = fig.add_subplot(gs[1, :3])
    ax_ts.set_facecolor('#1a1a1a')
    for key, color, lbl in [
        ('f_C2',          '#2ecc71', 'f_C2  高阶结构占比'),
        ('f_C1',          '#3498db', 'f_C1  中间态占比'),
        ('delta_sigma',   '#e67e22', 'ΔΣ  信息净增率'),
        ('Phi_net',       '#e74c3c', 'Φ_net  净跃迁通量'),
        ('Theta_barrier', '#f39c12', 'Θ  势垒（低谷期自适应）'),
    ]:
        ax_ts.plot(ticks_plot, h_plot[key], color=color, lw=1.0, label=lbl)

    if len(ticks_plot) > 1:
        phases_plot = h_plot['phase']
        seg_start, seg_phase = ticks_plot[0], phases_plot[0]
        for i in range(1, len(ticks_plot)):
            if phases_plot[i] != seg_phase:
                ax_ts.axvspan(seg_start, ticks_plot[i], alpha=0.12,
                              color=PHASE_COLORS.get(seg_phase, '#ffffff'))
                seg_start, seg_phase = ticks_plot[i], phases_plot[i]
        ax_ts.axvspan(seg_start, ticks_plot[-1], alpha=0.12,
                      color=PHASE_COLORS.get(seg_phase, '#ffffff'))

    ax_ts.axhline(0, color='white', lw=0.4, alpha=0.4)
    ax_ts.legend(loc='upper left', fontsize=6,
                 facecolor='#1a1a1a', labelcolor='white', framealpha=0.6)
    ax_ts.tick_params(colors='white', labelsize=7)
    ax_ts.set_xlabel('帧数 (tick)', color='white', fontsize=8)
    ax_ts.set_ylabel('比值 / 通量', color='white', fontsize=8)
    ax_ts.set_title('全局时序曲线  （背景色带 = 当前演化阶段）',
                    color='#aaaaaa', fontsize=8, pad=4)
    for spine in ax_ts.spines.values():
        spine.set_edgecolor('#444')

    ax_cl = fig.add_subplot(gs[1, 3])
    ax_cl.set_facecolor('#1a1a1a')
    ax_cl.plot(ticks_plot, h_plot['n_clusters'], color='#f39c12', lw=1.0, label='区块数量')
    ax_cl.plot(ticks_plot, h_plot['size_gini'],  color='#1abc9c', lw=1.0, label='大小基尼系数')
    ax_cl.legend(loc='upper left', fontsize=7,
                 facecolor='#1a1a1a', labelcolor='white', framealpha=0.6)
    ax_cl.tick_params(colors='white', labelsize=7)
    ax_cl.set_xlabel('帧数 (tick)', color='white', fontsize=8)
    ax_cl.set_title('空间结构统计', color='#aaaaaa', fontsize=8, pad=4)
    for spine in ax_cl.spines.values():
        spine.set_edgecolor('#444')

    phase_labels = {
        'TURING_GROWTH': '图灵斑图涌现', 'UNIFORM_GROWTH': '均匀生长期',
        'RESTRUCTURING': '信息活跃·结构重组', 'COLLAPSE': '信息衰退·结构崩解',
        'COARSENING': '粗化·区块吞噬', 'NEAR_EQUILIBRIUM': '接近局部平衡',
        'CHAOTIC_EDGE': '混沌边缘',
    }
    fig.text(0.5, 0.01,
             '  '.join(f'█ {v}' for v in phase_labels.values()),
             ha='center', va='bottom', fontsize=6.5, color='#888888',
             bbox=dict(facecolor='#1a1a1a', edgecolor='none', pad=2))

    phase_now = h['phase'][-1] if h['phase'] else '—'
    color_now = PHASE_COLORS.get(phase_now, 'white')
    theta_now = h['Theta_barrier'][-1] if h['Theta_barrier'] else cfg.Theta_barrier
    fig.suptitle(
        f'涌积态宇宙 v9.0   第 {tick:05d} 帧   '
        f'C2占比={h["f_C2"][-1]:.3f}   Θ={theta_now:.4f}   '
        f'当前阶段：{phase_labels.get(phase_now, phase_now)}',
        color=color_now, fontsize=11, fontweight='bold'
    )

    plt.savefig(OUTPUT_DIR / f'frame_{tick:05d}.png',
                dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f'  [render] tick={tick:05d}  phase={phase_now}  '
          f'f_C2={h["f_C2"][-1]:.3f}  '
          f'Theta={theta_now:.4f}  '
          f'n_cl={h["n_clusters"][-1]}')


# ─────────────────────────────────────────────
# 六、主循环
# ─────────────────────────────────────────────

def run():
    cfg   = Config()
    phys  = PhysicsLayer(cfg)
    obs   = ObservationLayer(cfg)
    tuner = AdaptiveTuner(cfg) if cfg.adaptive_tuning else None

    print("=" * 65)
    print("涌积态宇宙模型 v9.0 启动")
    print(f"  网格: {cfg.W}×{cfg.H}   dt={cfg.dt}   最大帧数={cfg.steps}")
    print(f"  [v8.0继承] alpha_metric={cfg.alpha_metric}  seed_freq={cfg.seed_freq}")
    print(f"  [v9.0新增] alpha_metric_C1={cfg.alpha_metric_C1}  (双层涌现度规)")
    print(f"  [v9.0新增] seed_adaptive={cfg.seed_adaptive}  "
          f"cvar_low={cfg.seed_cvar_low}  cvar_high={cfg.seed_cvar_high}")
    print(f"  [v9.0新增] trough_adapt={cfg.trough_adapt_enabled}  "
          f"fc2_thresh={cfg.trough_fc2_thresh}  "
          f"phi_thresh={cfg.trough_phi_thresh}  "
          f"decay={cfg.trough_decay_rate}  "
          f"theta_min={cfg.trough_theta_min}")
    print(f"  自适应调参: {'开启' if cfg.adaptive_tuning else '关闭'}（规则7/7b禁用，规则2弱化）")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 65)

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

        # 进度指示：每 500 帧输出一行关键指标
        if tick % 500 == 0 and tick > 0:
            h = obs.history
            pct = tick / cfg.steps * 100
            print(f'  [progress] {pct:5.1f}%  tick={tick:06d}  '
                  f'phase={h["phase"][-1]:<18s}  '
                  f'f_C2={h["f_C2"][-1]:.3f}  '
                  f'C_var={h["C_var"][-1]:.4f}  '
                  f'Phi={h["Phi_net"][-1]:.1f}  '
                  f'Θ={h["Theta_barrier"][-1]:.4f}')

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
    print("\n── 演化摘要 ──────────────────────────────")
    print(f"  总帧数:      {h['tick'][-1]}")
    print(f"  最终 f_C2:   {h['f_C2'][-1]:.4f}")
    print(f"  最终 f_C1:   {h['f_C1'][-1]:.4f}")
    print(f"  最终阶段:    {h['phase'][-1]}")
    print(f"  最终 Theta:  {h['Theta_barrier'][-1]:.4f}")
    m0, m1 = h['M_total'][0], h['M_total'][-1]
    drift = abs(m1 - m0) / (m0 + 1e-9) * 100
    print(f"  质量守恒漂移: {drift:.6f}%")
    if tuner and tuner.log:
        print(f"  自适应调参次数: {len(tuner.log)}")
    print("──────────────────────────────────────────")
    _save_report(obs, tuner)


def _save_report(obs: ObservationLayer, tuner=None):
    h   = obs.history
    cfg = obs.cfg
    if not h['tick']:
        return

    ticks     = np.array(h['tick'])
    f_c2      = np.array(h['f_C2'])
    f_c1      = np.array(h['f_C1'])
    delta_sig = np.array(h['delta_sigma'])
    phi_net   = np.array(h['Phi_net'])
    c_var     = np.array(h['C_var'])
    n_cl      = np.array(h['n_clusters'])
    gini_arr  = np.array(h['size_gini'])
    theta_arr = np.array(h['Theta_barrier'])
    phases    = h['phase']

    total_ticks = h['tick'][-1]
    m0, m1 = h['M_total'][0], h['M_total'][-1]
    drift = abs(m1 - m0) / (m0 + 1e-9) * 100

    peak_idx      = int(np.argmax(f_c2))
    cvar_peak_idx = int(np.argmax(c_var))
    ncl_peak_idx  = int(np.argmax(n_cl))

    phase_changes = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            phase_changes.append((int(ticks[i]), phases[i-1], phases[i]))

    milestones = []
    for pct in range(0, 101, 10):
        idx = min(int(pct / 100 * (len(ticks) - 1)), len(ticks) - 1)
        milestones.append({
            'tick': int(ticks[idx]), 'f_C2': float(f_c2[idx]),
            'f_C1': float(f_c1[idx]), 'delta_sig': float(delta_sig[idx]),
            'phi_net': float(phi_net[idx]), 'C_var': float(c_var[idx]),
            'n_cl': int(n_cl[idx]), 'gini': float(gini_arr[idx]),
            'theta': float(theta_arr[idx]), 'phase': phases[idx],
        })

    lines = []
    lines.append("# 涌积态宇宙模型 v9.0 — 运行报告")
    lines.append(f"\n**运行时间**：{_RUN_TIME}  ")
    lines.append(f"**输出目录**：`{OUTPUT_DIR}`\n")
    lines.append("---\n")
    lines.append("## 一、运行参数\n")
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 网格 | {cfg.W}×{cfg.H} |")
    lines.append(f"| dt / steps | {cfg.dt} / {cfg.steps} |")
    lines.append(f"| D_P / D_C1 / D_C2 | {cfg.D_P} / {cfg.D_C1} / {cfg.D_C2} |")
    lines.append(f"| k_noise / Theta_barrier(初始) | {cfg.k_noise} / 0.28 |")
    lines.append(f"| alpha_metric / alpha_metric_C1 | {cfg.alpha_metric} / {cfg.alpha_metric_C1} |")
    lines.append(f"| seed_freq / seed_strength | {cfg.seed_freq} / {cfg.seed_strength} |")
    lines.append(f"| seed_adaptive | {cfg.seed_adaptive} |")
    lines.append(f"| trough_adapt_enabled | {cfg.trough_adapt_enabled} |")
    lines.append(f"| trough_fc2_thresh / phi_thresh | {cfg.trough_fc2_thresh} / {cfg.trough_phi_thresh} |")
    lines.append(f"| trough_decay_rate / theta_min | {cfg.trough_decay_rate} / {cfg.trough_theta_min} |")
    lines.append(f"| adaptive_tuning | {cfg.adaptive_tuning} |")
    lines.append("\n---\n")
    lines.append("## 二、总体摘要\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 实际运行帧数 | {total_ticks} |")
    lines.append(f"| 终止原因 | {'热寂' if total_ticks < cfg.steps - 1 else '达到最大帧数'} |")
    lines.append(f"| 最终 f_C2 | {f_c2[-1]:.4f} |")
    lines.append(f"| 最终 f_C1 | {f_c1[-1]:.4f} |")
    lines.append(f"| 最终演化阶段 | {phases[-1]} |")
    lines.append(f"| 最终 Theta_barrier | {theta_arr[-1]:.4f} |")
    lines.append(f"| 质量守恒漂移 | {drift:.6f}% |")
    lines.append(f"| f_C2 峰值 | {float(f_c2[peak_idx]):.4f}（tick={int(ticks[peak_idx])}）|")
    lines.append(f"| C_var 峰值 | {float(c_var[cvar_peak_idx]):.5f}（tick={int(ticks[cvar_peak_idx])}）|")
    lines.append(f"| 区块数峰值 | {int(n_cl[ncl_peak_idx])}（tick={int(ticks[ncl_peak_idx])}）|")
    lines.append("\n---\n")
    lines.append("## 三、阶段切换节点\n")
    if phase_changes:
        lines.append("| tick | 从 | 到 |")
        lines.append("|---|---|---|")
        for t, frm, to in phase_changes:
            lines.append(f"| {t} | {frm} | {to} |")
    else:
        lines.append("_无阶段切换_")
    lines.append("\n---\n")
    lines.append("## 四、自适应调参日志\n")
    lines.append(tuner.summary() if tuner is not None else "_自适应调参未启用_\n")
    lines.append("\n---\n")
    lines.append("## 五、进度里程碑数据（每 10% 采样）\n")
    lines.append("| 进度 | tick | f_C2 | f_C1 | ΔΣ | Φ_net | C_var | 区块数 | Gini | Θ | 阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(milestones):
        lines.append(
            f"| {i*10}% | {m['tick']} | {m['f_C2']:.4f} | {m['f_C1']:.4f} | "
            f"{m['delta_sig']:.4e} | {m['phi_net']:.4e} | {m['C_var']:.5f} | "
            f"{m['n_cl']} | {m['gini']:.4f} | {m['theta']:.4f} | {m['phase']} |"
        )
    lines.append("\n---\n")
    lines.append("## 六、全量时序数据（每帧）\n")
    lines.append("| tick | f_C2 | f_C1 | ΔΣ | Φ_net | C_var | 区块数 | Gini | Θ | 阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in range(len(ticks)):
        lines.append(
            f"| {int(ticks[i])} | {f_c2[i]:.4f} | {f_c1[i]:.4f} | "
            f"{delta_sig[i]:.4e} | {phi_net[i]:.4e} | {c_var[i]:.5f} | "
            f"{int(n_cl[i])} | {gini_arr[i]:.4f} | {theta_arr[i]:.4f} | {phases[i]} |"
        )

    report_path = OUTPUT_DIR / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n  [报告] 已保存至 {report_path}")


if __name__ == '__main__':
    run()
