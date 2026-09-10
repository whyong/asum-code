"""
涌积态宇宙模型 v7.0 — 情境张量微积分与掩码门控架构
基于 v6_ex，新增三项核心升级：
  1. 情境自适应遗忘率 λ(x,y,t)：由上一帧 I_macro 梯度动态计算
     活跃前沿（高梯度）→ λ 小 → 记忆深；静默区域 → λ 大 → 遗忘快
  2. 零残差拓扑热汇 H_sink：局部 C2 方差超阈值时，掩码切断变率反馈 + 被动耗散
     精确定位极化死锁区域，替代 v6_ex 的全局回差法
  3. ΔΣ 观测修复：sig_loss 只基于变率特征群，阶段判定恢复准确

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
    steps  = 24000   # 日常运行默认值（长程验证用 96000）

    # ── 扩散系数 ──────────────────────────────────────────────────────
    D_P    = 0.20
    D_C1   = 0.05
    D_C2   = 0.004   # T-007: 降低（原0.008），压低C2扩散，促进局部聚集形成斑图

    # ── 传承失真 ──────────────────────────────────────────────────────
    k_noise   = 0.50    # T-004: 提高（原0.30），打破初始对称性，让启动期空间扰动更强
    blur_size = 5

    # ── 拓扑清洗 ──────────────────────────────────────────────────────
    I_max_capacity = 1.0
    D_decay        = 0.15
    gauss_size     = 5

    # ── 跃迁势垒与速率 ────────────────────────────────────────────────
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
    render_every   = 20
    struct_every   = 20
    phi_threshold  = None
    hd_window      = 300
    hd_drive_eps   = 5e-5

    # ── [v6_ex 继承] 广义算子参数 ─────────────────────────────────────
    w_rate_P   = 0.1
    w_rate_C1  = 0.7
    w_rate_C2  = 1.5    # T-003: 中间值（T-001=2.0太强，T-002=1.2太弱）
    w_state_C1 = 0.15
    w_state_C2 = 0.25

    # 遗忘率基线（情境化后作为 λ_base）
    lam_rate_P   = 0.15
    lam_rate_C1  = 0.04
    lam_rate_C2  = 0.005
    lam_state_C1 = 0.005
    lam_state_C2 = 0.003   # T-013: 提高（原0.001），防止长程积累导致S_macro过强锁死稳态

    kw_rate_P   = 0.001
    kw_rate_C1  = 0.005
    kw_rate_C2  = 0.0005
    kw_state_C1 = 0.05
    kw_state_C2 = 0.05

    k_norm_base   = 0.002
    kappa_inertia = 0.3
    alpha_metric  = 0.0

    # ── [v7.0 新增] 情境自适应遗忘率 ─────────────────────────────────
    # alpha_lambda=0 时退化为 v6_ex（全局固定 λ），用于基线验证
    alpha_lambda  = 1.5   # 梯度敏感度；建议范围 [0, 5.0]
    lam_min_ratio = 0.1   # λ 下界保护：λ >= lam_base * lam_min_ratio

    # ── [v7.0 新增] 零残差拓扑热汇 H_sink ────────────────────────────
    theta_sink  = 0.03    # T-016: 中间值（T-014=0.08太高无法触发，T-015=0.005过低压制图灵）
    gamma_sink  = 0.02    # 被动耗散强度；建议范围 [0.005, 0.10]
    sink_window = 5       # 局部方差计算窗口（奇数）

    # ── [v7.0 保留] C2 全局不稳定机制（回差法，与 H_sink 互补）────────
    # 回差法处理"过于均匀"，H_sink 处理"过于极化"，两者共存
    k_instab         = 0.005
    theta_instab_low = 0.005
    theta_instab_high= 0.020


# ─────────────────────────────────────────────
# 二、自适应调参模块（继承 v6_ex，规则不变）
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
        self._theta_rescued   = False
        self._theta_original  = None
        self._noise_boosted   = False
        self._noise_boost_tick = None
        self._noise_original  = None
        self._noise_boost_duration = 500

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
            self._theta_rescued = False

        self._chaotic_cnt = self._chaotic_cnt + 1 if phase == 'CHAOTIC_EDGE' else 0
        self._last_phase  = phase

        if (self._noise_boosted and self._noise_boost_tick is not None
                and tick - self._noise_boost_tick >= self._noise_boost_duration):
            old = cfg.k_noise
            cfg.k_noise = self._noise_original
            self.log.append((tick, 'k_noise', old, self._noise_original,
                             f'k_noise提升{self._noise_boost_duration}帧后自动恢复'))
            print(f'  [AdaptTune] tick={tick:05d}  k_noise: {old:.4f} → {self._noise_original:.4f}  (自动恢复)')
            self._noise_boosted = False
            self._noise_boost_tick = None
            self._noise_original = None

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

        # 规则2：C2 积累过慢
        if (phase == 'TURING_GROWTH' and self._turing_entry_tick is not None
                and tick - self._turing_entry_tick > 200):
            elapsed = tick - self._turing_entry_tick
            rate = (f_C2 - (self._turing_fc2_entry or 0)) / elapsed
            if rate < 0.0001 and f_C2 < 0.15:
                self._apply('Theta_barrier', cfg.Theta_barrier,
                            max(self.BOUNDS['Theta_barrier'][0], cfg.Theta_barrier * 0.90),
                            tick, f'C2积累过慢(rate={rate:.6f}/帧)，降低势垒')

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

        # 规则7：图灵期衰退救援 —— T-008 禁用
        # 诊断：C_var 刚过 0.01 进入图灵期时立刻触发，反而打断图灵斑图自然发展
        # if (phase == 'TURING_GROWTH' and not self._theta_rescued
        #         and 0.010 < c_var < 0.040 and phi_net < 40.0
        #         and (not self.log or tick - self.log[-1][0] >= 200)):
        #     self._theta_original = cfg.Theta_barrier
        #     self._theta_rescued  = True
        #     self._apply('Theta_barrier', cfg.Theta_barrier,
        #                 max(self.BOUNDS['Theta_barrier'][0], cfg.Theta_barrier * 0.72),
        #                 tick, f'图灵期衰退救援：C_var={c_var:.4f}，临时降低Theta_barrier')

        # 规则7b：救援后恢复 —— T-008 禁用（随规则7一起禁用）
        # if (self._theta_rescued and self._theta_original is not None
        #         and (not self.log or tick - self.log[-1][0] >= 300)):
        #     if c_var > 0.050 or phase != 'TURING_GROWTH':
        #         old = cfg.Theta_barrier
        #         self._apply('Theta_barrier', old, self._theta_original, tick,
        #                     f'图灵期救援结束，恢复Theta_barrier至{self._theta_original:.3f}')
        #         self._theta_rescued  = False
        #         self._theta_original = None
        #         if not self._noise_boosted:
        #             self._noise_original   = cfg.k_noise
        #             self._noise_boost_tick = tick
        #             self._noise_boosted    = True
        #             self._apply('k_noise', cfg.k_noise,
        #                         min(0.50, cfg.k_noise * 1.50), tick,
        #                         f'救援结束后临时提升k_noise（{self._noise_boost_duration}帧后恢复）')

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

        # 五轨存量场（继承 v6_ex）
        self.stock_rate_P   = np.zeros(shape, np.float32)
        self.stock_rate_C1  = np.zeros(shape, np.float32)
        self.stock_rate_C2  = np.zeros(shape, np.float32)
        self.stock_state_C1 = np.zeros(shape, np.float32)
        self.stock_state_C2 = np.zeros(shape, np.float32)

        # [v7.0] 上一帧 I_macro（用于计算情境遗忘率）
        self._I_macro_prev = np.zeros(shape, np.float32)

        # 上一帧状态
        self.P_prev  = np.maximum(self.P + cfg.D_P * lap(self.P) * cfg.dt, 0.0)
        self.C1_prev = self.C1.copy()
        self.C2_prev = self.C2.copy()

        # 预计算卷积核
        self.blur_k  = make_kernel(cfg.blur_size, 'blur')
        self.gauss_k = make_kernel(cfg.gauss_size, 'gauss')

        # 供观测层读取的中间量
        self.I_stock     = np.zeros(shape, np.float32)
        self.I_increment = np.zeros(shape, np.float32)
        self.I_drive     = np.zeros(shape, np.float32)
        self.A_C1_to_C2  = np.zeros(shape, np.float32)
        self.A_C2_to_C1  = np.zeros(shape, np.float32)
        self._I_norm_vis  = np.zeros(shape, np.float32)
        # [v7.0] H_sink 掩码（供可视化）
        self.sink_mask   = np.zeros(shape, np.float32)

        # 回差状态（全局不稳定机制，与 H_sink 互补）
        self._instab_active = False

    def step(self):
        cfg = self.cfg
        P, C1, C2 = self.P, self.C1, self.C2

        # ── 步骤0：情境遗忘率场计算（用上一帧 I_macro）────────────────
        # 梯度幅值：活跃前沿高，静默区域低
        if cfg.alpha_lambda > 0:
            gx = np.gradient(self._I_macro_prev, axis=1).astype(np.float32)
            gy = np.gradient(self._I_macro_prev, axis=0).astype(np.float32)
            grad_mag = np.sqrt(gx**2 + gy**2)
            # λ(x,y) = λ_base * exp(-α * |∇I| / (I + ε))
            # 梯度大 → 指数小 → λ 小 → 遗忘慢（记忆深）
            ctx_factor = np.exp(-cfg.alpha_lambda * grad_mag /
                                (self._I_macro_prev + 1e-6)).astype(np.float32)
        else:
            ctx_factor = np.ones((cfg.H, cfg.W), np.float32)

        def lam_field(lam_base):
            lf = lam_base * ctx_factor
            return np.maximum(lf, lam_base * cfg.lam_min_ratio)

        lam_rP  = lam_field(cfg.lam_rate_P)
        lam_rC1 = lam_field(cfg.lam_rate_C1)
        lam_rC2 = lam_field(cfg.lam_rate_C2)
        lam_sC1 = lam_field(cfg.lam_state_C1)
        lam_sC2 = lam_field(cfg.lam_state_C2)

        # ── 步骤1：全域特征张量提取 ──────────────────────────────────
        chi_rate_P  = np.abs(P  - self.P_prev)
        chi_rate_C1 = np.abs(C1 - self.C1_prev)
        chi_rate_C2 = np.abs(C2 - self.C2_prev)
        chi_state_C1 = C1
        chi_state_C2 = C2

        # ── 步骤2：H_sink 极化检测与掩码生成 ─────────────────────────
        # 局部 C2 方差：用 uniform_filter 计算 E[C2²] - E[C2]²
        C2_lmean = uniform_filter(C2, size=cfg.sink_window, mode='wrap')
        C2_lm2   = uniform_filter(C2**2, size=cfg.sink_window, mode='wrap')
        C2_lvar  = np.maximum(C2_lm2 - C2_lmean**2, 0.0)
        # 生成掩码（轻微高斯平滑，避免边界数值不连续）
        raw_mask = (C2_lvar > cfg.theta_sink).astype(np.float32)
        sink_mask = convolve(raw_mask, make_kernel(3, 'gauss'), mode='wrap')
        sink_mask = np.clip(sink_mask, 0.0, 1.0)
        self.sink_mask = sink_mask

        # 极化区域：切断变率反馈（C2 完全切断，C1 部分抑制）
        chi_rate_C2_eff = chi_rate_C2 * (1.0 - sink_mask)
        chi_rate_C1_eff = chi_rate_C1 * (1.0 - sink_mask * 0.5)
        # 被动耗散量（步骤6中使用）
        A_C2_sink = cfg.gamma_sink * sink_mask * C2

        # ── 步骤3：并行执行 L 有损算子（情境自适应版）───────────────
        I_macro          = np.zeros_like(P)
        I_macro_rate_sum = np.zeros_like(P)

        # (chi, stock_attr, k_write, lam_field_arr, weight, is_rate)
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
            # 情境自适应指数遗忘（lam_arr 是空间场）
            stock[:] = stock * (1.0 - lam_arr) + chi * mask * cfg.dt
            norm_stock = stock / (stock + cfg.k_norm_base)
            I_macro += weight * norm_stock
            if is_rate:
                I_macro_rate_sum += weight * norm_stock

        # 保存本帧 I_macro 供下帧计算情境遗忘率
        self._I_macro_prev = I_macro.copy()

        # 兼容性存量（供观测层使用）
        self.I_stock = (self.stock_rate_P + self.stock_rate_C1 + self.stock_rate_C2 +
                        self.stock_state_C1 + self.stock_state_C2)
        # [v7.0 修复] I_increment 只用变率特征群（供 ΔΣ 修复后的观测层使用）
        self.I_increment = (chi_rate_P * cfg.w_rate_P +
                            chi_rate_C1 * cfg.w_rate_C1 +
                            chi_rate_C2 * cfg.w_rate_C2)
        # 变率存量（供 ΔΣ 修复后的 sig_loss 计算）
        self.rate_stock_mean = float(
            (self.stock_rate_P + self.stock_rate_C1 + self.stock_rate_C2).mean())

        # ── 步骤4：执行 D 失真算子 ───────────────────────────────────
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

        # ── 步骤6：形态更新（能-信二元 PDE）─────────────────────────
        S_C2_norm = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        D_C1_eff  = cfg.D_C1 * np.maximum(0.1, 1.0 - cfg.alpha_metric * S_C2_norm)
        D_C2_eff  = cfg.D_C2 * np.maximum(0.1, 1.0 - cfg.alpha_metric * S_C2_norm)

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

        # H_sink 守恒：C2 耗散量完整转移到 P
        dP_dt  = (cfg.D_P  * lap(P)
                  - A_P_to_C1 + A_C1_to_P + A_C1_decay + A_C2_sink)
        dC1_dt = (D_C1_eff * lap(C1)
                  + A_P_to_C1 - A_C1_to_P - A_C1_to_C2 + A_C2_to_C1
                  - A_C1_decay + A_C2_decay)
        dC2_dt = (D_C2_eff * lap(C2)
                  + A_C1_to_C2 - A_C2_to_C1 - A_C2_decay - A_C2_sink)

        # 全局回差法（处理"过于均匀"，与 H_sink 互补）
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


# ─────────────────────────────────────────────
# 四、观测层（修复 ΔΣ 计算）
# ─────────────────────────────────────────────

class ObservationLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = {k: [] for k in [
            'tick', 'M_total', 'f_C2', 'f_C1',
            'delta_sigma', 'Phi_net', 'C_var', 'H_C2',
            'n_clusters', 'size_gini', 'phase'
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

        # [v7.0 修复] sig_loss 只基于变率特征群，避免状态群导致 ΔΣ 恒负
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
        return phase

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

    fields = [
        (phi_C2,             'viridis',  None,            'φ_C2  高阶结构密度\n蓝=低  黄=高'),
        (phys.I_drive,       'RdBu_r',   TwoSlopeNorm(0), 'I_drive  驱动场\n红=生长  蓝=退化  白=零'),
        (phys.sink_mask,     'hot',      None,            '[v7] H_sink掩码\n黑=正常  亮=极化热汇区'),
        (phys._I_norm_vis,   'cividis',  None,            'I_macro归一化  宏观因果场\n深=弱  亮=强'),
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
                  ['f_C2', 'f_C1', 'delta_sigma', 'Phi_net', 'n_clusters', 'size_gini', 'phase']}
    else:
        ticks_plot = ticks
        h_plot = h

    ax_ts = fig.add_subplot(gs[1, :3])
    ax_ts.set_facecolor('#1a1a1a')
    for key, color, lbl in [
        ('f_C2',        '#2ecc71', 'f_C2  高阶结构占比'),
        ('f_C1',        '#3498db', 'f_C1  中间态占比'),
        ('delta_sigma', '#e67e22', 'ΔΣ  信息净增率'),
        ('Phi_net',     '#e74c3c', 'Φ_net  净跃迁通量'),
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
    ax_ts.legend(loc='upper left', fontsize=7,
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
    fig.suptitle(
        f'涌积态宇宙 v7.0   第 {tick:05d} 帧   '
        f'C2占比={h["f_C2"][-1]:.3f}   '
        f'当前阶段：{phase_labels.get(phase_now, phase_now)}',
        color=color_now, fontsize=11, fontweight='bold'
    )

    plt.savefig(OUTPUT_DIR / f'frame_{tick:05d}.png',
                dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f'  [render] tick={tick:05d}  phase={phase_now}  '
          f'f_C2={h["f_C2"][-1]:.3f}  '
          f'n_cl={h["n_clusters"][-1]}  '
          f'Gini={h["size_gini"][-1]:.3f}')


# ─────────────────────────────────────────────
# 六、主循环
# ─────────────────────────────────────────────

def run():
    cfg   = Config()
    phys  = PhysicsLayer(cfg)
    obs   = ObservationLayer(cfg)
    tuner = AdaptiveTuner(cfg) if cfg.adaptive_tuning else None

    print("=" * 60)
    print("涌积态宇宙模型 v7.0 启动")
    print(f"  网格: {cfg.W}×{cfg.H}   dt={cfg.dt}   最大帧数={cfg.steps}")
    print(f"  D_P={cfg.D_P}  D_C1={cfg.D_C1}  D_C2={cfg.D_C2}")
    print(f"  Theta_barrier={cfg.Theta_barrier}  k_noise={cfg.k_noise}")
    print(f"  [v6_ex继承] 广义算子参数：")
    print(f"    变率权重: w_rate_P={cfg.w_rate_P}  w_rate_C1={cfg.w_rate_C1}  w_rate_C2={cfg.w_rate_C2}")
    print(f"    状态权重: w_state_C1={cfg.w_state_C1}  w_state_C2={cfg.w_state_C2}")
    print(f"    遗忘率基线: lam_rate_P={cfg.lam_rate_P}  lam_rate_C1={cfg.lam_rate_C1}  lam_rate_C2={cfg.lam_rate_C2}")
    print(f"    kappa_inertia={cfg.kappa_inertia}  k_entropy={cfg.k_entropy}")
    print(f"  [v7.0新增] 情境自适应遗忘率：alpha_lambda={cfg.alpha_lambda}  lam_min_ratio={cfg.lam_min_ratio}")
    print(f"  [v7.0新增] H_sink热汇：theta_sink={cfg.theta_sink}  gamma_sink={cfg.gamma_sink}  sink_window={cfg.sink_window}")
    print(f"  [v7.0修复] ΔΣ 观测：sig_loss 只基于变率特征群")
    print(f"  自适应调参: {'开启' if cfg.adaptive_tuning else '关闭'}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    for tick in range(cfg.steps):
        phys.step()
        phase = obs.observe(phys, tick)

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
    print(f"  最终区块数:  {h['n_clusters'][-1]}")
    print(f"  最终 Gini:   {h['size_gini'][-1]:.4f}")
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
    phases    = h['phase']

    total_ticks = h['tick'][-1]
    m0, m1 = h['M_total'][0], h['M_total'][-1]
    drift = abs(m1 - m0) / (m0 + 1e-9) * 100

    peak_idx  = int(np.argmax(f_c2))
    peak_tick = int(ticks[peak_idx])
    peak_fc2  = float(f_c2[peak_idx])
    cvar_peak = float(c_var.max())
    cvar_peak_tick = int(ticks[int(np.argmax(c_var))])

    ncl_peak_idx  = int(np.argmax(n_cl))
    ncl_peak_tick = int(ticks[ncl_peak_idx])
    ncl_peak_val  = int(n_cl[ncl_peak_idx])

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
            'phase': phases[idx],
        })

    lines = []
    lines.append("# 涌积态宇宙模型 v7.0 — 运行报告")
    lines.append(f"\n**运行时间**：{_RUN_TIME}  ")
    lines.append(f"**输出目录**：`{OUTPUT_DIR}`\n")
    lines.append("---\n")
    lines.append("## 一、运行参数\n")
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 网格 | {cfg.W}×{cfg.H} |")
    lines.append(f"| dt / steps | {cfg.dt} / {cfg.steps} |")
    lines.append(f"| D_P / D_C1 / D_C2 | {cfg.D_P} / {cfg.D_C1} / {cfg.D_C2} |")
    lines.append(f"| k_noise / Theta_barrier | {cfg.k_noise} / {cfg.Theta_barrier} |")
    lines.append(f"| w_rate_P/C1/C2 | {cfg.w_rate_P} / {cfg.w_rate_C1} / {cfg.w_rate_C2} |")
    lines.append(f"| w_state_C1/C2 | {cfg.w_state_C1} / {cfg.w_state_C2} |")
    lines.append(f"| lam_rate_P/C1/C2 (base) | {cfg.lam_rate_P} / {cfg.lam_rate_C1} / {cfg.lam_rate_C2} |")
    lines.append(f"| lam_state_C1/C2 (base) | {cfg.lam_state_C1} / {cfg.lam_state_C2} |")
    lines.append(f"| kappa_inertia / k_entropy | {cfg.kappa_inertia} / {cfg.k_entropy} |")
    lines.append(f"| **[v7.0]** alpha_lambda | {cfg.alpha_lambda} |")
    lines.append(f"| **[v7.0]** theta_sink / gamma_sink | {cfg.theta_sink} / {cfg.gamma_sink} |")
    lines.append(f"| **[v7.0]** sink_window | {cfg.sink_window} |")
    lines.append(f"| alpha_metric | {cfg.alpha_metric} |")
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
    lines.append(f"| 质量守恒漂移 | {drift:.6f}% |")
    lines.append(f"| f_C2 峰值 | {peak_fc2:.4f}（tick={peak_tick}）|")
    lines.append(f"| C_var 峰值 | {cvar_peak:.5f}（tick={cvar_peak_tick}）|")
    lines.append(f"| 区块数峰值 | {ncl_peak_val}（tick={ncl_peak_tick}）|")
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
    lines.append("| 进度 | tick | f_C2 | f_C1 | ΔΣ | Φ_net | C_var | 区块数 | Gini | 阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(milestones):
        lines.append(
            f"| {i*10}% | {m['tick']} | {m['f_C2']:.4f} | {m['f_C1']:.4f} | "
            f"{m['delta_sig']:.4e} | {m['phi_net']:.4e} | {m['C_var']:.5f} | "
            f"{m['n_cl']} | {m['gini']:.4f} | {m['phase']} |"
        )
    lines.append("\n---\n")
    lines.append("## 六、全量时序数据（每帧）\n")
    lines.append("| tick | f_C2 | f_C1 | ΔΣ | Φ_net | C_var | 区块数 | Gini | 阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i in range(len(ticks)):
        lines.append(
            f"| {int(ticks[i])} | {f_c2[i]:.4f} | {f_c1[i]:.4f} | "
            f"{delta_sig[i]:.4e} | {phi_net[i]:.4e} | {c_var[i]:.5f} | "
            f"{int(n_cl[i])} | {gini_arr[i]:.4f} | {phases[i]} |"
        )

    report_path = OUTPUT_DIR / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n  [报告] 已保存至 {report_path}")


if __name__ == '__main__':
    run()
