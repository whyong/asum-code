"""
涌积态宇宙模型 v12.1 — 纯关系矩阵场架构 (Universe v12.1: Pure Relational Matrix Field)
基于 v12.0（塔蒂尼量子层）与 v11.1（严格统计复现基准），实现底层拓扑的本体论彻底革新：

═════════════════════════════════════════════════════════════════════════════
  核心本体论突破：彻底消除空间假设，宇宙起源于纯数学关系矩阵
═════════════════════════════════════════════════════════════════════════════

  1. 空间假设的消解（Space Elimination）：
     - 彻底摒弃 128×128 二维连续几何网格，代之以 128 个纯抽象节点（N=128）。
     - 节点之间不存在欧几里得距离与坐标，相互作用全部由 128×128 复数权重矩阵 W ∈ ℂ^{N×N} 介导。
     - 拓扑即宇宙，关系即物理——空间不再是先验容器，而是关系强度的宏观涌现表现。

  2. 复数权重的守恒分解定理（Conservative Complex Decomposition）：
     - 任意复权重矩阵 W = K + i·Φ 精确分解为两条数学上严格守恒的代数通道：
       * 对称实部通道 K̃ = (K + Kᵀ)/2  → 保证图拉普拉斯扩散流严格成对抵消（Σ Diff = 0）；
       * 反对称虚部通道 Ã = (Φ - Φᵀ)/2 → 保证相位通量矩阵行和与列和均为零（Σ Adv = 0）。
     - 守恒性内嵌于矩阵代数结构，不依赖任何人工边界或坐标修补。

  3. 动态矩阵度规调制（Emergent Relational Metric）：
     - 节点的宏观存量记忆（stock_state_C1, stock_state_C2）动态调制节点间耦合乘子 M_ij = f(s_i, s_j)；
     - M_ij 保持对称性，使 K̃_eff = K̃ ⊙ M 与 Ã_eff = Ã ⊙ M 始终保持守恒代数结构不变。

  4. 完整量子-经典演化链继承：
     - 逐节点保留塔蒂尼非线性交叉项 β·P² 与局部相位锁定指数（LPI）；
     - 保留振荡计数器、双阈值扩展、自适应调参与死区守恒修正。

运行依赖：numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings, pathlib, os, sys
from datetime import datetime
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR  = pathlib.Path(__file__).parent
_SCRIPT_NAME = pathlib.Path(__file__).stem
_RUN_TIME    = os.environ.get('V12_OUTDIR', datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19])
OUTPUT_DIR   = _SCRIPT_DIR / _SCRIPT_NAME / _RUN_TIME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 一、参数配置
# ─────────────────────────────────────────────

class Config:
    # ── 节点规模（打破空间，纯 128 节点矩阵场）────────────────────────
    N_NODES = 128
    dt      = 0.05
    steps   = int(os.environ.get('V12_STEPS', 24000))

    # ── 可复现随机种子注入 ──────────────────────────────────────────
    rng_seed = int(os.environ['V12_SEED']) if 'V12_SEED' in os.environ else None

    # ── 节点间关系通道输运系数 ──────────────────────────────────────
    D_P    = 0.20
    D_C1   = 0.05
    D_C2   = 0.004

    # 相位通道（虚部反对称流）对流输运系数
    D_adv_P  = 0.05
    D_adv_C1 = 0.02
    D_adv_C2 = 0.002

    # ── 传承失真 ──────────────────────────────────────────────────────
    k_noise   = 0.50

    # ── 拓扑清洗 ──────────────────────────────────────────────────────
    I_max_capacity = 1.0
    D_decay        = 0.15

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
    render_every   = 999999
    struct_every   = 20
    phi_threshold  = None
    hd_window      = 300
    hd_drive_eps   = 5e-5

    # ── 广义算子参数 ──────────────────────────────────────────────────
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

    # ── 情境自适应遗忘率 ─────────────────────────────────────────────
    alpha_lambda  = 1.5
    lam_min_ratio = 0.1

    # ── 零残差拓扑热汇 H_sink ────────────────────────────────────────
    theta_sink  = 0.03
    gamma_sink  = 0.02

    # ── C2 全局不稳定机制（回差法）──────────────────────────────────
    k_instab         = 0.005
    theta_instab_low = 0.005
    theta_instab_high= 0.020

    # ── 关系度规动态调制强度 ─────────────────────────────────────────
    alpha_metric    = 0.3
    alpha_metric_C1 = 0.0

    # ── 早期拓扑特征种子扰动 ─────────────────────────────────────────
    seed_enabled    = True
    seed_fc2_thresh = 0.40
    seed_strength   = 0.002
    seed_adaptive   = True
    seed_cvar_low   = 0.005
    seed_cvar_high  = 0.030

    # ── 低谷期自适应势垒 ─────────────────────────────────────────────
    trough_adapt_enabled  = True
    trough_fc2_thresh     = 0.20
    trough_phi_thresh     = 25.0
    trough_decay_rate     = 0.9999
    trough_theta_min      = 0.18

    # ── 低谷期失真增强 ──────────────────────────────────────────────
    distortion_boost_enabled = True
    distortion_boost_max     = 0.20

    # ── 振荡计数器 ──────────────────────────────────────────────────
    osc_counter_enabled   = True
    osc_count_thresh      = 3
    osc_cvar_window       = 500
    osc_cvar_floor        = 0.03

    # ── 双阈值扩展 ──────────────────────────────────────────────────
    dual_thresh_enabled      = True
    trough_fc2_thresh_ext    = 0.28
    trough_phi_thresh_ext    = 20.0
    trough_decay_rate_ext    = 0.9999
    trough_ext_min_duration  = 2000
    trough_ext_cooldown      = 0      # v11.1 确立最佳无冷却期基准

    # ── 塔蒂尼量子非线性交叉项 ───────────────────────────────────────
    beta_tartini     = float(os.environ.get('V12_BETA', 0.0))
    compute_lpi      = True


# ─────────────────────────────────────────────
# 二、自适应调参模块
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

def softsign(x, cap):
    return x / (1.0 + np.abs(x) / cap)

def gini(arr):
    if len(arr) == 0: return 0.0
    arr = np.sort(arr.astype(float))
    n = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * (idx * arr).sum()) / (n * arr.sum() + 1e-9) - (n + 1) / n


# ─────────────────────────────────────────────
# 三、纯关系矩阵场物理层 (Relational Physics Layer)
# ─────────────────────────────────────────────

class PhysicsLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        N = cfg.N_NODES

        # 随机初始化（支持独立种子注入）
        if cfg.rng_seed is not None:
            np.random.seed(cfg.rng_seed)
            rng = np.random.default_rng(cfg.rng_seed)
        else:
            rng = np.random.default_rng(42)

        # 128 节点一维标量场
        self.P  = rng.uniform(0.2, 1.0, N).astype(np.float32)
        self.C1 = np.zeros(N, np.float32)
        self.C2 = np.zeros(N, np.float32)

        # ── 核心关系矩阵 W ∈ ℂ^{N×N} 构造与谱归一化 ─────────────────
        # 1. 生成复高斯矩阵并提取强复数关联键（稀疏关系网络拓扑，打破空间的同时允许局域非平衡分化）
        W_raw = (rng.normal(0, 1, (N, N)) + 1j * rng.normal(0, 1, (N, N))).astype(np.complex64)
        np.fill_diagonal(W_raw, 0.0)
        
        # 保留每节点最强的 k_bonds 条关系键，形成关系网络拓扑（对标图灵网络理论）
        k_bonds = 6
        abs_W = np.abs(W_raw)
        W_sparse = np.zeros_like(W_raw)
        for i in range(N):
            top_idx = np.argsort(abs_W[i])[-k_bonds:]
            W_sparse[i, top_idx] = W_raw[i, top_idx]
        
        # 2. 守恒分解：对称实部（扩散通道）与 反对称虚部（相位通道）
        K_raw = np.real(W_sparse)
        Phi_raw = np.imag(W_sparse)
        K_sym = (K_raw + K_raw.T) / 2.0
        A_antisym = (Phi_raw - Phi_raw.T) / 2.0
        
        # 清除对角元（自连接在输运差分中自然为零，置零避免干扰谱计算）
        np.fill_diagonal(K_sym, 0.0)
        np.fill_diagonal(A_antisym, 0.0)

        # 3. 构造基础图拉普拉斯矩阵并做谱半径归一化
        deg_K = K_sym.sum(axis=1)
        L_K = np.diag(deg_K) - K_sym
        eigvals, eigvecs = np.linalg.eigh(L_K)
        max_lambda = float(np.max(np.abs(eigvals)))
        
        # 将最大特征值尺度归一到 8.0（与原 128×128 二维网格 5 点拉普拉斯最大特征值等价）
        scale_K = 8.0 / (max_lambda + 1e-9)
        self.K_sym = (K_sym * scale_K).astype(np.float32)
        
        # 相位反对称矩阵同样归一化
        norm_A = float(np.linalg.norm(A_antisym, 2))
        scale_A = 4.0 / (norm_A + 1e-9)
        self.A_antisym = (A_antisym * scale_A).astype(np.float32)

        # 4. 行随机平滑算子 R（行和恒为 1，作用于向量精确保持均值）
        abs_K = np.abs(self.K_sym) + np.eye(N, dtype=np.float32) * 1.0
        self.R = (abs_K / (abs_K.sum(axis=1, keepdims=True) + 1e-9)).astype(np.float32)

        # 5. 早期特征种子模板（提取 K_sym 的第 2~4 个低频非平凡特征向量线性组合）
        # 对应图论中的 Fiedler 大尺度空间拓扑模态
        self._seed_template = (eigvecs[:, 1] + eigvecs[:, 2] * 0.5 + eigvecs[:, 3] * 0.3).astype(np.float32)
        self._seed_template /= (np.abs(self._seed_template).max() + 1e-9)

        # 记录谱统计
        self.max_spectral_radius = max_lambda * scale_K

        # 五轨存量场（128 节点）
        self.stock_rate_P   = np.zeros(N, np.float32)
        self.stock_rate_C1  = np.zeros(N, np.float32)
        self.stock_rate_C2  = np.zeros(N, np.float32)
        self.stock_state_C1 = np.zeros(N, np.float32)
        self.stock_state_C2 = np.zeros(N, np.float32)

        self._I_macro_prev = np.zeros(N, np.float32)
        init_transport_P = - cfg.D_P * (L_K @ self.P) + cfg.D_adv_P * (self.A_antisym @ self.P)
        self.P_prev  = np.maximum(self.P + init_transport_P * cfg.dt, 0.0)
        self.C1_prev = self.C1.copy()
        self.C2_prev = self.C2.copy()

        # 中间观测量
        self.I_stock         = np.zeros(N, np.float32)
        self.I_increment     = np.zeros(N, np.float32)
        self.I_drive         = np.zeros(N, np.float32)
        self.A_C1_to_C2      = np.zeros(N, np.float32)
        self.A_C2_to_C1      = np.zeros(N, np.float32)
        self._I_norm_vis     = np.zeros(N, np.float32)
        self.sink_mask       = np.zeros(N, np.float32)
        self.rate_stock_mean = 0.0
        self._instab_active  = False

        # v11.1 继承状态
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

        # v12.0 塔蒂尼观测量
        self.A_tartini       = np.zeros(N, np.float32)
        self.lpi             = 0.0
        self.tartini_total   = 0.0

        # v12.1 关系场特有观测量
        self.mean_eff_coupling = float(np.abs(self.K_sym).mean())
        self.phase_flux_total  = 0.0
        self.K_eff_vis         = self.K_sym.copy()

    def step(self, tick: int = 0, c_var: float = 0.0, phase: str = 'CHAOTIC_EDGE'):
        cfg = self.cfg
        N = cfg.N_NODES
        P, C1, C2 = self.P, self.C1, self.C2

        # ── 步骤0：情境自适应遗忘率场（基于图连通性的梯度）─────────
        if cfg.alpha_lambda > 0:
            # 节点信息流强度的图导数计算：g_i = sqrt(Σ_j K_ij * (I_j - I_i)^2)
            dI = self._I_macro_prev[None, :] - self._I_macro_prev[:, None] # (N, N)
            grad_mag = np.sqrt(np.maximum(0.0, np.sum(np.abs(self.K_sym) * (dI ** 2), axis=1))).astype(np.float32)
            grad_mag /= (grad_mag.max() + 1e-6)
            ctx_factor = np.exp(-cfg.alpha_lambda * grad_mag /
                                (self._I_macro_prev + 1e-6)).astype(np.float32)
        else:
            ctx_factor = np.ones(N, np.float32)

        def lam_field(lam_base):
            return np.maximum(lam_base * ctx_factor, lam_base * cfg.lam_min_ratio)

        lam_rP  = lam_field(cfg.lam_rate_P)
        lam_rC1 = lam_field(cfg.lam_rate_C1)
        lam_rC2 = lam_field(cfg.lam_rate_C2)
        lam_sC1 = lam_field(cfg.lam_state_C1)
        lam_sC2 = lam_field(cfg.lam_state_C2)

        # ── 步骤1：特征张量提取（节点级）────────────────────────────
        chi_rate_P  = np.abs(P  - self.P_prev)
        chi_rate_C1 = np.abs(C1 - self.C1_prev)
        chi_rate_C2 = np.abs(C2 - self.C2_prev)
        chi_state_C1 = C1
        chi_state_C2 = C2

        # ── 步骤2：H_sink 局部邻域方差检测 ──────────────────────────
        # 用行随机矩阵 R 计算邻域加权一阶矩与二阶矩
        C2_lmean = self.R @ C2
        C2_lm2   = self.R @ (C2 ** 2)
        C2_lvar  = np.maximum(C2_lm2 - C2_lmean ** 2, 0.0)
        raw_mask = (C2_lvar > cfg.theta_sink).astype(np.float32)
        sink_mask = np.clip(self.R @ raw_mask, 0.0, 1.0)
        self.sink_mask = sink_mask

        chi_rate_C2_eff = chi_rate_C2 * (1.0 - sink_mask)
        chi_rate_C1_eff = chi_rate_C1 * (1.0 - sink_mask * 0.5)
        A_C2_sink = cfg.gamma_sink * sink_mask * C2

        # ── 步骤3：L 有损算子（节点五轨积分）────────────────────────
        I_macro          = np.zeros(N, np.float32)
        I_macro_rate_sum = np.zeros(N, np.float32)

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
            mask = (np.random.rand(N) < p_write).astype(np.float32)
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

        # ── 步骤4：D 失真算子（图拉普拉斯惯性 + 节点噪声）────────────
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

        # R 平滑替代高斯/均值卷积
        I_context   = self.R @ I_macro
        S_norm_C1   = self.stock_state_C1 / (self.stock_state_C1 + cfg.k_norm_base)
        S_norm_C2   = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        S_macro     = cfg.w_state_C1 * S_norm_C1 + cfg.w_state_C2 * S_norm_C2
        
        # 惯性通量使用图拉普拉斯算子
        laplacian_S = - (np.diag(self.K_sym.sum(axis=1)) - self.K_sym) @ S_macro
        noise       = np.random.randn(N).astype(np.float32) * k_noise_eff * I_context
        I_distorted = I_context + cfg.kappa_inertia * laplacian_S + noise

        # ── 步骤5：柔性极化与死区清洗 ────────────────────────────────
        Omega_raw    = softsign(I_distorted, cfg.I_max_capacity)
        Omega_damped = (1.0 - cfg.D_decay) * (self.R @ Omega_raw)
        I_drive      = np.where(np.abs(Omega_damped) < 1e-4, 0.0, Omega_damped)
        self.I_drive     = I_drive
        self._I_norm_vis = I_macro / (I_macro.max() + 1e-9)

        # ── 步骤6：动态关系度规调制（保持对称/反对称守恒结构）────────
        S_C2_norm = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        
        # 节点对间的对称调制因子 M_ij = max(0.1, 1 - alpha * sqrt(s_i * s_j))
        s_pair = np.sqrt(S_C2_norm[:, None] * S_C2_norm[None, :]) # 对称矩阵
        metric_matrix = np.maximum(0.1, 1.0 - cfg.alpha_metric * s_pair).astype(np.float32)
        
        # 动态有效对称扩散核与有效反对称相位核
        K_eff = (self.K_sym * metric_matrix).astype(np.float32)
        A_eff = (self.A_antisym * metric_matrix).astype(np.float32)
        
        # 强制数值对称/反对称保障
        K_eff = (K_eff + K_eff.T) / 2.0
        A_eff = (A_eff - A_eff.T) / 2.0
        np.fill_diagonal(K_eff, 0.0)
        np.fill_diagonal(A_eff, 0.0)
        
        self.K_eff_vis = K_eff
        self.mean_eff_coupling = float(np.abs(K_eff).mean())

        # 有效图拉普拉斯矩阵
        L_eff = np.diag(K_eff.sum(axis=1)) - K_eff

        # ── 步骤7：形态跃迁与严格守恒关系输运 ────────────────────────
        A_P_to_C1  = np.maximum(0,  I_drive) * cfg.Base_Rate_1 * P
        A_C1_to_P  = np.abs(np.minimum(0, I_drive)) * cfg.Base_Rate_1 * C1
        A_C1_to_C2 = np.maximum(0, I_drive - cfg.Theta_barrier) * cfg.Base_Rate_2 * C1
        A_C2_to_C1 = np.abs(np.minimum(0, I_drive)) * cfg.Base_Rate_dec * C2

        A_C1_decay = cfg.decay_C1 * C1
        decay_C2_dyn = cfg.decay_C2 + cfg.k_entropy * I_macro_rate_sum.mean()
        A_C2_decay   = decay_C2_dyn * C2

        self.A_C1_to_C2 = A_C1_to_C2
        self.A_C2_to_C1 = A_C2_to_C1

        # 塔蒂尼量子项：A_tartini = beta * P^2
        if cfg.beta_tartini > 0:
            A_tartini = np.maximum(0.0, cfg.beta_tartini * P * P).astype(np.float32)
        else:
            A_tartini = np.zeros(N, np.float32)
        self.A_tartini = A_tartini
        self.tartini_total = float(A_tartini.sum())

        self.P_prev  = P.copy()
        self.C1_prev = C1.copy()
        self.C2_prev = C2.copy()

        # 记录更新前总质量
        M_before = float(P.sum() + C1.sum() + C2.sum())

        # 严格守恒的图输运项：
        # 1. 对称通道扩散流：- D * L_eff @ X = D * Σ_j K_eff_ij * (X_j - X_i) (和为 0)
        # 2. 反对称通道相位流：D_adv * A_eff @ X (和为 0)
        transport_P  = - cfg.D_P  * (L_eff @ P)  + cfg.D_adv_P  * (A_eff @ P)
        transport_C1 = - cfg.D_C1 * (L_eff @ C1) + cfg.D_adv_C1 * (A_eff @ C1)
        transport_C2 = - cfg.D_C2 * (L_eff @ C2) + cfg.D_adv_C2 * (A_eff @ C2)

        self.phase_flux_total = float(np.abs(A_eff @ P).sum() + np.abs(A_eff @ C1).sum())

        dP_dt  = (transport_P
                  - A_P_to_C1 + A_C1_to_P + A_C1_decay + A_C2_sink
                  - A_tartini)

        dC1_dt = (transport_C1
                  + A_P_to_C1 - A_C1_to_P - A_C1_to_C2 + A_C2_to_C1
                  - A_C1_decay + A_C2_decay
                  + A_tartini)

        dC2_dt = (transport_C2
                  + A_C1_to_C2 - A_C2_to_C1 - A_C2_decay - A_C2_sink)

        # 全局回差法
        C2_var = float(C2.var())
        if C2_var < cfg.theta_instab_low:
            self._instab_active = True
        elif C2_var > cfg.theta_instab_high:
            self._instab_active = False
        if self._instab_active and C2.mean() > 1e-6:
            noise_instab = np.random.randn(N).astype(np.float32)
            A_instab = cfg.k_instab * C2 * noise_instab
            A_instab -= A_instab.sum() / N
            dC2_dt = dC2_dt + A_instab

        self.P  = np.maximum(P  + dP_dt  * cfg.dt, 0.0)
        self.C1 = np.maximum(C1 + dC1_dt * cfg.dt, 0.0)
        self.C2 = np.maximum(C2 + dC2_dt * cfg.dt, 0.0)

        # 截断后守恒修正（带 0.001 死区）
        M_after = float(self.P.sum() + self.C1.sum() + self.C2.sum())
        mass_change = M_after - M_before
        if abs(mass_change) > 0.001 and M_after > 1e-9:
            scale_factor = M_before / M_after
            self.P  *= scale_factor
            self.C1 *= scale_factor
            self.C2 *= scale_factor

        # 局部相位锁定指数（LPI）
        if cfg.compute_lpi:
            var_P  = float(self.P.var())
            var_C1 = float(self.C1.var())
            self.lpi = var_C1 / (var_P + var_C1 + 1e-9)
        else:
            self.lpi = 0.0

        # ── 步骤8：拓扑特征种子扰动 ─────────────────────────────────
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

        # ── 步骤9：振荡计数器 + 双阈值扩展 ──────────────────────────
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

            if (cfg.dual_thresh_enabled
                    and self._osc_confirmed
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
                if cfg.trough_ext_cooldown > 0:
                    self._ext_below_count = 0
                    self._ext_cooldown_remaining = cfg.trough_ext_cooldown

        else:
            self._cvar_window_buf.clear()
            self._cvar_above_mean = False
            self._ext_below_count = 0
            self._trough_active = False
            self.dual_thresh_active = False


# ─────────────────────────────────────────────
# 四、观测层
# ─────────────────────────────────────────────

class ObservationLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = {k: [] for k in [
            'tick', 'M_total', 'f_C2', 'f_C1', 'f_P',
            'delta_sigma', 'Phi_net', 'C_var', 'H_C2',
            'n_clusters', 'size_gini', 'phase', 'Theta_barrier',
            'k_noise_eff', 'osc_count', 'dual_thresh_active',
            'lpi', 'tartini_flux', 'tartini_cumul',
            'mean_eff_coupling', 'phase_flux', # v12.1 新增
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

        # 节点场基尼系数与活跃节点数
        n_cl, s_gini = self._struct_stats(phi_C2)

        self._hd_buf.append((phys.I_increment.max(), abs(phys.I_drive).max()))
        if len(self._hd_buf) > cfg.hd_window:
            self._hd_buf.pop(0)

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
        h['mean_eff_coupling'].append(phys.mean_eff_coupling)
        h['phase_flux'].append(phys.phase_flux_total)
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
        threshold = phi_C2.mean() + 0.5 * phi_C2.std()
        active_nodes = int((phi_C2 > threshold).sum())
        s_gini = float(gini(phi_C2))
        return active_nodes, s_gini


# ─────────────────────────────────────────────
# 五、可视化层（关系矩阵与节点分布全景）
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

    # 上排 5 个关系/节点图谱
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(phys.K_eff_vis, cmap='viridis', interpolation='nearest')
    ax1.set_title('K_eff 有效对称扩散矩阵\n(128×128 节点关系度规)', color='white', fontsize=7, pad=3)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02).ax.tick_params(labelcolor='white', labelsize=5)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(phys.A_antisym, cmap='RdBu_r', interpolation='nearest', norm=TwoSlopeNorm(0))
    ax2.set_title('A_eff 反对称相位通道矩阵\n(复数虚部旋度流动)', color='white', fontsize=7, pad=3)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02).ax.tick_params(labelcolor='white', labelsize=5)

    # 节点态分布折线
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#1a1a1a')
    node_idx = np.arange(cfg.N_NODES)
    ax3.plot(node_idx, P,  color='#3498db', lw=1.0, label='P 概率波')
    ax3.plot(node_idx, C1, color='#e67e22', lw=1.0, label='C1 坍缩态')
    ax3.plot(node_idx, C2, color='#2ecc71', lw=1.0, label='C2 退相干态')
    ax3.legend(loc='upper right', fontsize=5, facecolor='#1a1a1a', labelcolor='white')
    ax3.tick_params(colors='white', labelsize=6)
    ax3.set_xlabel('节点序号 (0~127)', color='white', fontsize=7)
    ax3.set_title('128 节点标量场分布', color='#aaaaaa', fontsize=7, pad=3)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor('#1a1a1a')
    sorted_phi = np.sort(phi_C2)[::-1]
    ax4.bar(node_idx, sorted_phi, color='#9b59b6', width=0.8)
    ax4.tick_params(colors='white', labelsize=6)
    ax4.set_xlabel('降序节点排列', color='white', fontsize=7)
    ax4.set_title(f'φ_C2 节点分布谱 (Gini={obs.history["size_gini"][-1]:.3f})', color='#aaaaaa', fontsize=7, pad=3)

    ax5 = fig.add_subplot(gs[0, 4])
    ax5.set_facecolor('#1a1a1a')
    ax5.plot(node_idx, phys.I_drive, color='#e74c3c', lw=1.0, label='I_drive 驱动')
    ax5.plot(node_idx, phys._I_norm_vis, color='#f1c40f', lw=1.0, label='I_macro 宏观因果')
    ax5.legend(loc='upper right', fontsize=5, facecolor='#1a1a1a', labelcolor='white')
    ax5.tick_params(colors='white', labelsize=6)
    ax5.set_xlabel('节点序号', color='white', fontsize=7)
    ax5.set_title('因果驱动力分布', color='#aaaaaa', fontsize=7, pad=3)

    # 下排时序曲线
    h = obs.history
    ticks = h['tick']
    _n = len(ticks)
    if _n > 1000:
        _step = _n // 1000
        _idx  = list(range(0, _n, _step)) + [_n - 1]
        ticks_plot = [ticks[i] for i in _idx]
        h_plot = {k: [h[k][i] for i in _idx] for k in
                  ['f_C2', 'f_C1', 'f_P', 'C_var', 'Phi_net', 'Theta_barrier',
                   'lpi', 'phase_flux', 'osc_count', 'dual_thresh_active', 'phase']}
    else:
        ticks_plot = ticks
        h_plot = h

    ax_ts = fig.add_subplot(gs[1, :3])
    ax_ts.set_facecolor('#1a1a1a')
    for key, color, lbl in [
        ('f_P',           '#3498db', 'f_P 概率波占比'),
        ('f_C1',          '#e67e22', 'f_C1 坍缩态占比'),
        ('f_C2',          '#2ecc71', 'f_C2 退相干实在占比'),
        ('C_var',         '#e74c3c', 'C_var 节点方差'),
        ('Theta_barrier', '#f39c12', 'Θ 势垒'),
        ('lpi',           '#9b59b6', 'LPI 相位锁定指数'),
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
    ax_ts.legend(loc='upper left', fontsize=6, facecolor='#1a1a1a', labelcolor='white', framealpha=0.6)
    ax_ts.tick_params(colors='white', labelsize=7)
    ax_ts.set_xlabel('演化步数 (tick)', color='white', fontsize=8)
    ax_ts.set_title('关系矩阵场全局时序  （背景色带 = 演化阶段）', color='#aaaaaa', fontsize=8, pad=4)

    ax_ph = fig.add_subplot(gs[1, 3])
    ax_ph.set_facecolor('#1a1a1a')
    ax_ph.plot(ticks_plot, h_plot['phase_flux'], color='#1abc9c', lw=1.0, label='相位通道总通量')
    ax_ph.legend(loc='upper left', fontsize=6, facecolor='#1a1a1a', labelcolor='white')
    ax_ph.tick_params(colors='white', labelsize=7)
    ax_ph.set_xlabel('tick', color='white', fontsize=8)
    ax_ph.set_title('反对称虚部环流', color='#aaaaaa', fontsize=8, pad=4)

    ax_lpi = fig.add_subplot(gs[1, 4])
    ax_lpi.set_facecolor('#1a1a1a')
    ax_lpi.plot(ticks_plot, h_plot['lpi'], color='#9b59b6', lw=1.0)
    ax_lpi.plot(ticks_plot, h_plot['osc_count'], color='#e91e63', lw=1.0, label='振荡计数')
    ax_lpi.legend(loc='upper left', fontsize=6, facecolor='#1a1a1a', labelcolor='white')
    ax_lpi.tick_params(colors='white', labelsize=7)
    ax_lpi.set_title('LPI 指数与振荡周波数', color='#aaaaaa', fontsize=8, pad=4)

    phase_now = h['phase'][-1] if h['phase'] else '—'
    color_now = PHASE_COLORS.get(phase_now, 'white')
    fig.suptitle(
        f'涌积态宇宙 v12.1（纯关系矩阵场）   第 {tick:05d} 帧   '
        f'N=128 节点   β={cfg.beta_tartini:.4f}   '
        f'C_var={h["C_var"][-1]:.4f}   LPI={h["lpi"][-1]:.3f}   '
        f'阶段：{phase_now}',
        color=color_now, fontsize=10, fontweight='bold'
    )

    plt.savefig(OUTPUT_DIR / f'frame_{tick:05d}.png',
                dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────
# 六、主循环
# ─────────────────────────────────────────────

def run():
    cfg   = Config()
    phys  = PhysicsLayer(cfg)
    obs   = ObservationLayer(cfg)
    tuner = AdaptiveTuner(cfg) if cfg.adaptive_tuning else None

    print("=" * 70)
    print("涌积态宇宙模型 v12.1 — 纯关系矩阵场架构启动")
    print(f"  节点规模: N={cfg.N_NODES} (打破空间场)   dt={cfg.dt}   最大帧数={cfg.steps}")
    print(f"  随机种子: {cfg.rng_seed}   [v12] beta_tartini={cfg.beta_tartini}")
    print(f"  关系谱半径: λ_max={phys.max_spectral_radius:.4f} (已归一化，稳定性对齐)")
    print(f"  守恒分解: 对称扩散 (K_sym) + 反对称相位流 (A_antisym) (严格代数守恒)")
    print(f"  动态度规: alpha_metric={cfg.alpha_metric} (对称乘子矩阵调制)")
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
                  f'Θ={h["Theta_barrier"][-1]:.4f}  '
                  f'osc={h["osc_count"][-1]}')

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
    print("\n── 关系矩阵场演化摘要 ────────────────────────────")
    print(f"  总帧数:          {h['tick'][-1]}")
    print(f"  最终 f_P:        {h['f_P'][-1]:.4f}")
    print(f"  最终 f_C1:       {h['f_C1'][-1]:.4f}")
    print(f"  最终 f_C2:       {h['f_C2'][-1]:.4f}")
    print(f"  最终 LPI:        {h['lpi'][-1]:.4f}")
    print(f"  C_var 峰值:      {max(h['C_var']):.5f}")
    print(f"  最终阶段:        {h['phase'][-1]}")
    print(f"  最终 Theta:      {h['Theta_barrier'][-1]:.4f}")
    print(f"  最终振荡计数:    {h['osc_count'][-1]}")
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
    theta_arr = np.array(h['Theta_barrier'])
    osc_arr   = np.array(h['osc_count'])
    dual_arr  = np.array(h['dual_thresh_active'])
    gini_arr  = np.array(h['size_gini'])
    p_flux    = np.array(h['phase_flux'])
    phases    = h['phase']

    total_ticks = h['tick'][-1]
    m0, m1 = h['M_total'][0], h['M_total'][-1]
    drift = abs(m1 - m0) / (m0 + 1e-9) * 100

    cvar_peak_idx = int(np.argmax(c_var))
    lpi_peak_idx  = int(np.argmax(lpi_arr))
    dual_first = int(np.argmax(dual_arr > 0)) if dual_arr.max() > 0 else -1

    phase_changes = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            phase_changes.append((int(ticks[i]), phases[i-1], phases[i]))

    milestones = []
    for pct in range(0, 101, 10):
        idx = min(int(pct / 100 * (len(ticks) - 1)), len(ticks) - 1)
        milestones.append({
            'tick': int(ticks[idx]),
            'f_P': float(f_p[idx]), 'f_C1': float(f_c1[idx]), 'f_C2': float(f_c2[idx]),
            'C_var': float(c_var[idx]), 'lpi': float(lpi_arr[idx]),
            'theta': float(theta_arr[idx]), 'osc': int(osc_arr[idx]),
            'gini': float(gini_arr[idx]), 'phase_flux': float(p_flux[idx]),
            'dual': int(dual_arr[idx]), 'phase': phases[idx],
        })

    lines = []
    lines.append("# 涌积态宇宙模型 v12.1 — 关系矩阵场运行报告")
    lines.append(f"\n**运行时间**：{_RUN_TIME}  ")
    lines.append(f"**输出目录**：`{OUTPUT_DIR}`  ")
    lines.append(f"**RNG 种子**：`{cfg.rng_seed}`  ")
    lines.append(f"**塔蒂尼耦合系数 β**：`{cfg.beta_tartini}`\n")
    lines.append("---\n")
    lines.append("## 一、核心关系参数\n")
    lines.append("| 参数 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| 节点数 N | {cfg.N_NODES} | 纯拓扑抽象节点 |")
    lines.append(f"| dt / steps | {cfg.dt} / {cfg.steps} | 时步与离散帧数 |")
    lines.append(f"| rng_seed | {cfg.rng_seed} | 确定性随机种子 |")
    lines.append(f"| beta_tartini | {cfg.beta_tartini} | 塔蒂尼量子非线性耦合 |")
    lines.append(f"| alpha_metric | {cfg.alpha_metric} | 动态关系度规调制强度 |")
    lines.append(f"| D_P / D_C1 / D_C2 | {cfg.D_P} / {cfg.D_C1} / {cfg.D_C2} | 对称通道扩散常数 |")
    lines.append(f"| D_adv_P / C1 / C2 | {cfg.D_adv_P} / {cfg.D_adv_C1} / {cfg.D_adv_C2} | 反对称通道相位对流常数 |")
    lines.append(f"| trough_ext_cooldown | {cfg.trough_ext_cooldown} | 扩展冷却期（0=已移除） |")
    lines.append(f"| adaptive_tuning | {cfg.adaptive_tuning} | 自适应调参状态 |")
    lines.append("\n---\n")
    lines.append("## 二、总体摘要\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 实际运行帧数 | {total_ticks} |")
    lines.append(f"| 终止原因 | {'热寂' if total_ticks < cfg.steps - 1 else '达到最大帧数'} |")
    lines.append(f"| 最终 f_P | {f_p[-1]:.4f} |")
    lines.append(f"| 最终 f_C1 | {f_c1[-1]:.4f} |")
    lines.append(f"| 最终 f_C2 | {f_c2[-1]:.4f} |")
    lines.append(f"| 最终 LPI | {lpi_arr[-1]:.4f} |")
    lines.append(f"| C_var 峰值 | {float(c_var[cvar_peak_idx]):.5f}（tick={int(ticks[cvar_peak_idx])}）|")
    lines.append(f"| LPI 峰值 | {float(lpi_arr[lpi_peak_idx]):.4f}（tick={int(ticks[lpi_peak_idx])}）|")
    lines.append(f"| 最终演化阶段 | {phases[-1]} |")
    lines.append(f"| 最终 Theta_barrier | {theta_arr[-1]:.4f} |")
    lines.append(f"| 最终振荡计数 | {int(osc_arr[-1])} |")
    lines.append(f"| 扩展阈值首次激活 | {'tick=' + str(int(ticks[dual_first])) if dual_first >= 0 else '未触发'} |")
    lines.append(f"| 质量守恒漂移 | {drift:.6f}% |")
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
    lines.append("| 进度 | tick | f_P | f_C1 | f_C2 | C_var | LPI | Gini | Θ | 振荡 | 相位通量 | 阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(milestones):
        lines.append(
            f"| {i*10}% | {m['tick']} | {m['f_P']:.4f} | {m['f_C1']:.4f} | {m['f_C2']:.4f} | "
            f"{m['C_var']:.5f} | {m['lpi']:.4f} | {m['gini']:.4f} | {m['theta']:.4f} | "
            f"{m['osc']} | {m['phase_flux']:.2e} | {m['phase']} |"
        )
    lines.append("\n---\n")
    lines.append("## 六、全量时序数据（每帧）\n")
    lines.append("| tick | f_P | f_C1 | f_C2 | C_var | LPI | Gini | Θ | 振荡 | 阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in range(len(ticks)):
        lines.append(
            f"| {int(ticks[i])} | {f_p[i]:.4f} | {f_c1[i]:.4f} | {f_c2[i]:.4f} | "
            f"{c_var[i]:.5f} | {lpi_arr[i]:.4f} | {gini_arr[i]:.4f} | {theta_arr[i]:.4f} | "
            f"{int(osc_arr[i])} | {phases[i]} |"
        )

    report_path = OUTPUT_DIR / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n  [报告] 已保存至 {report_path}")


if __name__ == '__main__':
    run()
