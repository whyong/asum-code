"""
涌积态宇宙模型 v13.3 — 地球自然演化与生态相变引擎（双热源完备版）
(ASUM v13.3: Earth Natural Evolution & Ecological Phase Transition Engine - Dual Heat Source)

  v13.3 核心新增（地热能系统）：
    【重大】增加岩浆地热作为第二能量来源（恒定，不受灾变影响）
      geo_heat_amplitude = 0.004  地热基础热流（≈太阳泵的5%，生态放大）
      geo_hotspot_count  = 5      热点数量（板块边界/火山岛链）
      geo_hotspot_radius = 10     热点半径（格点）
      geo_to_C1_frac     = 0.30   地热直接催化C1比例（化能合成）
    → 地热热点 = 固定空间锚点，Turing斑块优先成核位置
    → 灾变核冬天时地热不中断 = 热液喷口生命避难所
    → 70%地热→P（热液矿化）+ 30%地热→C1（化能合成，绕过光合）
  v13.2 继承：废热辐射出口（sigma_rad_P/C1/C2），热力学耗散稳态
  v13.1 继承：gamma_drift=0.15  p_catastrophe=2.5e-5  lambda_eco=0.05
  v13.2 继承：Theta_barrier=0.18  Base_Rate_2=0.60  Mstar_accum_rate=0.008

═══════════════════════════════════════════════════════════════════
  核心范式跃迁：从"孤立宇宙"→ "开放耗散地球生态圈"
═══════════════════════════════════════════════════════════════════

  变量语义重映射（v13.0 生态域）：
    P  场  → 无机环境/游离可用能量（阳光、水汽、无机盐）
    A（作用）→ 生物化学反应事件（光合作用、捕食、代谢）
    C1 场  → 生物个体/瞬态种群（单株植物、游荡兽群）
    C2 场  → 顶极生态群落/地貌（热带雨林、珊瑚礁、坚固地质层）
    D（阳）→ 基因突变/灾变（micro: DNA漂变；macro: 陨石/冰川）
    L（阴）→ 自然选择/熵增耗散（生存过滤、食物链能量损耗）
    M*记忆 → 化石地层/基因库（历史生命痕迹与成功经验刻印）
    I*信息 → 生物多样性/食物网复杂度（相变门控阈值）

═══════════════════════════════════════════════════════════════════
  v13.0 四大核心新增
═══════════════════════════════════════════════════════════════════

  1. Fokker-Planck 方程（东西方双向因果）：
       ∂P/∂t = ∇·(D_eff∇P)              # 西方：盲目微观扩散
              - ∇·(P·V_macro)           # 东方：宏观生态位牵引
              + P_in(x,y,t)             # 太阳泵（开放系统输入）
              - A_{P→C1} - Decay

  2. 60甲子节律引擎（天干×地支双环振荡器）：
       P_in(x,y,t) = A_solar·(1+sin(2πt/10)+sin(2πt/12))/2·mask(x,y)

  3. L算子→环境容量K动态剪裁：
       Decay_C1 = λ·C1·exp(C1/(P+ε)-1)   # 生态饥荒指数耗散

  4. D算子双轨：D_micro（每帧基因突变） + D_macro（泊松灾变脉冲）

═══════════════════════════════════════════════════════════════════
  三大内置仿真实验
═══════════════════════════════════════════════════════════════════
  experiment='abiogenesis' → 原始汤无生源说涌现
  experiment='biome'       → 生物圈图灵划分（森林/草原/沙漠）
  experiment='extinction'  → 第六次大灭绝与哺乳动物崛起
  experiment='free'        → 自由演化（默认）

运行依赖：numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.ndimage import convolve, label, uniform_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
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
    steps  = 60000   # v13 默认步数（支撑灭绝实验，翻倍以观察完整演化周期）

    # ── 扩散系数 ──────────────────────────────────────────────────────
    D_P    = 0.20    # P（无机能量）扩散率
    D_C1   = 0.05    # C1（生物个体）扩散率
    D_C2   = 0.004   # C2（群落/地貌）扩散率

    # ── [v13.0 新增] 开放系统：太阳泵 ────────────────────────────────
    A_solar          = 0.08    # 太阳泵基础振幅（游离能量注入）
    P_in_spatial_var = 0.30    # 阳光空间异质性（0=均匀，1=强烈斑驳）
    P_in_blocked     = False   # 灾变实验：阻断阳光（核冬天）

    # ── [v13.0 新增] 60甲子节律引擎 ──────────────────────────────────
    tian_gan_period  = 10      # 天干周期（快变，温度/光照涨落）
    di_zhi_period    = 12      # 地支周期（慢变，水循环/洋流）
    jiazi_amplitude  = 0.40    # 甲子节律对 P_in 的调制深度

    # ── [v13.0 新增] Fokker-Planck 漂移场 ────────────────────────────
    gamma_drift      = 0.15    # 宏观漂移强度 γ（生态位引力强度）[v13.1: 0.20→0.15，防C1超载]
    alpha_Mstar      = 1.0     # M* 在漂移势中的权重
    alpha_Istar      = 0.5     # I* 在漂移势中的权重
    Mstar_accum_rate = 0.008   # M* 积累速率（C2→M* 沉积）[v13.2: 0.005→0.008，增强记忆引力]
    Mstar_decay_rate = 0.9998  # M* 每帧衰减（地质时间尺度缓慢侵蚀）

    # ── [v13.0 新增] L 算子：生态容量竞争 ────────────────────────────
    lambda_eco       = 0.05    # C1 生态竞争死亡率基值 λ  [v13.1: 0.08→0.05，给跃迁更多时间]
    eps_carrying     = 1e-3    # 防零除 ε

    # ── [v13.0 新增] D 算子双轨 ──────────────────────────────────────
    k_noise_micro    = 0.30    # D_micro 乘性噪声强度（日常基因突变）
    p_catastrophe    = 2.5e-5  # D_macro 每帧触发概率（≈40000帧1次）[v13.1: 5e-5→2.5e-5，降频]
    catastrophe_c1_suppress = 0.05   # 灾变后 C1 保留比例
    catastrophe_c2_suppress = 0.20   # 灾变后 C2 保留比例  [v13.1: 0.10→0.20，留足复苏种子]
    catastrophe_duration     = 200   # 灾变后 P_in 压制持续帧数

    # ── 跃迁势垒与速率 ────────────────────────────────────────────────
    Theta_barrier  = 0.18    # [v13.2: 0.28→0.18，开放系统势垒降低，释放C1→C2通量]
    Base_Rate_1    = 1.0
    Base_Rate_2    = 0.60    # [v13.2: 0.40→0.60，配合低势垒加速C2积累]
    Base_Rate_dec  = 1.2
    decay_C2       = 0.006

    # ── [v13.2 新增] 废热辐射出口（向宇宙空间）──────────────────────
    # 物理意义：地球吸收低熵阳光，排出高熵废热，在夹缝中维持有序结构
    # 三态辐射率不等 → 进化方向的热力学驱动力（C2最节能，故自然选择偏向C2）
    sigma_rad_P    = 0.006   # P（无机游离能）辐射率，最高：高熵态容易耗散
    sigma_rad_C1   = 0.002   # C1（生物个体）辐射率：代谢热损耗/呼吸蒸腾
    sigma_rad_C2   = 0.0004  # C2（顶极群落）辐射率，最低：有序结构维持效率高

    # ── [v13.3 新增] 地热能系统（岩浆散热/板块构造）────────────────
    # 物理意义：地球内部放射性衰变+原始热量→持续散热→独立于太阳的能量来源
    # 地热 ≠ 太阳：恒定不变、空间固定（热点锚定）、灾变时不中断
    # 30%地热直接催化C1（化能合成）= 热液喷口生态系统（不依赖光合作用）
    geo_heat_amplitude = 0.004   # 地热基础热流（≈太阳泵A_solar的5%）
    geo_hotspot_count  = 5       # 固定热点数（板块边界/火山岛链）
    geo_hotspot_radius = 10      # 热点高斯半径（格点）
    geo_to_C1_frac     = 0.30    # 地热直接催化C1比例（化能合成路径）

    # ── 传承失真（D_micro 空间结构）─────────────────────────────────
    blur_size = 5
    gauss_size= 5

    # ── 有损算子 L（继承 v12 广义算子骨架）──────────────────────────
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
    k_norm_base = 0.002

    # ── 涌现度规（继承 v12）──────────────────────────────────────────
    alpha_metric    = 0.3
    alpha_metric_C1 = 0.0

    # ── 自适应调参 ────────────────────────────────────────────────────
    adaptive_tuning = True
    adapt_interval  = 50

    # ── 观测 ──────────────────────────────────────────────────────────
    render_every  = 50
    struct_every  = 20
    phi_threshold = None
    hd_window     = 300
    hd_drive_eps  = 5e-5

    # ── [v13.0] 三大实验预设 ─────────────────────────────────────────
    # 'free' | 'abiogenesis' | 'biome' | 'extinction'
    experiment    = 'free'

    # ── [v12 继承，生态语义：自催化反应] 塔蒂尼项 ────────────────────
    beta_tartini  = 0.001

    # ── 熵产自适应退化 ────────────────────────────────────────────────
    k_entropy = 0.01


# ─────────────────────────────────────────────
# 二、工具函数
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

def div_diff(field, D_field):
    """守恒非均匀扩散 ∇·(D∇C)，周期边界"""
    D_xp = 0.5*(D_field + np.roll(D_field,-1,axis=1))
    D_xm = 0.5*(D_field + np.roll(D_field, 1,axis=1))
    D_yp = 0.5*(D_field + np.roll(D_field,-1,axis=0))
    D_ym = 0.5*(D_field + np.roll(D_field, 1,axis=0))
    return ((D_xp*(np.roll(field,-1,axis=1)-field)
             - D_xm*(field-np.roll(field, 1,axis=1))
             + D_yp*(np.roll(field,-1,axis=0)-field)
             - D_ym*(field-np.roll(field, 1,axis=0)))
            .astype(np.float32))

def softsign(x, cap):
    return x / (1.0 + np.abs(x) / cap)

def gini(arr):
    if len(arr) == 0: return 0.0
    arr = np.sort(arr.astype(float))
    n   = len(arr)
    idx = np.arange(1, n+1)
    return (2*(idx*arr).sum()) / (n*arr.sum()+1e-9) - (n+1)/n

def advection_upwind(P, Vx, Vy):
    """一阶迎风差分平流：-∇·(P·V)，周期边界，防止负扩散"""
    # x 方向
    dPdx_fwd = np.roll(P,-1,axis=1) - P
    dPdx_bwd = P - np.roll(P, 1,axis=1)
    adv_x = (np.maximum(Vx,0)*dPdx_bwd + np.minimum(Vx,0)*dPdx_fwd)
    # y 方向
    dPdy_fwd = np.roll(P,-1,axis=0) - P
    dPdy_bwd = P - np.roll(P, 1,axis=0)
    adv_y = (np.maximum(Vy,0)*dPdy_bwd + np.minimum(Vy,0)*dPdy_fwd)
    # 散度 ∇·(P·V) ≈ P*(∂Vx/∂x+∂Vy/∂y) + V·∇P → 用迎风近似
    div_P_V = P*(np.roll(Vx,-1,axis=1)-np.roll(Vx,1,axis=1)
                 + np.roll(Vy,-1,axis=0)-np.roll(Vy,1,axis=0))*0.5
    return -(adv_x + adv_y + div_P_V).astype(np.float32)


# ─────────────────────────────────────────────
# 三、60甲子节律引擎
# ─────────────────────────────────────────────

class JiaziClock:
    """天干地支双环振荡器——地球气候节律引擎

    天干（周期10）：局部温度/光照涨落（阳气升降）
    地支（周期12）：全球水循环/洋流深层节律（阴气流转）
    60甲子共振：两者叠加产生周期60的气候波
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        H, W = cfg.H, cfg.W
        # 空间掩膜：模拟大陆/海洋的阳光接收差异
        rng = np.random.default_rng(7)
        base = rng.uniform(0.6, 1.4, (H, W)).astype(np.float32)
        # 用高斯模糊平滑，形成地理性斑块
        self._spatial_mask = convolve(base, make_kernel(15,'gauss'), mode='wrap')
        self._spatial_mask /= self._spatial_mask.mean() + 1e-9

    def P_in(self, tick: int) -> np.ndarray:
        """计算当前帧的太阳泵输入场"""
        cfg = self.cfg
        if cfg.P_in_blocked:
            return np.zeros((cfg.H, cfg.W), np.float32)
        t = tick * cfg.dt
        # 60甲子节律调制（归一化到 [0, 1+amplitude]）
        phase = (np.sin(2*np.pi*t / cfg.tian_gan_period)
               + np.sin(2*np.pi*t / cfg.di_zhi_period))
        rhythm = 1.0 + cfg.jiazi_amplitude * phase / 2.0   # 归一化
        rhythm = max(rhythm, 0.0)
        # 空间异质性
        field = cfg.A_solar * rhythm * (
            (1-cfg.P_in_spatial_var)
            + cfg.P_in_spatial_var * self._spatial_mask
        )
        return field.astype(np.float32)

    def hexagram_index(self, tick: int) -> int:
        """计算当前帧对应的64卦索引（0-63）"""
        t = tick * self.cfg.dt
        tg = int(t / self.cfg.tian_gan_period) % 10   # 天干 0-9
        dz = int(t / self.cfg.di_zhi_period)  % 12   # 地支 0-11
        # 映射到 0-63：用天干(3bit)+地支(3bit) → 6bit
        tg3 = tg % 8
        dz3 = dz % 8
        return (tg3 << 3) | dz3


# ─────────────────────────────────────────────
# 四、自适应调参模块（继承 v12，语义重标注）
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
        self._last_phase  = None
        self._buf_len     = 100

    def update(self, tick, phase, f_C2, f_C1, phi_net, c_var, delta_sig):
        cfg = self.cfg
        self._phi_net_buf.append(phi_net)
        self._c_var_buf.append(c_var)
        if len(self._phi_net_buf) > self._buf_len:
            self._phi_net_buf.pop(0)
            self._c_var_buf.pop(0)
        self._chaotic_cnt = self._chaotic_cnt+1 if phase=='CHAOTIC_EDGE' else 0
        self._last_phase  = phase
        interval = 20 if phase == 'TURING_GROWTH' else cfg.adapt_interval
        if tick % interval != 0 or tick == 0:
            return
        self._check_rules(tick, phase, f_C2, f_C1, phi_net, c_var, delta_sig)

    def _check_rules(self, tick, phase, f_C2, f_C1, phi_net, c_var, delta_sig):
        cfg = self.cfg
        if (self._chaotic_cnt >= 300 and phi_net == 0.0 and f_C2 < 0.20):
            new = max(self.BOUNDS['lam_rate_C1'][0], cfg.lam_rate_C1*0.80)
            if new < cfg.lam_rate_C1 - 1e-6:
                self._apply('lam_rate_C1', cfg.lam_rate_C1, new, tick,
                            f'CHAOTIC_EDGE锁死{self._chaotic_cnt}帧，降低lam_rate_C1')
                self._chaotic_cnt = 0
        if (phase in ('UNIFORM_GROWTH','TURING_GROWTH') and f_C1 > 0.75
                and f_C2 < 0.05 and tick > 100):
            self._apply('Theta_barrier', cfg.Theta_barrier,
                        max(self.BOUNDS['Theta_barrier'][0], cfg.Theta_barrier*0.85),
                        tick, f'C1过度积累(f_C1={f_C1:.3f})，降低势垒')

    def _apply(self, param, old, new, tick, reason):
        if abs(new-old) < 1e-9: return
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
# 五、物理层（v13 生态引擎）
# ─────────────────────────────────────────────

class PhysicsLayer:
    def __init__(self, cfg: Config, clock: 'JiaziClock'):
        self.cfg   = cfg
        self.clock = clock
        shape = (cfg.H, cfg.W)
        rng   = np.random.default_rng(42)

        # ── 主场 ──────────────────────────────────────────────────────
        self.P  = rng.uniform(0.2, 1.0, shape).astype(np.float32)
        self.C1 = np.zeros(shape, np.float32)
        self.C2 = np.zeros(shape, np.float32)

        # ── [v13 新增] 记忆场 M* 与多样性场 I* ───────────────────────
        self.M_star = np.zeros(shape, np.float32)   # 化石地层/基因库
        self.I_star = np.zeros(shape, np.float32)   # 生物多样性/食物网

        # ── [v13 新增] 漂移速度场（可视化用）────────────────────────
        self.Vx = np.zeros(shape, np.float32)
        self.Vy = np.zeros(shape, np.float32)

        # ── [v13 新增] 当前帧太阳泵输入 ──────────────────────────────
        self.P_in_field = np.zeros(shape, np.float32)

        # ── [v13 新增] 灾变状态 ───────────────────────────────────────
        self.catastrophe_active    = False
        self._catastrophe_cooldown = 0      # 灾变后压制倒计时

        # ── v12 继承：广义算子存储 ────────────────────────────────────
        self.stock_rate_P   = np.zeros(shape, np.float32)
        self.stock_rate_C1  = np.zeros(shape, np.float32)
        self.stock_rate_C2  = np.zeros(shape, np.float32)
        self.stock_state_C1 = np.zeros(shape, np.float32)
        self.stock_state_C2 = np.zeros(shape, np.float32)
        self._I_macro_prev  = np.zeros(shape, np.float32)
        self.I_drive        = np.zeros(shape, np.float32)
        self._I_norm_vis    = np.zeros(shape, np.float32)
        self.metric_vis     = np.ones(shape, np.float32)
        self.rate_stock_mean = 0.0
        self.I_stock        = np.zeros(shape, np.float32)
        self.I_increment    = np.zeros(shape, np.float32)

        self.P_prev  = self.P.copy()
        self.C1_prev = self.C1.copy()
        self.C2_prev = self.C2.copy()

        self.blur_k  = make_kernel(cfg.blur_size, 'blur')
        self.gauss_k = make_kernel(cfg.gauss_size,'gauss')

        # v12 继承
        self.A_C1_to_C2 = np.zeros(shape, np.float32)
        self.A_C2_to_C1 = np.zeros(shape, np.float32)
        self.A_tartini  = np.zeros(shape, np.float32)
        self.tartini_total = 0.0

        # [v13.2] 废热辐射记录
        self.rad_total = 0.0

        # [v13.3] 地热场（固定空间，模拟板块构造）
        self.geo_field = self._make_geo_field()
        self.geo_flux  = 0.0

        # 实验初始化
        self._apply_experiment_init()

    def _make_geo_field(self) -> np.ndarray:
        """[v13.3] 生成固定地热热点空间图

        模拟板块边界/火山岛链：高斯形热点叠加均匀基底
        固定随机种子确保每次运行地热分布相同（地质构造不变）
        """
        cfg   = self.cfg
        shape = (cfg.H, cfg.W)
        rng   = np.random.default_rng(2024)  # 固定种子=地质构造固定
        field = np.zeros(shape, np.float32)
        for _ in range(cfg.geo_hotspot_count):
            cx = rng.integers(8, cfg.W - 8)
            cy = rng.integers(8, cfg.H - 8)
            yy, xx = np.ogrid[:cfg.H, :cfg.W]
            dist2  = (xx - cx)**2 + (yy - cy)**2
            sigma2 = (cfg.geo_hotspot_radius) ** 2
            field += np.exp(-dist2 / (2 * sigma2)).astype(np.float32)
        # 归一化到 [0, amplitude] 并叠加均匀基底（全球地温底噪）
        peak = field.max() + 1e-9
        field = field / peak * cfg.geo_heat_amplitude
        field += cfg.geo_heat_amplitude * 0.15   # 全球均匀基底
        return field.astype(np.float32)

    def _apply_experiment_init(self):
        cfg = self.cfg
        shape = (cfg.H, cfg.W)
        rng   = np.random.default_rng(99)
        exp   = cfg.experiment

        if exp == 'abiogenesis':
            # 原始汤：高浓度 P，无生命
            self.P  = rng.uniform(1.5, 3.0, shape).astype(np.float32)
            self.C1 = np.zeros(shape, np.float32)
            self.C2 = np.zeros(shape, np.float32)

        elif exp == 'biome':
            # 生物圈划分：有一定 C1 基数，开启竞争
            self.P  = rng.uniform(0.3, 0.8, shape).astype(np.float32)
            self.C1 = rng.uniform(0.0, 0.1, shape).astype(np.float32)
            self.C2 = np.zeros(shape, np.float32)

        elif exp == 'extinction':
            # 灭绝实验：大型 C2 霸主生态，稍后手动触发灾变
            self.P  = rng.uniform(0.1, 0.3, shape).astype(np.float32)
            self.C1 = rng.uniform(0.0, 0.05, shape).astype(np.float32)
            # C2 随机大斑块
            c2 = np.zeros(shape, np.float32)
            for _ in range(8):
                cx = rng.integers(0, cfg.W)
                cy = rng.integers(0, cfg.H)
                r  = rng.integers(10, 25)
                yy, xx = np.ogrid[:cfg.H, :cfg.W]
                mask = ((xx-cx)**2 + (yy-cy)**2) < r**2
                c2[mask] = rng.uniform(0.8, 1.5)
            self.C2 = c2.astype(np.float32)
            self.M_star = (c2 * 0.3).astype(np.float32)

        # 'free' → 使用 __init__ 的默认初始化

    def step(self, tick: int, c_var: float = 0.0, phase: str = 'CHAOTIC_EDGE'):
        cfg  = self.cfg
        P, C1, C2 = self.P, self.C1, self.C2
        eps  = 1e-9

        # ══════════════════════════════════════════════════════
        # [v13 Step-A] M* 记忆场更新（化石地层沉积）
        # M* += rate * C2;  M* *= decay（地质侵蚀）
        # ══════════════════════════════════════════════════════
        self.M_star = (self.M_star * cfg.Mstar_decay_rate
                       + cfg.Mstar_accum_rate * C2 * cfg.dt).astype(np.float32)
        np.clip(self.M_star, 0.0, 50.0, out=self.M_star)  # [P1修复: 上限5→50，恢复生态位记忆梯度]

        # ══════════════════════════════════════════════════════
        # [v13 Step-B] I* 多样性场（Shannon 空间多样性代理）
        # 用 C1+C2 的局部标准差作为食物网复杂度代理
        # ══════════════════════════════════════════════════════
        bio_total = C1 + C2
        bio_mean  = uniform_filter(bio_total, size=5, mode='wrap')
        bio_m2    = uniform_filter(bio_total**2, size=5, mode='wrap')
        bio_var   = np.maximum(bio_m2 - bio_mean**2, 0.0)
        self.I_star = np.sqrt(bio_var).astype(np.float32)

        # ══════════════════════════════════════════════════════
        # [v13 Step-C] Fokker-Planck 漂移场 V_macro
        # V = γ · ∇(M* + α·I*)
        # ══════════════════════════════════════════════════════
        potential = (cfg.alpha_Mstar * self.M_star
                     + cfg.alpha_Istar * self.I_star).astype(np.float32)
        # 梯度（中心差分，周期边界）
        grad_x = (np.roll(potential,-1,axis=1) - np.roll(potential,1,axis=1)) * 0.5
        grad_y = (np.roll(potential,-1,axis=0) - np.roll(potential,1,axis=0)) * 0.5
        self.Vx = (cfg.gamma_drift * grad_x).astype(np.float32)
        self.Vy = (cfg.gamma_drift * grad_y).astype(np.float32)

        # ══════════════════════════════════════════════════════
        # [v13 Step-D] 60甲子太阳泵 P_in
        # ══════════════════════════════════════════════════════
        if self._catastrophe_cooldown > 0:
            # 灾变后核冬天：P_in 压制
            suppression = self._catastrophe_cooldown / cfg.catastrophe_duration
            self.P_in_field = self.clock.P_in(tick) * (1.0 - 0.9*suppression)
            self._catastrophe_cooldown -= 1
        else:
            self.P_in_field = self.clock.P_in(tick)

        # ══════════════════════════════════════════════════════
        # [v13 Step-E] D_macro 灾变（泊松脉冲）
        # ══════════════════════════════════════════════════════
        self.catastrophe_active = False
        if (not self.catastrophe_active
                and np.random.random() < cfg.p_catastrophe
                and tick > 500):          # 系统稳定后才触发
            self.catastrophe_active = True
            self._catastrophe_cooldown = cfg.catastrophe_duration
            # 拉平 C1、C2（小行星撞击/冰川）
            self.C1 = (C1 * cfg.catastrophe_c1_suppress
                       + np.random.exponential(0.02, C1.shape)
                       ).astype(np.float32)
            self.C2 = (C2 * cfg.catastrophe_c2_suppress
                       ).astype(np.float32)
            C1, C2 = self.C1, self.C2   # 更新局部引用
            print(f'  [D_MACRO 灾变] tick={tick:06d}  C1↓{cfg.catastrophe_c1_suppress}  C2↓{cfg.catastrophe_c2_suppress}  核冬天持续{cfg.catastrophe_duration}帧')

        # ══════════════════════════════════════════════════════
        # 步骤0：情境遗忘率场（继承 v12）
        # ══════════════════════════════════════════════════════
        gx = np.gradient(self._I_macro_prev, axis=1).astype(np.float32)
        gy = np.gradient(self._I_macro_prev, axis=0).astype(np.float32)
        grad_mag   = np.sqrt(gx**2 + gy**2)
        ctx_factor = np.exp(-1.5 * grad_mag /
                            (self._I_macro_prev + 1e-6)).astype(np.float32)

        def lam_field(lam_base):
            return np.maximum(lam_base*ctx_factor, lam_base*0.1)

        lam_rP  = lam_field(cfg.lam_rate_P)
        lam_rC1 = lam_field(cfg.lam_rate_C1)
        lam_rC2 = lam_field(cfg.lam_rate_C2)
        lam_sC1 = lam_field(cfg.lam_state_C1)
        lam_sC2 = lam_field(cfg.lam_state_C2)

        # 步骤1：特征张量
        chi_rate_P   = np.abs(P  - self.P_prev)
        chi_rate_C1  = np.abs(C1 - self.C1_prev)
        chi_rate_C2  = np.abs(C2 - self.C2_prev)
        chi_state_C1 = C1
        chi_state_C2 = C2

        # 步骤2：H_sink 极化检测
        C2_lmean = uniform_filter(C2, size=5, mode='wrap')
        C2_lm2   = uniform_filter(C2**2, size=5, mode='wrap')
        C2_lvar  = np.maximum(C2_lm2 - C2_lmean**2, 0.0)
        raw_mask = (C2_lvar > 0.03).astype(np.float32)
        sink_mask = np.clip(convolve(raw_mask, make_kernel(3,'gauss'), mode='wrap'), 0.0, 1.0)
        chi_rate_C2_eff = chi_rate_C2 * (1.0 - sink_mask)
        chi_rate_C1_eff = chi_rate_C1 * (1.0 - sink_mask*0.5)
        A_C2_sink = 0.02 * sink_mask * C2

        # 步骤3：L 有损算子（I_macro 积累）
        I_macro          = np.zeros_like(P)
        I_macro_rate_sum = np.zeros_like(P)
        features = [
            (chi_rate_P,     'stock_rate_P',   cfg.kw_rate_P,   lam_rP,  cfg.w_rate_P,   True),
            (chi_rate_C1_eff,'stock_rate_C1',  cfg.kw_rate_C1,  lam_rC1, cfg.w_rate_C1,  True),
            (chi_rate_C2_eff,'stock_rate_C2',  cfg.kw_rate_C2,  lam_rC2, cfg.w_rate_C2,  True),
            (chi_state_C1,   'stock_state_C1', cfg.kw_state_C1, lam_sC1, cfg.w_state_C1, False),
            (chi_state_C2,   'stock_state_C2', cfg.kw_state_C2, lam_sC2, cfg.w_state_C2, False),
        ]
        for chi, sname, k_w, lam_arr, weight, is_rate in features:
            stock = getattr(self, sname)
            p_w   = chi / (chi + k_w)
            mask  = (np.random.rand(*P.shape) < p_w).astype(np.float32)
            stock[:] = stock*(1.0 - lam_arr) + chi*mask*cfg.dt
            ns = stock / (stock + cfg.k_norm_base)
            I_macro += weight * ns
            if is_rate: I_macro_rate_sum += weight * ns

        self._I_macro_prev  = I_macro.copy()
        self.I_stock        = (self.stock_rate_P + self.stock_rate_C1 + self.stock_rate_C2
                               + self.stock_state_C1 + self.stock_state_C2)
        self.I_increment    = (chi_rate_P*cfg.w_rate_P + chi_rate_C1*cfg.w_rate_C1
                               + chi_rate_C2*cfg.w_rate_C2)
        self.rate_stock_mean = float((self.stock_rate_P+self.stock_rate_C1+self.stock_rate_C2).mean())

        # 步骤4：D_micro 失真算子（乘性噪声 = 基因突变）
        I_context   = convolve(I_macro, self.blur_k, mode='wrap')
        S_norm_C1   = self.stock_state_C1 / (self.stock_state_C1 + cfg.k_norm_base)
        S_norm_C2   = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        S_macro     = cfg.w_state_C1*S_norm_C1 + cfg.w_state_C2*S_norm_C2
        noise_micro = np.random.randn(*P.shape).astype(np.float32) * cfg.k_noise_micro * I_context
        I_distorted = I_context + 0.3*lap(S_macro) + noise_micro

        # 步骤5：柔性极化与死区清洗
        Omega_raw    = softsign(I_distorted, 1.0)
        Omega_damped = (1.0-0.15) * convolve(Omega_raw, self.gauss_k, mode='wrap')
        I_drive      = np.where(np.abs(Omega_damped) < 1e-4, 0.0, Omega_damped)
        self.I_drive     = I_drive
        self._I_norm_vis = I_macro / (I_macro.max() + eps)

        # 步骤6：涌现度规
        S_C2_norm = self.stock_state_C2 / (self.stock_state_C2 + cfg.k_norm_base)
        S_C1_norm = self.stock_state_C1 / (self.stock_state_C1 + cfg.k_norm_base)
        metric_factor = np.maximum(0.1,
            1.0 - cfg.alpha_metric*S_C2_norm - cfg.alpha_metric_C1*S_C1_norm)
        D_C1_eff = cfg.D_C1 * metric_factor
        D_C2_eff = cfg.D_C2 * metric_factor
        self.metric_vis = metric_factor

        # 步骤7：跃迁通量
        A_P_to_C1  = np.maximum(0,  I_drive) * cfg.Base_Rate_1 * P
        A_C1_to_P  = np.abs(np.minimum(0, I_drive)) * cfg.Base_Rate_1 * C1
        A_C1_to_C2 = np.maximum(0, I_drive - cfg.Theta_barrier) * cfg.Base_Rate_2 * C1
        A_C2_to_C1 = np.abs(np.minimum(0, I_drive)) * cfg.Base_Rate_dec * C2
        self.A_C1_to_C2 = A_C1_to_C2
        self.A_C2_to_C1 = A_C2_to_C1

        # [v13] L 算子：生态容量竞争死亡率
        # Decay_C1 = λ·C1·exp(C1/(P+ε)-1)
        carrying_ratio = C1 / (P + cfg.eps_carrying)
        eco_decay_C1   = cfg.lambda_eco * C1 * np.exp(
            np.clip(carrying_ratio - 1.0, -3.0, 3.0))
        eco_decay_C1   = np.clip(eco_decay_C1, 0.0, C1)

        decay_C2_dyn = cfg.decay_C2 + cfg.k_entropy * float(I_macro_rate_sum.mean())
        A_C2_decay   = decay_C2_dyn * C2

        # [v12 继承] 塔蒂尼自催化项（自催化反应 = 原始汤化学循环）
        if cfg.beta_tartini > 0:
            A_tartini = np.maximum(cfg.beta_tartini * P * P, 0.0)
        else:
            A_tartini = np.zeros_like(P)
        self.A_tartini    = A_tartini
        self.tartini_total = float(A_tartini.sum())

        self.P_prev  = P.copy()
        self.C1_prev = C1.copy()
        self.C2_prev = C2.copy()

        # ══════════════════════════════════════════════════════
        # [v13.3 Step-G] 地热能输入（恒定，永不中断）
        # 物理意义：岩浆散热独立于太阳，灾变核冬天时维持热液生态系统
        # 70% → P（热液矿化/地温驱动无机循环）
        # 30% → C1（化能合成，绕过光合作用直接驱动生命）
        # ══════════════════════════════════════════════════════
        geo_to_P  = self.geo_field * (1.0 - cfg.geo_to_C1_frac)
        geo_to_C1 = self.geo_field * cfg.geo_to_C1_frac
        self.geo_flux = float(self.geo_field.sum()) * cfg.dt

        # ══════════════════════════════════════════════════════
        # [v13.2 Step-R] 废热辐射出口
        # σ_P > σ_C1 > σ_C2：熵越低的态，辐射越慢（热力学驱动进化）
        # ══════════════════════════════════════════════════════
        rad_P  = cfg.sigma_rad_P  * P
        rad_C1 = cfg.sigma_rad_C1 * C1
        rad_C2 = cfg.sigma_rad_C2 * C2
        self.rad_total = float(rad_P.sum() + rad_C1.sum() + rad_C2.sum()) * cfg.dt

        # 步骤8：PDE 更新
        # P：扩散 + FP漂移 + 太阳泵 + 地热(70%) - 消耗 - 自催化 - 废热辐射
        dP_dt = (cfg.D_P  * lap(P)
                 + advection_upwind(P, self.Vx, self.Vy)   # FP 漂移
                 + self.P_in_field                          # 太阳泵输入
                 + geo_to_P                                 # [v13.3] 地热→P（热液矿化）
                 - A_P_to_C1 + A_C1_to_P
                 + eco_decay_C1
                 + A_C2_sink
                 - A_tartini
                 - rad_P)

        dC1_dt = (div_diff(C1, D_C1_eff)
                  + A_P_to_C1 - A_C1_to_P
                  - A_C1_to_C2 + A_C2_to_C1
                  - eco_decay_C1
                  + A_C2_decay
                  + A_tartini
                  + geo_to_C1                               # [v13.3] 地热→C1（化能合成）
                  - rad_C1)

        dC2_dt = (div_diff(C2, D_C2_eff)
                  + A_C1_to_C2 - A_C2_to_C1
                  - A_C2_decay - A_C2_sink
                  - rad_C2)

        self.P  = np.maximum(P  + dP_dt  * cfg.dt, 0.0)
        self.C1 = np.maximum(C1 + dC1_dt * cfg.dt, 0.0)
        self.C2 = np.maximum(C2 + dC2_dt * cfg.dt, 0.0)

        # [v13] 开放系统不做全局守恒修正；仅防 NaN/Inf
        for arr in (self.P, self.C1, self.C2):
            np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=0.0, copy=False)
            np.clip(arr, 0.0, 50.0, out=arr)


# ─────────────────────────────────────────────
# 六、观测层
# ─────────────────────────────────────────────

class ObservationLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = {k: [] for k in [
            'tick','M_total','f_C2','f_C1','f_P',
            'delta_sigma','Phi_net','C_var','H_C2',
            'n_clusters','size_gini','phase','Theta_barrier',
            'solar_flux','M_star_mean','I_star_mean',
            'catastrophe','hexagram_idx','rad_total','energy_balance','geo_flux',
        ]}
        self._hd_buf = []

    def observe(self, phys: PhysicsLayer, tick: int, clock: JiaziClock):
        cfg = self.cfg
        P, C1, C2 = phys.P, phys.C1, phys.C2
        eps = 1e-9
        M_P  = float(P.sum())
        M_C1 = float(C1.sum())
        M_C2 = float(C2.sum())
        M_total = M_P + M_C1 + M_C2
        f_C2 = M_C2 / (M_total + eps)
        f_C1 = M_C1 / (M_total + eps)
        f_P  = M_P  / (M_total + eps)
        sig_inc  = float(phys.I_increment.sum())
        sig_loss = cfg.lam_rate_C2 * phys.rate_stock_mean * P.size
        delta_sig = sig_inc - sig_loss
        phi_C2 = C2 / (P + C1 + C2 + eps)
        H_C2   = float(-(phi_C2 * np.log(phi_C2 + eps)).sum())
        C_var  = float(phi_C2.var())
        Phi_net = float(phys.A_C1_to_C2.sum() - phys.A_C2_to_C1.sum())
        phase   = self._classify(delta_sig, Phi_net, C_var, f_C2)
        n_cl, s_gini = 0, 0.0
        if tick % cfg.struct_every == 0:
            n_cl, s_gini = self._struct_stats(phi_C2)
        self._hd_buf.append((float(phys.I_increment.max()), float(abs(phys.I_drive).max())))
        if len(self._hd_buf) > cfg.hd_window:
            self._hd_buf.pop(0)
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
        h['solar_flux'].append(float(phys.P_in_field.sum()))
        h['M_star_mean'].append(float(phys.M_star.mean()))
        h['I_star_mean'].append(float(phys.I_star.mean()))
        h['catastrophe'].append(int(phys.catastrophe_active))
        h['hexagram_idx'].append(clock.hexagram_index(tick))
        h['rad_total'].append(phys.rad_total)
        h['geo_flux'].append(phys.geo_flux)
        solar_in = float(phys.P_in_field.sum()) * cfg.dt
        # 总输入=太阳+地热，总输出=废热辐射
        h['energy_balance'].append(solar_in + phys.geo_flux - phys.rad_total)
        return phase, C_var

    @property
    def is_dead(self):
        if len(self._hd_buf) < self.cfg.hd_window:
            return False
        return (max(v[0] for v in self._hd_buf) < 1e-4
                and max(v[1] for v in self._hd_buf) < self.cfg.hd_drive_eps)

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
        threshold = (phi_C2.mean() + 0.5*phi_C2.std()
                     if self.cfg.phi_threshold is None else self.cfg.phi_threshold)
        binary = (phi_C2 > threshold).astype(int)
        labeled, n = label(binary)
        if n == 0:
            return 0, 0.0
        sizes = np.array([(labeled == i).sum() for i in range(1, n+1)])
        return n, gini(sizes)


# ─────────────────────────────────────────────
# 七、可视化层
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

_HEXAGRAM_NAMES = [
    '坤','剥','比','观','豫','晋','萃','否',
    '谦','艮','蹇','渐','小过','旅','咸','遁',
    '师','蒙','坎','涣','解','未济','困','讼',
    '升','蛊','井','巽','恒','鼎','大过','姤',
    '复','颐','屯','益','震','噬嗑','随','无妄',
    '明夷','贲','既济','家人','丰','离','革','同人',
    '临','损','节','中孚','归妹','睽','兑','履',
    '泰','大畜','需','小畜','大壮','大有','夬','乾',
]


def _draw_hexagram_panel(ax, hexagram_idx: int, history_idx: list):
    """绘制 8x8=64卦引力盆地宫格"""
    from collections import Counter
    ax.set_facecolor('#0d0d0d')
    grid = np.zeros((8, 8))
    recent = history_idx[-500:] if len(history_idx) > 500 else history_idx
    cnt = Counter(recent)
    for idx, freq in cnt.items():
        row, col = divmod(idx, 8)
        if row < 8 and col < 8:
            grid[row, col] = freq
    grid = grid / (grid.max() + 1e-9)
    ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    cur_row, cur_col = divmod(min(hexagram_idx, 63), 8)
    rect = plt.Rectangle((cur_col-0.5, cur_row-0.5), 1, 1,
                          linewidth=2, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.tick_params(colors='white', labelsize=5)
    name = _HEXAGRAM_NAMES[min(hexagram_idx, 63)]
    ax.set_title(f'64卦引力盆地  当前:{name}({hexagram_idx})',
                 color='#aaaaaa', fontsize=7, pad=3)


def render(phys: PhysicsLayer, obs: ObservationLayer,
           clock: JiaziClock, tick: int):
    cfg = phys.cfg
    P, C1, C2 = phys.P, phys.C1, phys.C2
    eps = 1e-9
    phi_C2 = C2 / (P + C1 + C2 + eps)

    fig = plt.figure(figsize=(20, 10), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(2, 5, figure=fig,
                            hspace=0.38, wspace=0.30,
                            left=0.04, right=0.97,
                            top=0.91, bottom=0.07)

    V_mag = np.sqrt(phys.Vx**2 + phys.Vy**2)
    fields_top = [
        (P,           'Blues',   'P 场  无机能量/游离资源\n(阳光·水汽·无机盐)'),
        (C1,          'Greens',  'C1 场  生物个体/瞬态种群\n(植物·游荡兽群)'),
        (phi_C2,      'YlGn',    'φ_C2  顶极群落密度\n(雨林·珊瑚礁·地质层)'),
        (phys.M_star, 'bone_r',  'M* 化石地层/基因库\n(历史生命痕迹)'),
        (V_mag,       'plasma',  'V_macro 生态位势场\n(漂移引力盆地强度)'),
    ]
    for col, (data, cmap, title) in enumerate(fields_top):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data, cmap=cmap, interpolation='nearest', aspect='auto')
        ax.set_title(title, color='white', fontsize=6.5, pad=3)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(
            labelcolor='white', labelsize=5)

    h  = obs.history
    tk = h['tick']
    _n = len(tk)
    if _n > 1000:
        _s   = _n // 1000
        _idx = list(range(0, _n, _s)) + [_n-1]
        tp = [tk[i] for i in _idx]
        hp = {k: [h[k][i] for i in _idx] for k in
              ['f_C2','f_C1','f_P','C_var','Theta_barrier',
               'M_star_mean','I_star_mean','solar_flux','phase',
               'catastrophe','rad_total','energy_balance','geo_flux']}
    else:
        tp = tk; hp = h

    ax_ts = fig.add_subplot(gs[1, :3])
    ax_ts.set_facecolor('#1a1a1a')
    for key, color, lbl in [
        ('f_P',        '#3498db', 'f_P  无机能量'),
        ('f_C1',       '#27ae60', 'f_C1  生物个体'),
        ('f_C2',       '#f39c12', 'f_C2  顶极群落'),
        ('C_var',      '#e74c3c', 'C_var  空间分化'),
        ('M_star_mean','#8e44ad', 'M*  地层积累'),
        ('I_star_mean','#1abc9c', 'I*  生物多样性'),
        ('Theta_barrier','#95a5a6','Θ 势垒'),
    ]:
        ax_ts.plot(tp, hp[key], color=color, lw=0.9, label=lbl)

    if len(tp) > 1:
        seg_start, seg_phase = tp[0], hp['phase'][0]
        for i in range(1, len(tp)):
            if hp['phase'][i] != seg_phase:
                ax_ts.axvspan(seg_start, tp[i], alpha=0.08,
                              color=PHASE_COLORS.get(seg_phase, '#fff'))
                seg_start, seg_phase = tp[i], hp['phase'][i]
        ax_ts.axvspan(seg_start, tp[-1], alpha=0.08,
                      color=PHASE_COLORS.get(seg_phase, '#fff'))
    for i, cat in enumerate(hp['catastrophe']):
        if cat:
            ax_ts.axvline(tp[i], color='red', lw=1.5, alpha=0.7, linestyle='--')

    ax_ts.axhline(0, color='white', lw=0.4, alpha=0.4)
    ax_ts.legend(loc='upper left', fontsize=5.5, facecolor='#1a1a1a',
                 labelcolor='white', framealpha=0.6, ncol=2)
    ax_ts.tick_params(colors='white', labelsize=7)
    ax_ts.set_xlabel('帧数 (tick)', color='white', fontsize=8)
    ax_ts.set_title('地球生态演化时序  (背景=演化阶段  红虚=灾变)',
                    color='#aaaaaa', fontsize=8, pad=4)
    for sp in ax_ts.spines.values():
        sp.set_edgecolor('#444')

    ax_sol = fig.add_subplot(gs[1, 3])
    ax_sol.set_facecolor('#1a1a1a')
    ax_sol.plot(tp, hp['solar_flux'], color='#f1c40f', lw=1.0, label='太阳泵')
    # 地热曲线：橙色，恒定
    geo_vals = hp['geo_flux']
    ax_sol.plot(tp, geo_vals, color='#e67e22', lw=1.2, linestyle='--', label='地热（恒定）')
    ax_sol.plot(tp, hp['rad_total'],  color='#e74c3c', lw=1.0, label='废热辐射')
    ax_sol.axhline(0, color='white', lw=0.4, alpha=0.4)
    solar_plus_geo = [s + g for s, g in zip(hp['solar_flux'], geo_vals)]
    ax_sol.fill_between(tp,
        [sg - r for sg, r in zip(solar_plus_geo, hp['rad_total'])],
        0, alpha=0.15, color='#2ecc71', label='能量收支')
    ax_sol.legend(fontsize=5.5, facecolor='#1a1a1a', labelcolor='white', framealpha=0.6)
    ax_sol.tick_params(colors='white', labelsize=7)
    ax_sol.set_title('太阳泵+地热 ― 废热辐射平衡\n(黄=太阳 橙=地热 红=辐射 綠=收支)',
                     color='#aaaaaa', fontsize=7, pad=4)
    for sp in ax_sol.spines.values():
        sp.set_edgecolor('#444')

    ax_hex = fig.add_subplot(gs[1, 4])
    hex_idx = h['hexagram_idx'][-1] if h['hexagram_idx'] else 0
    _draw_hexagram_panel(ax_hex, hex_idx, h['hexagram_idx'])

    phase_now  = h['phase'][-1]  if h['phase']  else '—'
    mstar_now  = h['M_star_mean'][-1] if h['M_star_mean'] else 0.0
    rad_now    = h['rad_total'][-1] if h['rad_total'] else 0.0
    geo_now    = h['geo_flux'][-1] if h['geo_flux'] else 0.0
    bal_now    = h['energy_balance'][-1] if h['energy_balance'] else 0.0
    color_now  = PHASE_COLORS.get(phase_now, 'white')
    bal_sign   = '+' if bal_now >= 0 else ''
    fig.suptitle(
        f'涌积态地球生态引擎 v13.3 [{cfg.experiment.upper()}]   第 {tick:05d} 帧   '
        f'f_P={h["f_P"][-1]:.3f}  f_C1={h["f_C1"][-1]:.3f}  f_C2={h["f_C2"][-1]:.3f}  '
        f'M*={mstar_now:.3f}  地热={geo_now:.2f}  废热={rad_now:.1f}  收支={bal_sign}{bal_now:.2f}',
        color=color_now, fontsize=9.5, fontweight='bold'
    )
    plt.savefig(OUTPUT_DIR / f'frame_{tick:05d}.png',
                dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  [render] tick={tick:05d}  {phase_now:<18s}  '
          f'f_P={h["f_P"][-1]:.3f}  f_C1={h["f_C1"][-1]:.3f}  '
          f'f_C2={h["f_C2"][-1]:.3f}  M*={mstar_now:.4f}  '
          f'地热={geo_now:.2f}  废热={rad_now:.2f}  收支={bal_sign}{bal_now:.3f}')


# ─────────────────────────────────────────────
# 八、主循环
# ─────────────────────────────────────────────

def run():
    import sys
    cfg = Config()
    # 支持命令行指定实验预设： python universe_v13_3.py [extinction|abiogenesis|biome|free]
    _valid = {'extinction', 'abiogenesis', 'biome', 'free'}
    if len(sys.argv) > 1 and sys.argv[1] in _valid:
        cfg.experiment = sys.argv[1]
        print(f'  [命令行指定] 实验预设切换为: {cfg.experiment.upper()}')
    clock = JiaziClock(cfg)
    phys  = PhysicsLayer(cfg, clock)
    obs   = ObservationLayer(cfg)
    tuner = AdaptiveTuner(cfg) if cfg.adaptive_tuning else None

    print('=' * 72)
    print('涌积态宇宙模型 v13.3 — 地球自然演化与生态相变引擎')
    print(f'  实验预设  : {cfg.experiment.upper()}')
    print(f'  网格      : {cfg.W}x{cfg.H}   dt={cfg.dt}   步数={cfg.steps}')
    print(f'  太阳泵    : A_solar={cfg.A_solar}  甲子: 天干{cfg.tian_gan_period}/地支{cfg.di_zhi_period}')
    print(f'  地热能    : geo_amp={cfg.geo_heat_amplitude}  热点数={cfg.geo_hotspot_count}  化能比={cfg.geo_to_C1_frac}')
    print(f'  漂移强度  : γ={cfg.gamma_drift}  灾变概率: p={cfg.p_catastrophe:.2e}/帧')
    print(f'  变量语义  : P=无机能量  C1=生物个体  C2=顶极群落  M*=化石地层')
    print(f'  输出目录  : {OUTPUT_DIR}')
    print('=' * 72)

    c_var_current = 0.0
    phase_current = 'CHAOTIC_EDGE'

    for tick in range(cfg.steps):
        phys.step(tick, c_var_current, phase_current)
        phase, c_var_current = obs.observe(phys, tick, clock)
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
            render(phys, obs, clock, tick)

        if tick % 500 == 0 and tick > 0:
            h = obs.history
            pct = tick / cfg.steps * 100
            hi  = h['hexagram_idx'][-1]
            name = _HEXAGRAM_NAMES[min(hi, 63)]
            print(f'  [{pct:5.1f}%] tick={tick:06d}  {h["phase"][-1]:<18s}  '
                  f'f_P={h["f_P"][-1]:.3f}  f_C1={h["f_C1"][-1]:.3f}  '
                  f'f_C2={h["f_C2"][-1]:.3f}  M*={h["M_star_mean"][-1]:.4f}  卦:{name}')

        if obs.is_dead:
            print(f'\n[热寂] tick={tick}  系统进入平衡态，终止模拟。')
            render(phys, obs, clock, tick)
            break

    print('\n模拟结束。')
    _save_summary(obs, tuner)


def _save_summary(obs: ObservationLayer, tuner=None):
    h = obs.history
    if not h['tick']:
        return
    cat_cnt = sum(h['catastrophe'])
    print('\n── 生态演化摘要 ──────────────────────────────────────')
    print(f'  总帧数   : {h["tick"][-1]}')
    print(f'  最终 f_P : {h["f_P"][-1]:.4f}  (无机能量残余)')
    print(f'  最终 f_C1: {h["f_C1"][-1]:.4f}  (生物个体占比)')
    print(f'  最终 f_C2: {h["f_C2"][-1]:.4f}  (顶极群落占比)')
    print(f'  M* 峰值  : {max(h["M_star_mean"]):.5f}  (地层积累峰值)')
    print(f'  I* 峰值  : {max(h["I_star_mean"]):.5f}  (多样性峰值)')
    print(f'  灾变次数 : {cat_cnt}')
    print(f'  最终阶段 : {h["phase"][-1]}')
    print('──────────────────────────────────────────────────────')

    lines = [
        f'# 涌积态地球生态引擎 v13.3 [{h["phase"][-1]}] — 模拟报告',
        f'\n**实验预设**：{obs.cfg.experiment.upper()}',
        f'\n**运行时间**：{_RUN_TIME}',
        f'**输出目录**：`{OUTPUT_DIR}`\n',
        '---\n',
        '## 生态演化摘要\n',
        '| 指标 | 值 |', '|---|---|',
        f'| 总帧数 | {h["tick"][-1]} |',
        f'| 最终 f_P | {h["f_P"][-1]:.4f} |',
        f'| 最终 f_C1 | {h["f_C1"][-1]:.4f} |',
        f'| 最终 f_C2 | {h["f_C2"][-1]:.4f} |',
        f'| M* 峰值 | {max(h["M_star_mean"]):.5f} |',
        f'| I* 峰值 | {max(h["I_star_mean"]):.5f} |',
        f'| 灾变次数 | {cat_cnt} |',
        f'| 最终阶段 | {h["phase"][-1]} |',
        '\n## 自适应调参日志\n',
        tuner.summary() if tuner else '_未启用_\n',
    ]
    rp = OUTPUT_DIR / 'report.md'
    rp.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  [报告] 已保存至 {rp}')


if __name__ == '__main__':
    run()


