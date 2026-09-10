"""
涌积态宇宙模型 v5.0 — 测试版本
基于设计方案 v5.0（严密闭环版）实现

运行依赖：numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.ndimage import convolve, label
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings, os, pathlib
from datetime import datetime
warnings.filterwarnings('ignore')

# ── 中文字体配置（Windows）────────────────────
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ── 输出目录：<脚本同名文件夹>/<启动时间> ────────
_SCRIPT_DIR  = pathlib.Path(__file__).parent          # 脚本所在目录
_SCRIPT_NAME = pathlib.Path(__file__).stem            # 脚本名（不含扩展名）
_RUN_TIME    = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR   = _SCRIPT_DIR / _SCRIPT_NAME / _RUN_TIME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 一、参数配置
# ─────────────────────────────────────────────

class Config:
    # ── 网格 ──────────────────────────────────────────────────────────
    W, H   = 128, 128
    dt     = 0.05         # 时间步长（CFL: D_P*dt=0.01 << 0.25 ✓）
    steps  = 6000         # 最大帧数（扩展至 6000，观察图灵斑图长期维持行为）

    # ── 扩散系数（D_P >> D_C1 > D_C2，满足图灵不稳定性条件）─────────
    # 推导：比值 D_P/D_C2 = 25，足以触发 Turing 不稳定性
    D_P    = 0.20
    D_C1   = 0.05
    D_C2   = 0.008

    # ── 信息权重 ──────────────────────────────────────────────────────
    w_P    = 0.1
    w_C1   = 1.0
    w_C2   = 5.0

    # ── 有损存储（分层记忆）──────────────────────────────────────────
    # 理论要求：P 产生的信息记忆弱且衰减快，高阶形态记忆强且持久
    # 实现：将 I_increment 按来源分离，分别用不同 loss_rate 写入 I_stock
    #
    # P 层：写入门槛高（k_write_P 大），衰减快（loss_P 大）
    # C1层：中等写入门槛，中等衰减
    # C2层：写入门槛低（k_write_C2 小），衰减慢（loss_C2 小）
    k_write_P  = 0.001   # P 层写入门槛：与初始信号量级匹配，确保启动
    k_write_C1 = 0.005   # C1 层中等
    k_write_C2 = 0.0005  # C2 层写入门槛降低，使 decay_C2 产生的微弱 ΔC2 也能写入
    loss_P     = 0.15    # P 层记忆衰减快（半衰期 ~7 帧）
    loss_C1    = 0.04    # C1 层（半衰期 ~25 帧）；0.02 导致 C1 全域均匀积累，抹平图灵空间分化
    loss_C2    = 0.005   # C2 层衰减（半衰期 ~200 帧）；0.003 过低导致全域均匀化抹平图灵斑图

    # ── 传承失真 ──────────────────────────────────────────────────────
    k_norm    = 0.002    # 与分层存储稳态量级匹配（stock_max ~ 0.02）
    k_noise   = 0.30
    blur_size = 5

    # ── 拓扑清洗 ──────────────────────────────────────────────────────
    # softsign 防止 I_drive 过大；gauss 低通滤波保留中尺度结构；D_decay 整体缩放
    # 实验证明（T-050~T-052）：去掉 gauss 导致噪声直接驱动跃迁，C2 极化均匀化
    I_max_capacity = 1.0  # softsign 上限，需 > Theta_barrier（0.28）
    D_decay        = 0.15 # 整体缩放系数，等效于 Base_Rate × 0.85
    gauss_size     = 5

    # ── 跃迁势垒与速率 ────────────────────────────────────────────────
    # Theta=0.28：约为 max_I_drive 的 33%，创生有门槛但可达
    # Base_Rate_dec=1.2 > Base_Rate_2=0.4：信息驱动退化速率 > 创生速率，防止 C2 极化
    # decay_C1=0.06：C1 自发衰减回 P，消除 C1 积累陷阱（半衰期 ~17 帧）
    # decay_C2=0.005：C2 自发衰减回 C1，触发信息再生心跳（半衰期 ~200 帧）
    #   推导约束：decay_C2 ∈ (0.001, 0.015)
    #   下界：decay_C2 * C2* > 0.0002（能产生足够 I_inc_C2 重启信息链路）
    #   上界：decay_C2 * C2* < A_C1_to_C2_avg（不能超过供给量，否则 C2 无法积累）
    Theta_barrier  = 0.28
    Base_Rate_1    = 1.0
    Base_Rate_2    = 0.4
    Base_Rate_dec  = 1.2  # 信息驱动退化速率（需负向 I_drive 触发）
    decay_C1       = 0.06 # C1 自发衰减回 P，∈ (0, Base_Rate_1)
    decay_C2       = 0.008 # C2 自发衰减回 C1，增强心跳强度（0.005 时图灵期后 Phi_net 无法回升）

    # ── C2 空间不稳定机制（回差法控制）──────────────────────────────
    # 防止 C2 分布走向高度均匀，始终维持空间异质性
    # 原理：向 dC2_dt 注入去均值乘性噪声，全局积分为零（严格守恒）
    # 回差控制：C_var < theta_instab_low 时激活，C_var > theta_instab_high 时关闭
    k_instab         = 0.005  # 不稳定项强度（乘性噪声，维持 C_var 不继续下降）
    theta_instab_low = 0.005  # C_var 下界：低于此值激活不稳定项
    theta_instab_high= 0.020  # C_var 上界：高于此值关闭不稳定项（原 0.015，扩大激活范围）

    # ── I_stock_C2 空间扰动（方向A：重建 I_drive 空间梯度）──────────
    # 图灵期退出后 I_stock_C2 空间均匀 → I_drive 均匀 → 无法触发第二轮图灵不稳定性
    # 在 I_stock_C2 上注入空间扰动，重建 I_drive 梯度，为第二轮提供种子
    # 守恒性：乘性噪声去均值后 ΣΔI_stock_C2 = 0，I_stock_C2 总量不变（只是重分布）
    # 回差控制：I_stock_C2 的空间变异系数（CV = std/mean）低于阈值时激活
    k_mem_instab      = 0.008  # I_stock_C2 扰动强度（比 C2 场更小，记忆场更敏感）
    mem_instab_cv_low = 0.10   # CV 下界：低于此值激活（I_stock_C2 过于均匀）
    mem_instab_cv_high= 0.30   # CV 上界：高于此值关闭（已有足够空间异质性）

    # ── 自适应调参开关 ────────────────────────────────────────────────
    # 根据运行过程中的实时指标，在线微调关键参数，防止系统过早锁死
    # 设计原则：只在明确的"危险信号"出现时才介入，调整幅度保守（±20%以内）
    adaptive_tuning = True   # 总开关
    adapt_interval  = 50     # 每隔多少帧检查一次（避免过于频繁）

    # ── 观测 ──────────────────────────────────────────────────────────
    render_every   = 20
    struct_every   = 20
    # 区块识别阈值改为动态：用 phi_C2 均值 + 0.5*std，避免固定值失效
    phi_threshold  = None   # None = 动态阈值（均值 + 0.5*std）
    hd_window      = 300    # 热寂检测窗口加长，避免假平衡误判
    # 热寂判定改为双条件：活跃度 AND 驱动场同时趋零
    hd_drive_eps   = 5e-5   # I_drive 最大值低于此值才计入热寂


# ─────────────────────────────────────────────
# 二、自适应调参模块
# ─────────────────────────────────────────────

class AdaptiveTuner:
    """
    在线自适应调参器。
    
    设计原则：
    - 基于运行过程中的实时指标（而非事后分析）识别"危险信号"
    - 每隔 adapt_interval 帧检查一次，触发条件明确，调整幅度保守
    - 所有调整记录到日志，供事后分析
    - 参数有硬性上下界，防止调参失控
    
    当前支持的自适应规则（基于 2 次完整运行数据归纳）：
    
    规则1：图灵期衰退预警（Phi_net 连续下降 + C_var 下降）
      → 降低 loss_C2，增强 C2 层记忆，延缓信息链路断路
    
    规则2：C2 积累过慢（TURING_GROWTH 期 f_C2 增速 < 阈值）
      → 降低 Theta_barrier，降低跃迁门槛
    
    规则3：CHAOTIC_EDGE 锁死（连续 N 帧 Phi_net=0 且 f_C2 下降）
      → 降低 loss_C1，为 C2 提供更持久的 C1 供给缓冲
    
    规则4：C1 过度积累（f_C1 > 0.8 且 f_C2 < 0.05，UNIFORM_GROWTH 期过长）
      → 降低 Theta_barrier，加速 C1→C2 跃迁
    """

    # 参数硬性边界（防止调参失控）
    BOUNDS = {
        'loss_C2':       (0.001, 0.015),
        'loss_C1':       (0.005, 0.04),
        'Theta_barrier': (0.10,  0.40),
        'decay_C2':      (0.001, 0.020),
        'Base_Rate_2':   (0.2,   1.0),
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = []          # 调参日志：[(tick, param, old_val, new_val, reason)]
        self._phi_net_buf = [] # 最近 N 帧的 Phi_net
        self._c_var_buf   = [] # 最近 N 帧的 C_var
        self._chaotic_cnt = 0  # 连续 CHAOTIC_EDGE 帧计数
        self._turing_fc2_entry = None  # 进入 TURING_GROWTH 时的 f_C2
        self._turing_entry_tick = None
        self._last_phase = None
        self._buf_len = 100    # 滑动窗口长度
        self._theta_rescued = False   # 规则7是否已临时降低 Theta_barrier
        self._theta_original = None   # 规则7降低前的原始 Theta_barrier 值
        self._noise_boosted = False   # 规则7b 是否已临时提升 k_noise
        self._noise_boost_tick = None # k_noise 提升的起始 tick
        self._noise_original = None   # 提升前的原始 k_noise 值
        self._noise_boost_duration = 500  # k_noise 提升持续帧数

    def update(self, tick: int, phase: str, f_C2: float, f_C1: float,
               phi_net: float, c_var: float, delta_sig: float):
        """每帧调用，更新内部状态"""
        cfg = self.cfg

        # 更新滑动缓冲
        self._phi_net_buf.append(phi_net)
        self._c_var_buf.append(c_var)
        if len(self._phi_net_buf) > self._buf_len:
            self._phi_net_buf.pop(0)
            self._c_var_buf.pop(0)

        # 追踪 TURING_GROWTH 入口
        if phase == 'TURING_GROWTH' and self._last_phase != 'TURING_GROWTH':
            self._turing_fc2_entry = f_C2
            self._turing_entry_tick = tick
            self._theta_rescued = False   # 新一轮图灵期重置救援状态

        # 连续 CHAOTIC_EDGE 计数
        if phase == 'CHAOTIC_EDGE':
            self._chaotic_cnt += 1
        else:
            self._chaotic_cnt = 0

        self._last_phase = phase

        # k_noise 临时提升的自动恢复（每帧检查，不受 check_interval 限制）
        if (self._noise_boosted
                and self._noise_boost_tick is not None
                and tick - self._noise_boost_tick >= self._noise_boost_duration):
            old = cfg.k_noise
            cfg.k_noise = self._noise_original
            entry = (tick, 'k_noise', old, self._noise_original,
                     f'k_noise提升持续{self._noise_boost_duration}帧后自动恢复')
            self.log.append(entry)
            print(f'  [AdaptTune] tick={tick:05d}  k_noise: {old:.4f} → {self._noise_original:.4f}')
            print(f'              原因: k_noise提升持续{self._noise_boost_duration}帧后自动恢复')
            self._noise_boosted = False
            self._noise_boost_tick = None
            self._noise_original = None

        # 只在 adapt_interval 的整数倍帧触发检查
        # TURING_GROWTH 期间缩短检查间隔至 20 帧，提高规则7的捕捉概率
        check_interval = 20 if phase == 'TURING_GROWTH' else cfg.adapt_interval
        if tick % check_interval != 0 or tick == 0:
            return

        self._check_rules(tick, phase, f_C2, f_C1, phi_net, c_var, delta_sig)

    def _check_rules(self, tick, phase, f_C2, f_C1, phi_net, c_var, delta_sig):
        """检查所有规则，按优先级触发"""
        cfg = self.cfg

        # ── 规则1：图灵期衰退预警 ──────────────────────────────────
        # 信号：处于 TURING_GROWTH，但 Phi_net 在最近 100 帧内持续下降
        # 且 C_var 也在下降（空间结构正在瓦解）
        if (phase == 'TURING_GROWTH'
                and len(self._phi_net_buf) >= 50
                and self._is_declining(self._phi_net_buf[-50:], threshold=0.3)
                and self._is_declining(self._c_var_buf[-50:], threshold=0.1)):
            # 降低 loss_C2，增强 C2 层记忆持久性
            old = cfg.loss_C2
            new = max(self.BOUNDS['loss_C2'][0], old * 0.80)
            if new < old - 1e-6:
                self._apply('loss_C2', old, new, tick,
                            f'图灵期衰退预警：Phi_net和C_var持续下降，降低loss_C2增强记忆')

        # ── 规则2：C2 积累过慢 ──────────────────────────────────────
        # 信号：处于 TURING_GROWTH 超过 200 帧，但 f_C2 增速 < 0.0001/帧
        if (phase == 'TURING_GROWTH'
                and self._turing_entry_tick is not None
                and tick - self._turing_entry_tick > 200):
            elapsed = tick - self._turing_entry_tick
            fc2_gain = f_C2 - (self._turing_fc2_entry or 0)
            rate = fc2_gain / elapsed
            if rate < 0.0001 and f_C2 < 0.15:
                old = cfg.Theta_barrier
                new = max(self.BOUNDS['Theta_barrier'][0], old * 0.90)
                if new < old - 1e-6:
                    self._apply('Theta_barrier', old, new, tick,
                                f'C2积累过慢(rate={rate:.6f}/帧)，降低势垒')

        # ── 规则3：CHAOTIC_EDGE 锁死 ────────────────────────────────
        # 信号：连续 300 帧 CHAOTIC_EDGE，且 f_C2 在下降（不是稳定平衡）
        if (self._chaotic_cnt >= 300
                and len(self._phi_net_buf) >= 50
                and phi_net == 0.0
                and f_C2 < 0.20):
            # 降低 loss_C1，延长 C1 记忆，为 C2 提供更持久的供给缓冲
            old = cfg.loss_C1
            new = max(self.BOUNDS['loss_C1'][0], old * 0.80)
            if new < old - 1e-6:
                self._apply('loss_C1', old, new, tick,
                            f'CHAOTIC_EDGE锁死{self._chaotic_cnt}帧，降低loss_C1延长C1记忆')
                self._chaotic_cnt = 0  # 重置计数，避免连续触发

        # ── 规则4：C1 过度积累 ──────────────────────────────────────
        # 信号：f_C1 > 0.75 且 f_C2 < 0.05，说明 C1→C2 跃迁受阻
        if (phase in ('UNIFORM_GROWTH', 'TURING_GROWTH')
                and f_C1 > 0.75 and f_C2 < 0.05
                and tick > 100):
            old = cfg.Theta_barrier
            new = max(self.BOUNDS['Theta_barrier'][0], old * 0.85)
            if new < old - 1e-6:
                self._apply('Theta_barrier', old, new, tick,
                            f'C1过度积累(f_C1={f_C1:.3f})，降低势垒加速C1→C2跃迁')

        # ── 规则5：C_var 长期过低（全域均匀化）────────────────────────
        # 信号：UNIFORM_GROWTH 持续超过 500 帧，且 C_var 始终 < 0.001
        # 注意：此规则仅在 loss_C1 < 0.03 时激活（loss_C1=0.04 时图灵斑图正常，
        #       规则5会把 loss_C2 推至上界 0.015，破坏图灵期稳定性）
        if (phase == 'UNIFORM_GROWTH'
                and tick > 500
                and c_var < 0.001
                and f_C2 > 0.1
                and cfg.loss_C1 < 0.03   # 仅在 loss_C1 过低（全域均匀化根因）时才激活
                and (not self.log or tick - self.log[-1][0] >= 200)):  # 至少间隔 200 帧
            old = cfg.loss_C2
            new = min(self.BOUNDS['loss_C2'][1], old * 1.10)
            if new > old + 1e-6:
                self._apply('loss_C2', old, new, tick,
                            f'C_var长期过低({c_var:.5f})，UNIFORM_GROWTH已持续{tick}帧，'
                            f'提高loss_C2恢复空间分化')

        # ── 规则6：loss_C2 上调后仍崩溃（防止触顶后 CHAOTIC_EDGE）──────
        # 信号：CHAOTIC_EDGE 锁死，且 loss_C2 已被规则5上调过（> 初始值 0.005）
        # 动作：适度回调 loss_C2，防止记忆衰减过快导致信息链路断路
        if (self._chaotic_cnt >= 200
                and phi_net == 0.0
                and cfg.loss_C2 > 0.007   # 只在被上调过的情况下才回调
                and (not self.log or tick - self.log[-1][0] >= 200)):
            old = cfg.loss_C2
            new = max(0.005, old * 0.90)  # 回调 10%，但不低于初始值
            if new < old - 1e-6:
                self._apply('loss_C2', old, new, tick,
                            f'loss_C2上调后仍CHAOTIC_EDGE锁死{self._chaotic_cnt}帧，'
                            f'回调loss_C2防止记忆衰减过快')

        # ── 规则7：图灵期衰退救援（临时降低 Theta_barrier）────────────
        # 信号：处于 TURING_GROWTH，C_var 在 0.010~0.040 之间且 Phi_net < 40
        # 机制：Phi_net 指数衰减形成正反馈（C_var↓→Phi_net↓→C_var↓），
        #       临时降低 Theta_barrier 让弱 I_drive 也能维持 C1→C2 跃迁，
        #       打破衰减循环，待 C_var 回升后由规则7b 恢复原值
        # T-025 修复：去掉 _is_declining 检查（C_var 慢速衰减在 50 帧窗口内
        #             下降幅度 <3%，永远无法满足 10% 阈值），改为直接判断绝对值
        if (phase == 'TURING_GROWTH'
                and not self._theta_rescued          # 尚未救援过（每轮图灵期只救援一次）
                and c_var < 0.040                    # C_var 已从峰值下降约 40%
                and c_var > 0.010                    # 但尚未完全退出图灵期（留有余地）
                and phi_net < 40.0                   # Phi_net 已衰减，心跳减弱
                and (not self.log or tick - self.log[-1][0] >= 200)):
            self._theta_original = cfg.Theta_barrier
            old = cfg.Theta_barrier
            new = max(self.BOUNDS['Theta_barrier'][0], old * 0.72)  # 降至约 0.20
            self._theta_rescued = True
            self._apply('Theta_barrier', old, new, tick,
                        f'图灵期衰退救援：C_var={c_var:.4f}，Phi_net={phi_net:.2f}，'
                        f'临时降低Theta_barrier打破衰减循环')

        # ── 规则7b：图灵期救援后恢复 Theta_barrier + 临时提升 k_noise ──
        # 信号：规则7已触发，且 C_var 已回升超过救援触发值（说明救援有效）
        #       或 C_var 继续下降退出图灵期（救援无效，也需恢复避免影响后续）
        # T-027 修正：恢复阈值从 0.030 提高至 0.050，避免过早恢复后 C_var 再次下滑
        # T-029 新增：恢复 Theta_barrier 的同时临时提升 k_noise（0.30→0.45），
        #             为图灵期退出后的空间重分化提供更强扰动种子，持续 500 帧后自动恢复
        if (self._theta_rescued
                and self._theta_original is not None
                and (not self.log or tick - self.log[-1][0] >= 300)):
            should_restore = (
                c_var > 0.050                        # C_var 已回升至较高水平（原 0.030）
                or phase != 'TURING_GROWTH'          # 已退出图灵期，救援结束
            )
            if should_restore:
                old = cfg.Theta_barrier
                new = self._theta_original
                if abs(new - old) > 1e-6:
                    self._apply('Theta_barrier', old, new, tick,
                                f'图灵期救援结束（C_var={c_var:.4f}，phase={phase}），'
                                f'恢复Theta_barrier至原值{new:.3f}')
                self._theta_rescued = False
                self._theta_original = None

                # 同步临时提升 k_noise，为空间重分化提供扰动种子
                if not self._noise_boosted:
                    self._noise_original = cfg.k_noise
                    self._noise_boost_tick = tick
                    self._noise_boosted = True
                    new_noise = min(0.50, cfg.k_noise * 1.50)  # 提升 50%，上限 0.50
                    self._apply('k_noise', cfg.k_noise, new_noise, tick,
                                f'图灵期救援结束后临时提升k_noise，为空间重分化提供扰动种子'
                                f'（将在{self._noise_boost_duration}帧后自动恢复）')

    def _is_declining(self, buf, threshold=0.2):
        """判断序列是否整体下降：后半段均值 < 前半段均值 * (1-threshold)"""
        if len(buf) < 10:
            return False
        mid = len(buf) // 2
        first_half = np.mean(buf[:mid])
        second_half = np.mean(buf[mid:])
        if first_half < 1e-9:
            return False
        return second_half < first_half * (1 - threshold)

    def _apply(self, param: str, old_val: float, new_val: float,
               tick: int, reason: str):
        """应用参数调整"""
        setattr(self.cfg, param, new_val)
        entry = (tick, param, old_val, new_val, reason)
        self.log.append(entry)
        print(f'  [AdaptTune] tick={tick:05d}  {param}: {old_val:.6f} → {new_val:.6f}')
        print(f'              原因: {reason}')

    def summary(self) -> str:
        """返回调参日志的 Markdown 格式摘要"""
        if not self.log:
            return '_本次运行未触发任何自适应调参_\n'
        lines = ['| tick | 参数 | 原值 | 新值 | 触发原因 |',
                 '|---|---|---|---|---|']
        for tick, param, old, new, reason in self.log:
            lines.append(f'| {tick} | {param} | {old:.6f} | {new:.6f} | {reason} |')
        return '\n'.join(lines) + '\n'




def make_kernel(size, kind='blur'):
    """生成归一化卷积核"""
    k = np.ones((size, size), dtype=np.float32)
    if kind == 'gauss':
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                k[i, j] = np.exp(-((i-cx)**2 + (j-cy)**2) / (2*(size/4)**2))
    k /= k.sum()
    return k

# 拉普拉斯算子（5点差分，周期边界）
LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                               [1,-4, 1],
                               [0, 1, 0]], dtype=np.float32)

def laplacian(field):
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

        # 初始化场：P 非均匀随机，C1/C2 为零
        rng = np.random.default_rng(42)
        self.P  = rng.uniform(0.2, 1.0, shape).astype(np.float32)
        self.C1 = np.zeros(shape, np.float32)
        self.C2 = np.zeros(shape, np.float32)
        self.I_stock  = np.zeros(shape, np.float32)
        self.I_stock_P  = np.zeros(shape, np.float32)  # P 层独立存量
        self.I_stock_C1 = np.zeros(shape, np.float32)  # C1 层独立存量
        self.I_stock_C2 = np.zeros(shape, np.float32)  # C2 层独立存量

        # 上一帧状态（用于计算 delta）
        # 关键：P_prev 初始化为扩散一步后的状态，确保第一帧有非零 delta_P
        self.P_prev  = np.maximum(
            self.P + cfg.D_P * laplacian(self.P) * cfg.dt, 0.0)
        self.C1_prev = self.C1.copy()
        self.C2_prev = self.C2.copy()

        # 预计算卷积核
        self.blur_k  = make_kernel(cfg.blur_size, 'blur')
        self.gauss_k = make_kernel(cfg.gauss_size, 'gauss')

        # 上一帧的中间量（供观测层读取）
        self.I_increment = np.zeros(shape, np.float32)
        self.I_drive     = np.zeros(shape, np.float32)
        self.A_C1_to_C2  = np.zeros(shape, np.float32)
        self.A_C2_to_C1  = np.zeros(shape, np.float32)

        # C2 空间不稳定机制的回差状态
        self._instab_active = False  # 当前是否激活不稳定项
        # I_stock_C2 空间扰动的回差状态
        self._mem_instab_active = False  # 当前是否激活记忆场扰动

    def step(self):
        cfg = self.cfg
        P, C1, C2 = self.P, self.C1, self.C2

        # ── 步骤1：信息增量生成（按形态等级分离）──────────
        dP  = np.abs(P  - self.P_prev)
        dC1 = np.abs(C1 - self.C1_prev)
        dC2 = np.abs(C2 - self.C2_prev)
        # 各层独立增量（用于分层写入）
        I_inc_P  = cfg.w_P  * dP
        I_inc_C1 = cfg.w_C1 * dC1
        I_inc_C2 = cfg.w_C2 * dC2
        I_inc    = I_inc_P + I_inc_C1 + I_inc_C2
        self.I_increment = I_inc

        # ── 步骤2：分层有损存储（三个独立存量场，各自独立衰减）──────
        # P 层：高写入门槛 + 快衰减（底噪记忆弱且短暂）
        pw_P  = I_inc_P  / (I_inc_P  + cfg.k_write_P)
        mk_P  = (np.random.rand(*P.shape) < pw_P).astype(np.float32)
        self.I_stock_P = (self.I_stock_P * (1.0 - cfg.loss_P)
                          + I_inc_P * mk_P * cfg.dt)

        # C1层：中等写入门槛 + 中等衰减
        pw_C1 = I_inc_C1 / (I_inc_C1 + cfg.k_write_C1)
        mk_C1 = (np.random.rand(*P.shape) < pw_C1).astype(np.float32)
        self.I_stock_C1 = (self.I_stock_C1 * (1.0 - cfg.loss_C1)
                           + I_inc_C1 * mk_C1 * cfg.dt)

        # C2层：低写入门槛 + 慢衰减（高阶结构记忆强且持久）
        pw_C2 = I_inc_C2 / (I_inc_C2 + cfg.k_write_C2)
        mk_C2 = (np.random.rand(*P.shape) < pw_C2).astype(np.float32)
        self.I_stock_C2 = (self.I_stock_C2 * (1.0 - cfg.loss_C2)
                           + I_inc_C2 * mk_C2 * cfg.dt)

        # ── I_stock_C2 空间扰动（方向A：重建 I_drive 空间梯度）──────
        # 图灵期退出后 I_stock_C2 空间均匀，I_drive 失去梯度，无法触发第二轮图灵不稳定性
        # 在 I_stock_C2 上注入乘性噪声（去均值），重建记忆场的空间异质性
        # 守恒：ΣΔI_stock_C2 = 0，总存量不变，只是空间重分布
        # 回差控制：用变异系数 CV = std/mean 衡量 I_stock_C2 的空间均匀程度
        _s2_mean = self.I_stock_C2.mean()
        if _s2_mean > 1e-8:
            _s2_cv = self.I_stock_C2.std() / _s2_mean
            if _s2_cv < cfg.mem_instab_cv_low:
                self._mem_instab_active = True
            elif _s2_cv > cfg.mem_instab_cv_high:
                self._mem_instab_active = False
            # 中间区间：保持当前状态（回差）

            if self._mem_instab_active:
                mem_noise = spatial_noise(C2.shape)
                delta_mem = cfg.k_mem_instab * self.I_stock_C2 * mem_noise
                delta_mem -= delta_mem.mean()   # 去均值：总存量不变
                self.I_stock_C2 = np.maximum(self.I_stock_C2 + delta_mem, 0.0)

        # 合并：总存量 = 三层之和（各层以自己的速率独立衰减，数学严格）
        self.I_stock = self.I_stock_P + self.I_stock_C1 + self.I_stock_C2

        # ── 步骤3：传承失真 ──────────────────────
        I_norm    = self.I_stock / (self.I_stock + cfg.k_norm)
        I_context = convolve(I_norm, self.blur_k, mode='wrap')
        noise     = spatial_noise(P.shape) * cfg.k_noise * I_context
        I_raw     = I_context + noise

        # ── 步骤4：拓扑清洗 ──────────────────────
        I_pol    = softsign(I_raw, cfg.I_max_capacity)
        I_damped = (1.0 - cfg.D_decay) * convolve(I_pol, self.gauss_k, mode='wrap')
        I_drive  = np.where(np.abs(I_damped) < 1e-4, 0.0, I_damped)
        self.I_drive = I_drive
        self._I_norm_vis = I_norm

        # ── 步骤5：形态更新 ──────────────────────────────────────────
        # 作用力算子（信息驱动，需要 I_drive 触发）
        A_P_to_C1  = np.maximum(0,  I_drive) * cfg.Base_Rate_1 * P
        A_C1_to_P  = np.abs(np.minimum(0, I_drive)) * cfg.Base_Rate_1 * C1
        A_C1_to_C2 = np.maximum(0, I_drive - cfg.Theta_barrier) * cfg.Base_Rate_2 * C1
        A_C2_to_C1 = np.abs(np.minimum(0, I_drive)) * cfg.Base_Rate_dec * C2

        # 自发退化项（不依赖 I_drive，信息枯竭时仍持续运作）
        # C1 自发衰减回 P：C1 是不稳定粒子态，天然有半衰期
        A_C1_decay = cfg.decay_C1 * C1
        # C2 自发衰减回 C1：C2 稳定后产生 ΔC2，触发信息再生，打破惰性冻结
        # 退化方向：C2→C1（而非 C2→P），退化产生的 C1 变化也贡献信息增量
        A_C2_decay = cfg.decay_C2 * C2

        self.A_C1_to_C2 = A_C1_to_C2
        self.A_C2_to_C1 = A_C2_to_C1

        # 保存当前帧为下一帧的 prev
        self.P_prev  = P.copy()
        self.C1_prev = C1.copy()
        self.C2_prev = C2.copy()

        # 偏微分更新方程（完整供给-消耗平衡）
        # P：被 C1 生成消耗，被 C1 信息退化和自发衰减补充
        dP_dt  = (cfg.D_P  * laplacian(P)
                  - A_P_to_C1 + A_C1_to_P + A_C1_decay)
        # C1：被 P 供给，被信息退化/自发衰减消耗，被 C2 信息退化和自发衰减补充
        dC1_dt = (cfg.D_C1 * laplacian(C1)
                  + A_P_to_C1 - A_C1_to_P - A_C1_to_C2 + A_C2_to_C1
                  - A_C1_decay + A_C2_decay)
        # C2：被 C1 供给（需突破势垒），被信息退化和自发衰减消耗
        dC2_dt = (cfg.D_C2 * laplacian(C2)
                  + A_C1_to_C2 - A_C2_to_C1 - A_C2_decay)

        # ── C2 空间不稳定机制（回差法控制）──────────────────────────
        # 当 C_var 低于下界时激活，高于上界时关闭，中间区间保持当前状态
        C2_var = float(C2.var())
        if C2_var < cfg.theta_instab_low:
            self._instab_active = True
        elif C2_var > cfg.theta_instab_high:
            self._instab_active = False
        # 中间区间：保持 _instab_active 当前状态（回差）

        if self._instab_active and C2.mean() > 1e-6:
            # 乘性噪声：去均值后严格守恒（∫A dV = 0），不受 np.maximum 截断影响
            noise = spatial_noise(C2.shape)
            A_instab = cfg.k_instab * C2 * noise
            A_instab -= A_instab.mean()
            dC2_dt = dC2_dt + A_instab

        self.P  = np.maximum(P  + dP_dt  * cfg.dt, 0.0)
        self.C1 = np.maximum(C1 + dC1_dt * cfg.dt, 0.0)
        self.C2 = np.maximum(C2 + dC2_dt * cfg.dt, 0.0)


# ─────────────────────────────────────────────
# 四、观测层
# ─────────────────────────────────────────────

class ObservationLayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = {k: [] for k in [
            'tick', 'M_total', 'f_C2', 'f_C1',
            'delta_sigma', 'Phi_net', 'C_var', 'H_C2',
            'n_clusters', 'size_gini', 'phase'
        ]}
        self._hd_buf = []   # 热寂检测缓冲

    def observe(self, phys: PhysicsLayer, tick: int):
        cfg = self.cfg
        P, C1, C2 = phys.P, phys.C1, phys.C2
        eps = 1e-9

        # 全局质量
        M_P, M_C1, M_C2 = P.sum(), C1.sum(), C2.sum()
        M_total = M_P + M_C1 + M_C2
        f_C2 = M_C2 / (M_total + eps)
        f_C1 = M_C1 / (M_total + eps)

        # 信息热力学（分层记忆：用三层 loss 的加权均值估算总遗忘率）
        I_bar     = phys.I_stock.mean()
        sig_inc   = phys.I_increment.sum()
        # 用 loss_C2 作为基础衰减率（I_stock 以 loss_C2 为基础衰减）
        sig_loss  = cfg.loss_C2 * I_bar * P.size
        delta_sig = sig_inc - sig_loss

        # 结构复杂度
        phi_C2 = C2 / (P + C1 + C2 + eps)
        H_C2   = -(phi_C2 * np.log(phi_C2 + eps)).sum()
        C_var  = phi_C2.var()

        # 跃迁通量
        Phi_net = phys.A_C1_to_C2.sum() - phys.A_C2_to_C1.sum()

        # 演化阶段
        phase = self._classify(delta_sig, Phi_net, C_var, f_C2)

        # 区块统计（每 struct_every 帧）
        n_cl, s_gini = 0, 0.0
        if tick % cfg.struct_every == 0:
            n_cl, s_gini = self._struct_stats(phi_C2)

        # 热寂检测：双条件（活跃度 + 驱动场同时趋零）
        self._hd_buf.append((phys.I_increment.max(), abs(phys.I_drive).max()))
        if len(self._hd_buf) > cfg.hd_window:
            self._hd_buf.pop(0)

        # 记录
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
        # 双条件：活跃度 AND 驱动场在整个窗口内都低于阈值
        inc_vals   = [v[0] for v in self._hd_buf]
        drive_vals = [v[1] for v in self._hd_buf]
        return (max(inc_vals) < 1e-4 and
                max(drive_vals) < self.cfg.hd_drive_eps)

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
        # 动态阈值：均值 + 0.5*std，自适应 phi_C2 的实际值域
        if self.cfg.phi_threshold is None:
            threshold = phi_C2.mean() + 0.5 * phi_C2.std()
        else:
            threshold = self.cfg.phi_threshold
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

    # ── 空间场图 ────────────────────────────────
    fields = [
        (phi_C2,
         'viridis', None,
         'φ_C2  高阶结构密度\n蓝=低  黄=高'),
        (phys.I_drive,
         'RdBu_r', TwoSlopeNorm(0),
         'I_drive_C1  C1层驱动场\n红=生长前沿  蓝=退化前沿  白=零'),
        (phys.I_increment,
         'hot', None,
         '活跃度  I_increment\n黑=静止  黄白=剧烈变化'),
        (phys._I_norm_vis,
         'cividis', None,
         '信息存量  I_norm\n深=记忆弱  亮=记忆强'),
    ]
    axes_map = []
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
        axes_map.append(ax)

    # ── 时序曲线 ────────────────────────────────
    h = obs.history
    ticks = h['tick']

    curve_specs = [
        ('f_C2',        '#2ecc71', 'f_C2  高阶结构占比'),
        ('f_C1',        '#3498db', 'f_C1  中间态占比'),
        ('delta_sigma', '#e67e22', 'ΔΣ  信息净增率'),
        ('Phi_net',     '#e74c3c', 'Φ_net  净跃迁通量'),
    ]
    ax_ts = fig.add_subplot(gs[1, :3])
    ax_ts.set_facecolor('#1a1a1a')
    for key, color, label_ in curve_specs:
        ax_ts.plot(ticks, h[key], color=color, lw=1.0, label=label_)

    # 阶段色带
    if len(ticks) > 1:
        phases = h['phase']
        for i in range(len(ticks) - 1):
            ax_ts.axvspan(ticks[i], ticks[i+1],
                          alpha=0.12,
                          color=PHASE_COLORS.get(phases[i], '#ffffff'))

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

    # ── 区块统计 ────────────────────────────────
    ax_cl = fig.add_subplot(gs[1, 3])
    ax_cl.set_facecolor('#1a1a1a')
    ax_cl.plot(ticks, h['n_clusters'], color='#f39c12', lw=1.0, label='区块数量')
    ax_cl.plot(ticks, h['size_gini'],  color='#1abc9c', lw=1.0, label='大小基尼系数')
    ax_cl.legend(loc='upper left', fontsize=7,
                 facecolor='#1a1a1a', labelcolor='white', framealpha=0.6)
    ax_cl.tick_params(colors='white', labelsize=7)
    ax_cl.set_xlabel('帧数 (tick)', color='white', fontsize=8)
    ax_cl.set_title('空间结构统计', color='#aaaaaa', fontsize=8, pad=4)
    for spine in ax_cl.spines.values():
        spine.set_edgecolor('#444')

    # ── 阶段图例 ────────────────────────────────
    phase_labels = {
        'TURING_GROWTH':    '图灵斑图涌现',
        'UNIFORM_GROWTH':   '均匀生长期',
        'RESTRUCTURING':    '信息活跃·结构重组',
        'COLLAPSE':         '信息衰退·结构崩解',
        'COARSENING':       '粗化·区块吞噬',
        'NEAR_EQUILIBRIUM': '接近局部平衡',
        'CHAOTIC_EDGE':     '混沌边缘',
    }
    legend_text = '  '.join(
        f'█ {phase_labels[k]}' for k in phase_labels
    )
    fig.text(0.5, 0.01, legend_text, ha='center', va='bottom',
             fontsize=6.5, color='#888888',
             bbox=dict(facecolor='#1a1a1a', edgecolor='none', pad=2))

    # ── 标题 ────────────────────────────────────
    phase_now = h['phase'][-1] if h['phase'] else '—'
    phase_cn = {
        'TURING_GROWTH': '图灵斑图涌现', 'UNIFORM_GROWTH': '均匀生长',
        'RESTRUCTURING': '结构重组', 'COLLAPSE': '崩解',
        'COARSENING': '粗化吞噬', 'NEAR_EQUILIBRIUM': '近平衡',
        'CHAOTIC_EDGE': '混沌边缘',
    }
    color_now = PHASE_COLORS.get(phase_now, 'white')
    fig.suptitle(
        f'涌积态宇宙 v5.0   第 {tick:05d} 帧   '
        f'C2占比={h["f_C2"][-1]:.3f}   '
        f'当前阶段：{phase_cn.get(phase_now, phase_now)}',
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
    cfg  = Config()
    phys = PhysicsLayer(cfg)
    obs  = ObservationLayer(cfg)
    tuner = AdaptiveTuner(cfg) if cfg.adaptive_tuning else None

    print("=" * 60)
    print("涌积态宇宙模型 v5.0  启动")
    print(f"  网格: {cfg.W}×{cfg.H}   dt={cfg.dt}   最大帧数={cfg.steps}")
    print(f"  D_P={cfg.D_P}  D_C1={cfg.D_C1}  D_C2={cfg.D_C2}")
    print(f"  Theta_barrier={cfg.Theta_barrier}  k_noise={cfg.k_noise}")
    print(f"  自适应调参: {'开启' if cfg.adaptive_tuning else '关闭'}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    for tick in range(cfg.steps):
        phys.step()
        phase = obs.observe(phys, tick)

        # 自适应调参：每帧更新状态，每 adapt_interval 帧检查规则
        if tuner is not None:
            h = obs.history
            tuner.update(
                tick=tick,
                phase=phase,
                f_C2=h['f_C2'][-1],
                f_C1=h['f_C1'][-1],
                phi_net=h['Phi_net'][-1],
                c_var=h['C_var'][-1],
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
    # 守恒验证
    m0, m1 = h['M_total'][0], h['M_total'][-1]
    drift = abs(m1 - m0) / (m0 + 1e-9) * 100
    print(f"  质量守恒漂移: {drift:.4f}%")
    if tuner and tuner.log:
        print(f"  自适应调参次数: {len(tuner.log)}")
    print("──────────────────────────────────────────")
    _save_report(obs, tuner)


def _save_report(obs: ObservationLayer, tuner=None):
    """生成本次运行的关键节点数据报告，保存到输出目录下的 report.md"""
    h   = obs.history
    cfg = obs.cfg
    if not h['tick']:
        return

    ticks      = np.array(h['tick'])
    f_c2       = np.array(h['f_C2'])
    f_c1       = np.array(h['f_C1'])
    delta_sig  = np.array(h['delta_sigma'])
    phi_net    = np.array(h['Phi_net'])
    c_var      = np.array(h['C_var'])
    n_cl       = np.array(h['n_clusters'])
    gini_arr   = np.array(h['size_gini'])
    phases     = h['phase']

    total_ticks = h['tick'][-1]
    m0, m1 = h['M_total'][0], h['M_total'][-1]
    drift = abs(m1 - m0) / (m0 + 1e-9) * 100

    # ── 关键节点提取 ──────────────────────────────
    # 1. f_C2 峰值帧
    peak_idx  = int(np.argmax(f_c2))
    peak_tick = int(ticks[peak_idx])
    peak_fc2  = float(f_c2[peak_idx])

    # 2. 区块数峰值帧
    ncl_peak_idx  = int(np.argmax(n_cl))
    ncl_peak_tick = int(ticks[ncl_peak_idx])
    ncl_peak_val  = int(n_cl[ncl_peak_idx])

    # 3. 阶段切换节点
    phase_changes = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            phase_changes.append((int(ticks[i]), phases[i-1], phases[i]))

    # 4. 每 10% 进度的快照数据
    milestones = []
    for pct in range(0, 101, 10):
        idx = min(int(pct / 100 * (len(ticks) - 1)), len(ticks) - 1)
        milestones.append({
            'tick':      int(ticks[idx]),
            'f_C2':      float(f_c2[idx]),
            'f_C1':      float(f_c1[idx]),
            'delta_sig': float(delta_sig[idx]),
            'phi_net':   float(phi_net[idx]),
            'C_var':     float(c_var[idx]),
            'n_cl':      int(n_cl[idx]),
            'gini':      float(gini_arr[idx]),
            'phase':     phases[idx],
        })

    # ── 写入 MD ──────────────────────────────────
    lines = []
    lines.append(f"# 涌积态宇宙模型 v5.0 — 运行报告")
    lines.append(f"\n**运行时间**：{_RUN_TIME}  ")
    lines.append(f"**输出目录**：`{OUTPUT_DIR}`\n")

    lines.append("---\n")
    lines.append("## 一、运行参数\n")
    lines.append(f"| 参数 | 值 |")
    lines.append(f"|---|---|")
    lines.append(f"| 网格 | {cfg.W}×{cfg.H} |")
    lines.append(f"| dt | {cfg.dt} |")
    lines.append(f"| 最大帧数 | {cfg.steps} |")
    lines.append(f"| D_P / D_C1 / D_C2 | {cfg.D_P} / {cfg.D_C1} / {cfg.D_C2} |")
    lines.append(f"| k_write_P / C1 / C2 | {cfg.k_write_P} / {cfg.k_write_C1} / {cfg.k_write_C2} |")
    lines.append(f"| loss_P / C1 / C2 | {cfg.loss_P} / {cfg.loss_C1} / {cfg.loss_C2} |")
    lines.append(f"| k_norm | {cfg.k_norm} |")
    lines.append(f"| k_noise | {cfg.k_noise} |")
    lines.append(f"| Theta_barrier | {cfg.Theta_barrier} |")
    lines.append(f"| Base_Rate_1 / 2 / dec | {cfg.Base_Rate_1} / {cfg.Base_Rate_2} / {cfg.Base_Rate_dec} |")
    lines.append(f"| decay_C1 | {cfg.decay_C1} |")
    lines.append(f"| decay_C2 | {cfg.decay_C2} |")
    lines.append(f"| adaptive_tuning | {cfg.adaptive_tuning} |")

    lines.append("\n---\n")
    lines.append("## 二、总体摘要\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|---|---|")
    lines.append(f"| 实际运行帧数 | {total_ticks} |")
    lines.append(f"| 终止原因 | {'热寂' if total_ticks < cfg.steps - 1 else '达到最大帧数'} |")
    lines.append(f"| 最终 f_C2 | {f_c2[-1]:.4f} |")
    lines.append(f"| 最终 f_C1 | {f_c1[-1]:.4f} |")
    lines.append(f"| 最终演化阶段 | {phases[-1]} |")
    lines.append(f"| 最终区块数 | {int(n_cl[-1])} |")
    lines.append(f"| 最终 Gini 系数 | {float(gini_arr[-1]):.4f} |")
    lines.append(f"| 质量守恒漂移 | {drift:.6f}% |")
    lines.append(f"| f_C2 峰值 | {peak_fc2:.4f}（tick={peak_tick}）|")
    lines.append(f"| 区块数峰值 | {ncl_peak_val}（tick={ncl_peak_tick}）|")

    lines.append("\n---\n")
    lines.append("## 三、阶段切换节点\n")
    if phase_changes:
        lines.append(f"| tick | 从 | 到 |")
        lines.append(f"|---|---|---|")
        for t, frm, to in phase_changes:
            lines.append(f"| {t} | {frm} | {to} |")
    else:
        lines.append("_无阶段切换（系统始终处于同一阶段）_")

    lines.append("\n---\n")
    lines.append("## 四、自适应调参日志\n")
    if tuner is not None:
        lines.append(tuner.summary())
    else:
        lines.append("_自适应调参未启用_\n")

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
