"""
涌积态宇宙模型 v13.4 — 自动参数扫频与长程演化调度器
(ASUM v13.4: Automated Parameter Sweep & Deep-Time Scheduler)
"""

import subprocess, sys, time, os, glob, pathlib
from datetime import datetime

_SCRIPT_DIR = pathlib.Path(__file__).parent

def run_single(name, steps=60000, seed=42, geo_amp=0.004, hotspot_count=5, drift_speed=0.0008, experiment="free", render_every=100):
    env = os.environ.copy()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"{ts}_{name}"
    env["ASUM_STEPS"]          = str(steps)
    env["ASUM_SEED"]           = str(seed)
    env["ASUM_GEO_AMP"]        = str(geo_amp)
    env["ASUM_HOTSPOT_COUNT"]  = str(hotspot_count)
    env["ASUM_DRIFT_SPEED"]    = str(drift_speed)
    env["ASUM_EXPERIMENT"]     = experiment
    env["ASUM_RENDER_EVERY"]   = str(render_every)
    env["ASUM_OUTDIR"]         = out_name
    env["PYTHONIOENCODING"]    = "utf-8"

    cmd = [sys.executable, str(_SCRIPT_DIR / "universe_v13_4.py")]
    print(f"\n[启动任务] {name} (steps={steps}, geo_amp={geo_amp}, hotspots={hotspot_count}, drift={drift_speed}, exp={experiment})")
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8')
    out, err = p.communicate()
    elapsed = time.time() - t0
    print(f"[完成任务] {name} 耗时: {elapsed:.1f}s, 退出码: {p.returncode}")
    if err and p.returncode != 0:
        print(f"Error output:\n{err[:500]}")
    return out_name, elapsed

def parse_report(report_path):
    """解析 report.md 提取关键指标"""
    if not report_path.exists():
        return {}
    txt = report_path.read_text(encoding='utf-8')
    res = {}
    for line in txt.splitlines():
        if '| 最终 f_P |' in line:
            res['f_P'] = float(line.split('|')[2].strip())
        elif '| 最终 f_C1 |' in line:
            res['f_C1'] = float(line.split('|')[2].strip())
        elif '| 最终 f_C2 |' in line:
            res['f_C2'] = float(line.split('|')[2].strip())
        elif '| M* 峰值 |' in line:
            res['M_star'] = float(line.split('|')[2].strip())
        elif '| I* 峰值 |' in line:
            res['I_star'] = float(line.split('|')[2].strip())
        elif '| 灾变次数 |' in line:
            res['cat_cnt'] = int(line.split('|')[2].strip())
        elif '| 最终阶段 |' in line:
            res['phase'] = line.split('|')[2].strip()
    return res

from concurrent.futures import ProcessPoolExecutor, as_completed

def run_suite():
    print("=" * 72)
    print("开始并行执行 ASUM v13.4 完整参数扫频与长程演化套件 (多进程加速)")
    print("=" * 72)

    tasks = [
        # G1: 地热强度敏感性扫频 (geo_amp)
        ("sweep_geo_0001", 30000, 42, 0.001, 5, 0.0008, "free", 0),
        ("sweep_geo_0004", 30000, 42, 0.004, 5, 0.0008, "free", 0),
        ("sweep_geo_0010", 30000, 42, 0.010, 5, 0.0008, "free", 0),
        ("sweep_geo_0020", 30000, 42, 0.020, 5, 0.0008, "free", 0),

        # G2: 板块构造破碎度扫频 (hotspot_count)
        ("sweep_hotspot_2",  30000, 42, 0.004, 2,  0.0008, "free", 0),
        ("sweep_hotspot_10", 30000, 42, 0.004, 10, 0.0008, "free", 0),

        # G3: 深时演化与动态板块大漂移实验 (100k 帧)
        ("deeptime_drift_100k", 100000, 42, 0.004, 5, 0.0015, "free", 0),
    ]

    t_start = time.time()
    results = []

    # 使用多进程并发运行
    with ProcessPoolExecutor(max_workers=7) as executor:
        future_map = {
            executor.submit(run_single, name, steps, seed, geo_amp, hs_cnt, drift, exp, rend): (name, steps, seed, geo_amp, hs_cnt, drift, exp)
            for name, steps, seed, geo_amp, hs_cnt, drift, exp, rend in tasks
        }
        for future in as_completed(future_map):
            name, steps, seed, geo_amp, hs_cnt, drift, exp = future_map[future]
            try:
                out_dir_name, elapsed = future.result()
                rep_path = _SCRIPT_DIR / "universe_v13_4" / out_dir_name / "report.md"
                data = parse_report(rep_path)
                data['name'] = name
                data['steps'] = steps
                data['geo_amp'] = geo_amp
                data['hotspots'] = hs_cnt
                data['drift'] = drift
                data['elapsed'] = elapsed
                data['dir'] = out_dir_name
                results.append(data)
            except Exception as exc:
                print(f"任务 {name} 生成异常: {exc}")

    total_time = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"全部实验并发执行完毕！总耗时: {total_time:.1f}s")
    print("=" * 72)

    # 按照任务原有顺序排序
    task_order = {t[0]: i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: task_order.get(r.get('name', ''), 999))

    # 生成总对比汇总表
    table_lines = [
        "# ASUM v13.4 动态板块与地热扫频实验总对比表",
        f"\n**完成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**套件总耗时**：{total_time:.1f}s\n",
        "| 实验标识 | 步数 | 地热强度 | 热点数 | 漂移率 | f_P | f_C1 | f_C2 | M*峰值 | 灾变 | 最终相态 | 耗时(s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        table_lines.append(
            f"| {r.get('name','—')} | {r.get('steps','—')} | {r.get('geo_amp','—')} | {r.get('hotspots','—')} | {r.get('drift','—')} "
            f"| {r.get('f_P', 0):.4f} | {r.get('f_C1', 0):.4f} | **{r.get('f_C2', 0):.4f}** | {r.get('M_star', 0):.2f} "
            f"| {r.get('cat_cnt', 0)} | {r.get('phase', '—')} | {r.get('elapsed', 0):.1f} |"
        )

    summary_file = _SCRIPT_DIR / "universe_v13_4" / "sweep_summary.md"
    summary_file.write_text("\n".join(table_lines) + "\n", encoding='utf-8')
    print("\n" + "\n".join(table_lines))
    print(f"\n[汇总表已保存] {summary_file}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "all":
        run_suite()
    else:
        # 单个或部分运行
        tasks_map = {
            "g1": [
                ("sweep_geo_0001", 60000, 42, 0.001, 5, 0.0008, "free", 2000),
                ("sweep_geo_0004", 60000, 42, 0.004, 5, 0.0008, "free", 2000),
                ("sweep_geo_0010", 60000, 42, 0.010, 5, 0.0008, "free", 2000),
                ("sweep_geo_0020", 60000, 42, 0.020, 5, 0.0008, "free", 2000),
            ],
            "g2": [
                ("sweep_hotspot_2",  60000, 42, 0.004, 2,  0.0008, "free", 2000),
                ("sweep_hotspot_10", 60000, 42, 0.004, 10, 0.0008, "free", 2000),
            ],
            "g3": [
                ("deeptime_drift_150k", 150000, 42, 0.004, 5, 0.0015, "free", 5000),
            ]
        }
        tsks = tasks_map.get(mode, [])
        for name, steps, seed, geo_amp, hs_cnt, drift, exp, rend in tsks:
            run_single(name, steps, seed, geo_amp, hs_cnt, drift, exp, rend)

