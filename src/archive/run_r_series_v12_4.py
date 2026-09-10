import subprocess, sys, time, os, glob, pathlib
from datetime import datetime

def run_single(steps, seed, beta=0.0, gamma=0.05, eta_h=0.001, rho=0.01):
    env = os.environ.copy()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"{ts}_{steps//1000}k_s{seed}"
    env["V12_STEPS"] = str(steps)
    env["V12_SEED"] = str(seed)
    env["V12_BETA"] = str(beta)
    env["V12_GAMMA"] = str(gamma)
    env["V12_ETA_H"] = str(eta_h)
    env["V12_RHO"] = str(rho)
    env["V12_OUTDIR"] = out_name
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "universe_v12-4.py"]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p, out_name

def run_batch_parallel(steps, seeds, beta=0.0, gamma=0.05, eta_h=0.001, rho=0.01):
    t0 = time.time()
    print(f"Starting batch: steps={steps}, seeds={seeds}")
    procs = [(seed, out_name, p) for seed in seeds for p, out_name in [run_single(steps, seed, beta, gamma, eta_h, rho)]]
    for seed, out_name, p in procs:
        out, err = p.communicate()
        print(f"Seed {seed} ({out_name}) finished, code={p.returncode}, elapsed={time.time()-t0:.1f}s")
    print(f"Batch completed in {time.time()-t0:.1f}s\n")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "200k"
    if mode == "200k":
        run_batch_parallel(200000, [42, 101, 202])
    elif mode == "500k":
        run_batch_parallel(500000, [42, 101, 202])
