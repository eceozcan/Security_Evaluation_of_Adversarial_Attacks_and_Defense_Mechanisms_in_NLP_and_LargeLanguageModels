import os
import time
import subprocess
from tqdm import tqdm
from collections import deque

# --- GPU YÖNETİMİ ---
NUM_GPUS = 1
JOBS_PER_GPU = 1

gpu_queue = deque()
for gpu_id in range(NUM_GPUS):
    for _ in range(JOBS_PER_GPU):
        gpu_queue.append(gpu_id)

active_jobs = {}

def run(model, dataset, injected_dataset, data_num, attack_strategy, api_key_pos, defense, gpu_id):
    model_config_path = f'./configs/model_configs/{model}_config.json'
    target_data_config_path = f'./configs/task_configs/{dataset}_config.json'
    injected_data_config_path = f'./configs/task_configs/{injected_dataset}_config.json'

    log_dir = './log5'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/{model}{dataset}{injected_dataset}{data_num}{attack_strategy}_{defense}.txt'
    save_path = f'./result5/{model}{dataset}{injected_dataset}{data_num}{attack_strategy}_{defense}'
    os.makedirs(save_path, exist_ok=True)

    cmd = [
        "python3", "-u", "main.py",
        "--model_config_path", model_config_path,
        "--target_data_config_path", target_data_config_path,
        "--injected_data_config_path", injected_data_config_path,
        "--data_num", str(data_num),
        "--save_path", save_path,
        "--attack_strategy", attack_strategy,
        "--api_key_pos", str(api_key_pos),
        "--defense", defense,
        "--recompute_attack",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)

    print(f"  [GPU {gpu_id}] Başlatıldı: {model} | {dataset} -> {injected_dataset}")
    return log_file, proc


def is_job_done(log_path, proc):
    if proc.poll() is None:
        return False
    if not os.path.exists(log_path):
        return False
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        return '[END]' in f.read()


def wait_for_free_gpu():
    while True:
        done = [lp for lp, (proc, gid) in active_jobs.items() if is_job_done(lp, proc)]
        for lp in done:
            _, freed_gpu = active_jobs.pop(lp)
            gpu_queue.append(freed_gpu)
            print(f"  [GPU {freed_gpu}] Serbest kaldı.")
        if gpu_queue:
            return gpu_queue.popleft()
        time.sleep(15)


def wait_all():
    print(f"\nTüm job'ların bitmesi bekleniyor ({len(active_jobs)} kaldı)...")
    while active_jobs:
        done = [lp for lp, (proc, gid) in active_jobs.items() if is_job_done(lp, proc)]
        for lp in done:
            _, freed_gpu = active_jobs.pop(lp)
            gpu_queue.append(freed_gpu)
            print(f"  [GPU {freed_gpu}] Tamamlandı.")
        if active_jobs:
            time.sleep(15)
    print("Tüm job'lar tamamlandı.\n")


# --- KONFİGÜRASYON ---
models = ['deepseek-llm-7b-chat','flan']
datasets = ['sst2', 'sms_spam', 'mrpc', 'hsol', 'rte', 'jfleg']
injected_datasets =  ['sst2', 'sms_spam', 'mrpc', 'hsol', 'rte', 'jfleg']
data_num = 100
attack_strategy = "combine"
defense = 'adaptive'

api_key_counts = {m: 1 for m in models}

# --- ANA DÖNGÜ ---
for model in models:
    print(f"\n{'='*50}\nModel: {model}\n{'='*50}")
    api_key_pos = 0

    for data in datasets:
        print(f"\n[Dataset: {data}]")
        for injected_data in injected_datasets:
            gpu_id = wait_for_free_gpu()
            log_path, proc = run(
                model, data, injected_data, data_num,
                attack_strategy, api_key_pos, defense, gpu_id
            )
            active_jobs[log_path] = (proc, gpu_id)
            api_key_pos = (api_key_pos + 1) % api_key_counts.get(model, 1)

wait_all()
print("TÜMÜ TAMAMLANDI.")