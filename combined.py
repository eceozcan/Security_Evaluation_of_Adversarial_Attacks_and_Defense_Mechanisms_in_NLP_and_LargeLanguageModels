import PromptInjection as PI
from PromptInjection.utils import open_config
import time


def log(msg, start_time):
    elapsed = time.time() - start_time
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


start_time = time.time()
log("Script started", start_time)

# Create the target task
log("Loading target task config", start_time)
target_task = PI.create_task(open_config(config_path='./configs/task_configs/sst2_config.json'), 60)
log("Target task ready", start_time)

# Create the model
log("Loading model config", start_time)
model_config = open_config(config_path='./configs/model_configs/deepseek-r1-distill-qwen-1.5b_config.json')
log("Creating model (this may take a long time on first run)", start_time)
model = PI.create_model(config=model_config)
log("Model ready", start_time)

# Create the injected task
log("Loading injected task", start_time)
inject_task = PI.create_task(open_config(config_path='./configs/task_configs/sms_spam_config.json'), 60, for_injection=True)
attacker = PI.create_attacker('combine', inject_task)
log("Injected task and attacker ready", start_time)

# Create the LLM-integrated App
log("Creating target app", start_time)
target_app = PI.create_app(target_task, model, defense='no')
log("Target app ready", start_time)

# Queries the model with the attacked data prompt and save the responses
log(f"Starting attack loop with {len(target_app)} samples", start_time)
attack_responses = list()
for i, (data_prompt, ground_truth_label) in enumerate(target_app):
    log(f"Sample {i+1}/{len(target_app)}: preparing prompt", start_time)
    data_prompt_after_attack = attacker.inject(data_prompt, i, target_task=target_task.task)
    log(f"Sample {i+1}/{len(target_app)}: querying model", start_time)
    response = target_app.query(data_prompt_after_attack, verbose=1, idx=i, total=len(target_app))
    attack_responses.append(response)
    log(f"Sample {i+1}/{len(target_app)}: response received", start_time)

# Create an evaluator to calculate the ASR
log("Evaluating ASR", start_time)
evaluator = PI.create_evaluator(
    target_task_responses=None, 
    target_task=target_task,
    injected_task_responses=None, 
    injected_task=attacker.task,
    attack_responses=attack_responses
)

log("Done", start_time)
print(f"ASR = {evaluator.asr}")