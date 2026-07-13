import argparse
import os
import time
import numpy as np

import PromptInjection as PI
from PromptInjection.utils import open_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open Prompt Injection Experiments')
    parser.add_argument('--model_config_path', default='./configs/model_configs/gemini2.5_config.json', type=str)
    parser.add_argument('--target_data_config_path', default='./configs/task_configs/rte_config.json', type=str)
    parser.add_argument('--injected_data_config_path', default='./configs/task_configs/hsol_config.json', type=str)
    parser.add_argument('--data_num', default=100, type=int)
    parser.add_argument('--save_path', default='./result', type=str)
    parser.add_argument('--attack_strategy', default='combine', type=str)
    parser.add_argument('--api_key_pos', default=0, type=int)
    parser.add_argument('--defense', default='adaptive', type=str)
    parser.add_argument('--recompute_target', action='store_true')
    parser.add_argument('--recompute_injected', action='store_true')
    parser.add_argument('--recompute_attack', action='store_true')
    parser.add_argument('--verbose_target', default=0, type=int)
    parser.add_argument('--verbose_attack', default=1, type=int)
    parser.add_argument('--quiet_eval', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Preparation
    target_task = PI.create_task(open_config(config_path=args.target_data_config_path), args.data_num)

    model_config = open_config(config_path=args.model_config_path)
    assert (0 <= args.api_key_pos < len(model_config["api_key_info"]["api_keys"]))
    model_config["api_key_info"]["api_key_use"] = args.api_key_pos
    model = PI.create_model(config=model_config)

    inject_task = PI.create_task(open_config(config_path=args.injected_data_config_path), args.data_num, for_injection=True)
    attacker = PI.create_attacker(args.attack_strategy, inject_task)
    injected_dataset_name = inject_task.dataset

    target_app = PI.create_app(target_task, model, defense=args.defense)

    # Target task
    target_task_responses_path = f'{args.save_path}/target_task_responses.npz'
    if args.recompute_target and os.path.exists(target_task_responses_path):
        os.remove(target_task_responses_path)
    try:
        target_task_responses = np.load(target_task_responses_path, allow_pickle=True)['data']
    except:
        target_task_responses = []
        for i, (data_prompt, ground_truth_label) in enumerate(target_app):
            response = target_app.query(
                data_prompt,
                injected_task=injected_dataset_name,
                verbose=args.verbose_target,
                idx=i,
                total=len(target_app)
            )
            target_task_responses.append(response)
            if i % 2 == 0:
                time.sleep(1)
        np.savez(target_task_responses_path, data=target_task_responses)

    # Injected task
    injected_task_responses_path = f'{args.save_path}/injected_task_responses.npz'
    if args.recompute_injected and os.path.exists(injected_task_responses_path):
        os.remove(injected_task_responses_path)
    try:
        injected_task_responses = np.load(injected_task_responses_path, allow_pickle=True)['data']
    except:
        injected_task_responses = []
        for i, (data_prompt, ground_truth_label) in enumerate(attacker.task):
            injected_prompt = attacker.task.get_instruction() + '\nText: ' + data_prompt
            response = model.query(injected_prompt)
            injected_task_responses.append(response)
            conv_info = f"========== Conversation {i + 1}/{len(attacker.task)} =========="
            bar = "=" * len(conv_info)
            print(f"\n\n{bar}\n{conv_info}\n{bar}")
            print(f"\n>> USER: \n{injected_prompt}\n\n   ---   \n\n>> AI: \n{response}\n")
            print(f"{bar}\n{bar}\n{bar}\n")
            if i % 2 == 0:
                time.sleep(1)
        np.savez(injected_task_responses_path, data=injected_task_responses)

    # Attack — verbose=1 ile __nice_print tam prompt'u yazdırır, print_conversation kaldırıldı
    attack_responses_path = f'{args.save_path}/attack_responses.npz'
    if args.recompute_attack and os.path.exists(attack_responses_path):
        os.remove(attack_responses_path)
    try:
        attack_responses = np.load(attack_responses_path, allow_pickle=True)['data']
    except:
        attack_responses = []
        for i, (data_prompt, ground_truth_label) in enumerate(target_app):
            try:
                data_prompt_after_attack = attacker.inject(data_prompt, i, target_task.task)  # type: ignore
            except TypeError:
                data_prompt_after_attack = attacker.inject(data_prompt, i)  # type: ignore
            response = target_app.query(
                data_prompt_after_attack,
                injected_task=injected_dataset_name,
                verbose=args.verbose_attack,
                idx=i,
                total=len(target_app)
            )
            attack_responses.append(response)
            if i % 2 == 0:
                time.sleep(1)
        np.savez(attack_responses_path, data=attack_responses)

    # Evaluation
    evaluator = PI.create_evaluator(
        target_task_responses=target_task_responses,
        target_task=target_task,
        injected_task_responses=injected_task_responses,
        injected_task=attacker.task,
        attack_responses=attack_responses
    )

    print(f"TSR = {evaluator.tsr}")
    print(f"IRR = {evaluator.irr}")
    print(f"ASR = {evaluator.asr}")
    print(f"MR = {evaluator.mr}\n")
    print('[END]')