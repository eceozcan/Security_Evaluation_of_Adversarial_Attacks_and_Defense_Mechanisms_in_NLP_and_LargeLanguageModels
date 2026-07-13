import os
import pandas as pd
import numpy as np

def analyze_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    metrics = ["ASR", "TSR", "IRR"]
    results = {}
    
    for metric in metrics:
        baseline_col = f"baseline_{metric}"
        defense_col = f"defense_{metric}"
        
        if baseline_col not in df.columns or defense_col not in df.columns:
            continue
        
        baseline = pd.to_numeric(df[baseline_col], errors='coerce')
        defense = pd.to_numeric(df[defense_col], errors='coerce')
        
        # Remove NaN values
        valid = ~(baseline.isna() | defense.isna())
        baseline = baseline[valid]
        defense = defense[valid]
        
        if len(baseline) == 0:
            continue
        
        avg_baseline = baseline.mean()
        avg_defense = defense.mean()
        avg_reduction = avg_baseline - avg_defense
        pct_reduction = (avg_reduction / avg_baseline * 100) if avg_baseline != 0 else 0
        
        results[metric] = {
            'avg_baseline': avg_baseline,
            'avg_defense': avg_defense,
            'avg_reduction': avg_reduction,
            'pct_reduction': pct_reduction,
            'count': len(baseline)
        }
    
    return results

def main():
    result_dir = "result"
    csvs = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
    
    print("=" * 80)
    print("DEFENSE IMPACT ANALYSIS - SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for csv_file in sorted(csvs):
        model_name = os.path.splitext(csv_file)[0].upper()
        csv_path = os.path.join(result_dir, csv_file)
        
        results = analyze_csv(csv_path)
        all_results[model_name] = results
        
        print(f"\n[{model_name}]")
        print("-" * 80)
        
        for metric in ["ASR", "TSR", "IRR"]:
            if metric in results:
                r = results[metric]
                print(f"  {metric}:")
                print(f"    - Baseline (avg):  {r['avg_baseline']:.4f}")
                print(f"    - Defense (avg):   {r['avg_defense']:.4f}")
                print(f"    - Absolute reduction: {r['avg_reduction']:.4f}")
                print(f"    - Percentage reduction: {r['pct_reduction']:.2f}%")
                print(f"    - Samples: {r['count']}")
    
    print("\n" + "=" * 80)
    print("OVERALL CROSS-MODEL SUMMARY")
    print("=" * 80)
    
    for metric in ["ASR", "TSR", "IRR"]:
        print(f"\n{metric} (Average across all models):")
        baselines = []
        defenses = []
        reductions = []
        pcts = []
        
        for model, result in all_results.items():
            if metric in result:
                baselines.append(result[metric]['avg_baseline'])
                defenses.append(result[metric]['avg_defense'])
                reductions.append(result[metric]['avg_reduction'])
                pcts.append(result[metric]['pct_reduction'])
        
        if baselines:
            avg_baseline = np.mean(baselines)
            avg_defense = np.mean(defenses)
            avg_reduction = np.mean(reductions)
            avg_pct = np.mean(pcts)
            
            print(f"  - Mean baseline:        {avg_baseline:.4f}")
            print(f"  - Mean defense:         {avg_defense:.4f}")
            print(f"  - Mean reduction:       {avg_reduction:.4f} ({avg_pct:.2f}%)")
            print(f"  - Min baseline:         {min(baselines):.4f}")
            print(f"  - Max baseline:         {max(baselines):.4f}")
            print(f"  - Min defense:          {min(defenses):.4f}")
            print(f"  - Max defense:          {max(defenses):.4f}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Calculate ASR improvement
    asr_baselines = []
    asr_defenses = []
    for model, result in all_results.items():
        if "ASR" in result:
            asr_baselines.append(result["ASR"]['avg_baseline'])
            asr_defenses.append(result["ASR"]['avg_defense'])
    
    avg_asr_baseline = np.mean(asr_baselines)
    avg_asr_defense = np.mean(asr_defenses)
    asr_reduction = avg_asr_baseline - avg_asr_defense
    asr_pct = (asr_reduction / avg_asr_baseline * 100) if avg_asr_baseline != 0 else 0
    
    print(f"\n1. Attack Success Rate (ASR) Reduction:")
    print(f"   Baseline: {avg_asr_baseline:.4f} → Defense: {avg_asr_defense:.4f}")
    print(f"   Absolute reduction: {asr_reduction:.4f}")
    print(f"   Percentage reduction: {asr_pct:.2f}%")
    print(f"   Interpretation: Defenses reduce attack success by ~{asr_pct:.0f}%")
    
    # Calculate TSR impact
    tsr_baselines = []
    tsr_defenses = []
    for model, result in all_results.items():
        if "TSR" in result:
            tsr_baselines.append(result["TSR"]['avg_baseline'])
            tsr_defenses.append(result["TSR"]['avg_defense'])
    
    if tsr_baselines:
        avg_tsr_baseline = np.mean(tsr_baselines)
        avg_tsr_defense = np.mean(tsr_defenses)
        tsr_impact = avg_tsr_defense - avg_tsr_baseline  # negative = loss
        tsr_pct = (tsr_impact / avg_tsr_baseline * 100) if avg_tsr_baseline != 0 else 0
        
        print(f"\n2. Task Success Rate (TSR) Impact:")
        print(f"   Baseline: {avg_tsr_baseline:.4f} → Defense: {avg_tsr_defense:.4f}")
        print(f"   Change: {tsr_impact:.4f} ({tsr_pct:+.2f}%)")
        if abs(tsr_pct) < 5:
            print(f"   Interpretation: Minimal performance degradation (~{abs(tsr_pct):.1f}%)")
        elif tsr_pct < 0:
            print(f"   Interpretation: Moderate performance loss (~{abs(tsr_pct):.1f}%)")
        else:
            print(f"   Interpretation: Performance maintained or slightly improved")
    
    # Calculate IRR impact
    irr_baselines = []
    irr_defenses = []
    for model, result in all_results.items():
        if "IRR" in result:
            irr_baselines.append(result["IRR"]['avg_baseline'])
            irr_defenses.append(result["IRR"]['avg_defense'])
    
    if irr_baselines:
        avg_irr_baseline = np.mean(irr_baselines)
        avg_irr_defense = np.mean(irr_defenses)
        irr_impact = avg_irr_defense - avg_irr_baseline
        irr_pct = (irr_impact / avg_irr_baseline * 100) if avg_irr_baseline != 0 else 0
        
        print(f"\n3. Information Retention Rate (IRR) Impact:")
        print(f"   Baseline: {avg_irr_baseline:.4f} → Defense: {avg_irr_defense:.4f}")
        print(f"   Change: {irr_impact:.4f} ({irr_pct:+.2f}%)")
        if abs(irr_pct) < 5:
            print(f"   Interpretation: Information integrity largely preserved")
        elif irr_pct < 0:
            print(f"   Interpretation: Some information loss (~{abs(irr_pct):.1f}%)")
        else:
            print(f"   Interpretation: Information retention improved")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
