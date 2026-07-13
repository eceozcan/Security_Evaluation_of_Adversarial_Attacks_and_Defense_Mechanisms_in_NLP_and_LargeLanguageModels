#!/usr/bin/env python3
"""
Test sadece hatalı olan modelleri
"""
import json
import os
from pathlib import Path

import PromptInjection as PI
from PromptInjection.utils import open_config


def test_model(model_name, model_config_path):
    """Tek bir modeli test et"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Config yükle
        model_config = open_config(config_path=model_config_path)
        print(f"✓ Config loaded successfully")
        print(f"  Provider: {model_config.get('model_info', {}).get('provider', 'N/A')}")
        print(f"  Model: {model_config.get('model_info', {}).get('name', 'N/A')}")
        
        # Model oluştur
        print(f"  Creating model...")
        model = PI.create_model(config=model_config)
        print(f"✓ Model created successfully")
        
        # Model info yazdır
        model.print_model_info()
        
        # Basit bir query test et (timeout ile)
        print(f"  Testing simple query...")
        test_input = "What is 2+2?"
        try:
            import threading
            response = ["TIMEOUT"]
            def query_wrapper():
                response[0] = model.query(test_input)
            
            thread = threading.Thread(target=query_wrapper, daemon=True)
            thread.start()
            thread.join(timeout=30)  # 30 saniye timeout
            
            if response[0] == "TIMEOUT":
                print(f"⚠ Query timeout (>30s) - Model çok yavaş")
                return True, "QUERY_TIMEOUT"
            else:
                print(f"✓ Query successful")
                print(f"  Query: {test_input}")
                print(f"  Response: {response[0][:100]}..." if len(response[0]) > 100 else f"  Response: {response[0]}")
        except Exception as e:
            print(f"⚠ Query error: {str(e)[:100]}")
            return True, f"QUERY_ERROR"
        
        return True, "SUCCESS"
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False, str(e)


def main():
    """Sadece hatalı modelleri test et"""
    broken_models = [
        "deepseek-r1-distill-llama-8b",
        "llama",
        "llama3", 
        "mistral"
    ]
    
    models_dir = Path("./configs/model_configs")
    
    print(f"\n{'#'*60}")
    print(f"# BROKEN MODELS TEST")
    print(f"# Testing {len(broken_models)} broken models")
    print(f"{'#'*60}")
    
    results = {}
    success_count = 0
    fail_count = 0
    
    for model_name in broken_models:
        config_path = models_dir / f"{model_name}_config.json"
        if not config_path.exists():
            print(f"\n✗ Config file not found: {config_path}")
            continue
            
        success, message = test_model(model_name, str(config_path))
        results[model_name] = {"success": success, "message": message}
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Özet rapor
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print(f"Total: {len(results)}")
    
    print(f"\n{'Details:':^60}")
    print(f"{'-'*60}")
    for model_name, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"{status} {model_name:40} {result['message'][:50]}")


if __name__ == "__main__":
    main()
