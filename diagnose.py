
import os
import sys
import pandas as pd
import numpy as np
from main import run_pipeline

def run_diagnostic():
    # Run on pharmacy_daily as it's a standard retail dataset
    dataset = "pharmacy_daily"
    print(f"Running diagnostic on {dataset} with Tuning ENABLED...")
    
    try:
        # Enable tuning to reach >0.8 R2
        results = run_pipeline(dataset, forecast_periods=30, mode="quick", tune=True)
        print("\nDIAGNOSTIC RESULTS:")
        for m in results['metrics']:
            r2 = m['R2']
            status = "PASSED (>0.8)" if r2 > 0.8 else "FAILED (<0.8)"
            print(f"Model: {m['Model']}, R2: {r2}, RMSE: {m['RMSE']} -> {status}")
            
        best = min(results['metrics'], key=lambda x: x['RMSE'])
        if best['R2'] > 0.8:
            print(f"\n✅ SUCCESS: Best model ({best['Model']}) reached R2 of {best['R2']}")
        else:
            print(f"\n❌ FAILURE: Best model only reached R2 of {best['R2']}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostic()
