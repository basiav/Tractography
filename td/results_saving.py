import pandas as pd
import os

def save_results(results_dict, filename="results_survival.csv"):
    if 'covariates' in results_dict and isinstance(results_dict['covariates'], (list, pd.Index)):
        results_dict['covariates'] = ", ".join(map(str, results_dict['covariates']))
        
    df_new = pd.DataFrame([results_dict])
    
    if os.path.isfile(filename):
        df_new.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_new.to_csv(filename, mode='w', header=True, index=False)