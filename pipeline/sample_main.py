import pandas as pd
import numpy as np
import os
import joblib
from loan_predictor_package import LoanPredictor 

if __name__ == "__main__":
    
    # 1. 初始化 predictor, 參數 custom_paths 可自訂義檔案來源路徑, 請參考 loan_predictor.py
    predictor = LoanPredictor()
    
    # 2. 準備測資, 丟棄目標標籤欄位(如果有的話)
    test_data = pd.read_csv(predictor.paths['test_data'])
    all_sample_data = test_data.drop(columns=['loan_repaid', 'loan_status'], errors='ignore')

    # 3. 載入特徵工程 pipeline 
    predictor = predictor.load_pipeline(predictor.paths["pipeline_pkl"])
       
    # 4. 預測結果
    # ** 直接呼叫 predictor.predict 即可
    # ** 我這裡用 generate_report, 裡面也是呼叫 predict
    results = predictor.generate_report(all_sample_data, save_to_gcs=False)
         
    # 5. 輸出違約/還款的數量分佈
    print(f"\n>> 載入驗證成功！共對 {len(all_sample_data)} 筆資料進行了預測")
    print("\n>> 預測結果 DataFrame：")
    print(results[['Predicted_Probability', 'Predicted_Status']])
