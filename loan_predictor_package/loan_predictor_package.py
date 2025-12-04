# ==============================================================================
# 檔案: loan_predictor_package.py
# 說明: 核心模型定義庫
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import fsspec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ==================================================================================
# 自定義 Scikit-learn Transformer :

# 必要繼承 : (為了確保此類別能完全兼容 Scikit-learn Pipeline)
# - BaseEstimator
# - TransformerMixin

# 必要實作介面 :
# - fit : 學習/定義數據的規則, 統計量或模式, 將知識儲存到物件的內部狀態中 (學習、訓練、儲存狀態)
# - transform : 應用在 fit 階段學到的規則，將輸入數據轉換成模型可用的格式 (應用、轉換、固定規則) 
# ==================================================================================

# 特徵工程 (FeatureEngineer)
# - 資料清理 (不必要欄位移除, 空值列移除) 
# - 資料標準化處理 (時間, 字串)
# - Label Encoding (標籤編碼)
# - One-Hot Encoding (獨熱編碼)
class FeatureEngineer(BaseEstimator, TransformerMixin):

    # 定義不必要的欄位    
    COLUMNS_TO_DROP = [
        'loan_status', # 貸款償還結果
        'issue_d',     # 貸款發放的月份
        'grade',       # 貸款信用等級 (因為信用子等級[sub_grade]夠詳細, 所以不需此欄位了)
        'emp_title',   # 貸款人的工作職位
        'title',       # 貸款目的標題
        'purpose',     # 貸款目的
        'address',     # 貸款人地址
        'initial_list_status', # 放款平台
    ]

    # 初始化 LabelEncoder (用於 sub_grade) 及用於儲存最終特徵順序的變數
    def __init__(self):
        self.le_subgrade = LabelEncoder()
        self.subgrade_classes_ = None
        self.final_columns_ = None

    # 特徵工程主程式
    # is_training : True 代表訓練階段 
    # is_training : False 代表應用階段
    def _main_engineer_proc(self, df, is_training=False):
        # 1) 處理時間特徵, 只取到年
        if 'earliest_cr_line' in df.columns:
            df['earliest_cr_line_year'] = df['earliest_cr_line'].apply(lambda date: int(str(date)[-4:]))
            df = df.drop('earliest_cr_line', axis=1)

        # 2) 標準化字串欄位, 處理工作年限的特定值
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.replace(' ', '_').str.replace('+', '_plus').str.replace('<', 'less_than').str.replace('[', '').str.replace(']', '')
            if col == 'emp_length':
                df[col] = df[col].str.replace('10_plus_years', '10_plus_years')
                df[col] = df[col].str.replace('less_than_1_year', '0_years')

        # 3) 有序性資料, 使用 Label Encoding (標籤編碼)
        if 'sub_grade' in df.columns:
            # is_training 為 False 代表應用階段, 一定要處理新數據裡非預期的值, 因為丟進來預測的值有可能為非預期
            if not is_training and self.subgrade_classes_ is not None:
                # 處理測試集出現訓練集沒有的新 sub_grade 類別
                # 如果新的數據中出現了在訓練集中從未見過的 sub_grade 類別, 它會被強制替換為訓練集類別清單中的 [最後一個類別]
                # 原因 1 : 確保 LabelEncoder 正常執行
                # 原因 2 : 業務/數據假設：風險保守原則, 因為我們目前的排序是愈後面風險愈高
                df['sub_grade'] = df['sub_grade'].apply(lambda x: x if x in self.subgrade_classes_ else self.subgrade_classes_[-1])
            
            # 帶有底線 (_) 的屬性（如 classes_, mean_, scale_ 等）
            # 表示這些屬性是透過 fit() 方法從數據中學習或計算出來的
            # 代表已經學習了數據的狀態  
            # 所以, 只有當 FeatureEngineer.fit() 曾被成功呼叫
            # 並且 LabelEncoder 已經學會了所有的 sub_grade 類別和它們的對應順序之後
            # 我們才做執行 transform 轉換
            if hasattr(self.le_subgrade, 'classes_'):
                df['sub_grade'] = self.le_subgrade.transform(df['sub_grade'])

        # 4) 無序性資料, 使用 One-Hot Encoding (獨熱編碼)
        categorical_cols = df.select_dtypes(include=['object']).columns
        # drop_first=True 是為了避免多重共線性 (Multicollinearity) 的問題 (例如: 已知其他項為0, 則此項必為1)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df        

    # 特徵工程的學習階段
    def fit(self, X, y=None):
        X_temp = X.copy()
        
        # 1) 資料清理 (包括去除不必要欄位及空值列)
        drop_cols = [c for c in self.COLUMNS_TO_DROP if c in X_temp.columns]
        X_temp = X_temp.drop(drop_cols, axis=1, errors='ignore') # errors='ignore' 處理欄位可能已被移除
        X_temp = X_temp.dropna() # 移除有空值欄位的列

        # 2) 學習 sub_grade 的類別順序
        if 'sub_grade' in X_temp.columns:
            unique_subgrades = sorted(X_temp['sub_grade'].astype(str).unique())
            self.le_subgrade.fit(unique_subgrades)
            self.subgrade_classes_ = self.le_subgrade.classes_
            
        # 3) 執行 特徵工程主程式
        X_processed = self._main_engineer_proc(X_temp, is_training=True)
        self.final_columns_ = X_processed.columns.tolist()
        return self

    # 執行特徵轉換, 將清理過的原始數據轉換成模型可以使用的數值特徵矩陣
    def transform(self, X):
        X_temp = X.copy()

        # 執行數據清理, 移除預設要 drop 的欄位, 以及有缺失值欄位的列資料
        drop_cols = [c for c in self.COLUMNS_TO_DROP if c in X_temp.columns]
        X_temp = X_temp.drop(drop_cols, axis=1, errors='ignore') # 移除要 drop 的欄位
        X_temp = X_temp.dropna() # 移除有空值欄位的列
        
        # 呼叫 _main_engineer_proc 進行轉換 (原 FeatureEngineer.transform 邏輯)
        X_processed = self._main_engineer_proc(X_temp, is_training=False)
        
        # 確保輸出數據的欄位順序與訓練集一致 (對於缺失的特徵欄位補 0)
        if self.final_columns_ is not None:
            for col in self.final_columns_:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            # 特徵對齊, 強制確保新數據集的特徵欄位, 與模型訓練時所使用的欄位清單和順序完全一致
            X_processed = X_processed[self.final_columns_] 
        return X_processed

# ======================================================================
# 貸款還款預測器 : 整合數據管道、訓練、預測、閾值計算及模型解釋性 (SHAP) 的核心類別
# ======================================================================
class LoanPredictor:
    
    # custom_paths : 可覆寫資料路徑來源
    def __init__(self, gcs_bucket_name='loan_predictor_bucket', custom_paths=None):
        # 建立 pipeline (FeatureEngineer)
        self.pipeline = Pipeline([
            ('feature_engineer', FeatureEngineer())
        ])
        self.model = None # 機器模型
        self.optimal_threshold = 0.5 # 預設決策閾值
        self.best_n_estimators = 100 # 預設早停條件, 值太大容易造成過擬合      
        self.gcs_bucket_name = gcs_bucket_name
        
        base_url = f'gs://{self.gcs_bucket_name}'
        
        default_paths = {
            'train_data': f'{base_url}/data/lending_data.csv', # 原始訓練資料
            'pipeline_pkl': f'{base_url}/models/lending_pipeline.pkl', # 特徵工程 pipeline
            'model_json': f'{base_url}/models/lending_model.json', # 機器模型 (json)
            'model_pkl': f'{base_url}/models/lending_model.pkl', # 機器模型 (pkl)
            'best_estimators': f'{base_url}/models/lending_best_estimators.json', # 早停值
            'threshold': f'{base_url}/models/lending_optimal_threshold.json', # 決策閾值          
            'feature_cols': f'{base_url}/models/lending_feautre_cols.json', # 最終特徵欄位
            'shap_summary': f'{base_url}/models/lending_shap_summary.json', # Top 5 SHAP          
            'test_data': f'{base_url}/data/realTestData.csv', # 新資料(測試用)
            'report': f'{base_url}/report/lending_report.doc' # 訓練結果報表
        }

        # 使用 custom_paths 覆寫預設路徑
        self.paths = default_paths.copy()
        if custom_paths:
            self.paths.update(custom_paths)
        # -------------------------------------

        # SHAP 中文對照表
        self.feature_glossary = {
            'loan_amnt': '貸款金額',
            'term': '貸款期限 (月)',
            'int_rate': '貸款利率',
            'installment': '月付款',
            'sub_grade': '次級信用評級',
            'emp_length': '工作年限',         
            'annual_inc': '年收入',
            'dti': '債務收入比',
            'open_acc': '開放信用額度數',
            'pub_rec': '公共紀錄數',
            'revol_bal': '循環信貸餘額',
            'revol_util': '循環信貸使用率',
            'total_acc': '總信用帳戶數量',
            'mort_acc': '房貸帳戶數',
            'pub_rec_bankruptcies': '公共破產紀錄數',
            'earliest_cr_line_year': '最早信用紀錄年份'
        }

    # 載入特徵工程 pipeline (靜態方法) : 從指定的路徑載入完整的 LoanPredictor instance
    def load_pipeline(self, path):
        try:
            with fsspec.open(path, 'rb') as f:
                return joblib.load(f) # 回傳一個完整的 LoanPredictor Instance
        except Exception as e:
            print(f"無法載入 Pipeline: {e}")
            raise

    # 執行模型預測
    def predict(self, data, return_proba=False):
        if self.model is None:
            raise Exception("模型尚未載入！請先使用 load_pipeline 載入模型。")
        X_processed = self.pipeline.transform(data)
        probs = self.model.predict_proba(X_processed)[:, 1]
        predictions = (probs >= self.optimal_threshold).astype(int)
        if return_proba:
            return predictions, probs
        return predictions

    # 生成預測報告
    def generate_report(self, data, save_to_gcs=False):
        preds, probs = self.predict(data, return_proba=True)
        results = data.copy()
        results['Predicted_Probability'] = probs
        results['Predicted_Status'] = np.where(preds == 1, '會還款 (低風險)', '會違約 (高風險)')
        
        report_lines = [
            f"貸款違約預測報告 (Pipeline + SHAP Version)",
            f"報告生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"決策閾值: {self.optimal_threshold:.4f}",
            "="*80, "\n"
        ]
        
        for index, row in results.iterrows():
            report_lines.append(f"客戶 ID: {index}")
            report_lines.append(f"* 貸款金額: {row.get('loan_amnt', 'N/A')}")
            report_lines.append(f"* 預測結果: {row['Predicted_Status']} (機率: {row['Predicted_Probability']:.4f})")
            report_lines.append("-" * 40)

        final_report = "\n".join(report_lines)
        if save_to_gcs:
            self._save_to_gcs(final_report, self.paths['report'], mode='w')
        return results