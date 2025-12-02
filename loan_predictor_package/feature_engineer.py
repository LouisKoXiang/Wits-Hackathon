# ==============================================================================
# FeatureEngineer
# - 清理欄位
# - 處理字串格式
# - Label Encoding (sub_grade)
# - One-Hot Encoding
# - 特徵欄位對齊
# ==============================================================================

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

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
