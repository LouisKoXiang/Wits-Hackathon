# ===========================================
# 檔案: loan_trainer_package.py
# 說明: 負責模型訓練、儲存、SHAP解釋、封裝 pipeline
# ===========================================

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import fsspec
import shap
import matplotlib.pyplot as plt # 繪圖
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score # 模型評估
from loan_predictor_package import LoanPredictor # 為了利用 LoanPredictor 製作 pipeline

class LoanTrainer:
    def _save_to_gcs(self, content, gcs_path, mode='w', is_json=False):
        try:
            with fsspec.open(gcs_path, mode) as f:
                if is_json:
                    json.dump(content, f, ensure_ascii=False, indent=4) # 4 個空格作為縮排的單位
                else:
                    f.write(content)
            print(f">> GCS 儲存成功: {gcs_path}")
        except Exception as e:
            print(f">> GCS 儲存失敗 ({gcs_path}): {e}")
    
    # 特徵翻譯輔助方法 : 根據預設的對照表, 將英文特徵名稱轉換為易讀的中文, 主要用於 SHAP 報告  
    def _get_chinese_feature_name(self, feature_en):
        base_feature = feature_en.split('_')[0]
        if feature_en.startswith('term_'):
            return f"{self.feature_glossary.get('term', '貸款期限')} ({feature_en.replace('term_', '')} 個月)"
        elif feature_en.startswith('home_ownership_'):
            return f"{self.feature_glossary.get('home_ownership', '居住狀態')} ({feature_en.replace('home_ownership_', '')})"
        elif feature_en.startswith('emp_length_'):
            return f"{self.feature_glossary.get('emp_length', '工作年限')} ({feature_en.replace('emp_length_', '')})"
        else:
            return self.feature_glossary.get(feature_en, feature_en)

    # SHAP 解釋性分析 : 計算模型在 測試數據(X_test) 上的 SHAP 值, 並將 Top 5 重要特徵的摘要 (包含中文翻譯) 儲存至檔案
    def _generate_shap_summary(self, X_test_processed):
        print("\n>> 正在計算 SHAP 可解釋性摘要...")
        try:
            # 從完整的測試集 (X_test_processed) 中抽取最多 5,000 個樣本來進行 SHAP 分析
            # 將抽樣後的數據轉換為 SHAP 所需的 NumPy 陣列 (X_sample_np), 並確保數據類型為浮點數 (float32)
            max_sample_size = 10000
            if self.mode == 'debug':
                max_sample_size = 100
            sample_size = min(max_sample_size, len(X_test_processed))           
            X_sample = X_test_processed.sample(n=sample_size, random_state=101)
            X_sample_np = X_sample.values.astype(np.float32)
            
            # 建立 SHAP 解釋器 (Explainer)
            # 使用的是模型對正類（還款成功）的機率進行解釋 [:, 1] index 0 是違約機率, index 1 是還款機率
            model_predict_proba_func = lambda X: self.model.predict_proba(X)[:, 1]
            explainer = shap.Explainer(model_predict_proba_func, X_sample_np)
            
            # 取 Top 5 SHAP 值, 並儲存
            topNum = 5
            shap_result = explainer(X_sample_np)
            vals = shap_result.values
            
            feature_names = X_sample.columns.tolist()
            shap_abs_mean = np.abs(vals).mean(axis=0)
            shap_importances = pd.Series(shap_abs_mean, index=feature_names)
            top_shap_df = shap_importances.sort_values(ascending=False).head(topNum).reset_index()
            top_shap_df.columns = ['Feature_EN', 'Mean_Absolute_SHAP_Value']
            top_shap_df['Feature_CN'] = top_shap_df['Feature_EN'].apply(self._get_chinese_feature_name)
            top_shap_output = top_shap_df.to_dict('records')
            
            # 繪圖
            print("\n[1] SHAP 摘要圖 (Beeswarm Plot)")
            shap.summary_plot(shap_result, X_sample, max_display=15, show=False)
            plt.gcf().set_size_inches(10, 6)
            plt.show()

            print("\n[2] SHAP 條形圖 (Bar Plot)")
            shap.plots.bar(shap_result.abs.mean(0), max_display=15, show=False)
            plt.gcf().set_size_inches(10, 6)
            plt.show()
            
            self._save_to_gcs(top_shap_output, self.paths['shap_summary'], is_json=True)
            print(f">> Top 5 SHAP 特徵摘要已計算並上傳。")
            
        except Exception as e:
            print(f">> SHAP 計算失敗 (跳過此步驟): {e}")
            import traceback
            traceback.print_exc()

    # 執行完整訓練流程 : 使用 self.pipeline.fit_transform 進行數據預處理, 並訓練模型, 計算閾值
    def _fit(self, data_source=None, mode=None):
        # 0) 訓練模式 (設定 'debug' 用最簡單參數快速執行, 以節省等待時間)
        self.mode = mode
        print(f">> Step 0: {mode}模式")   

        '''
        第一步驟 : 資料清理與特徵工程
        '''               
        print("\n開始訓練模型 : ")    
        # 1) 訓練資料載入
        print(">> Step 1: 訓練資料載入...")
        path = data_source if data_source else self.paths['train_data']
        try:
            df = pd.read_csv(path)
            print(f">> 資料載入成功: {path}")
        except Exception as e:
            print(f">> 資料載入失敗: {e}")
            return

        # 2) 目標標籤 處理
        print(">> Step 2: 目標標籤 處理")
        df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

        # 3) 資料清理與特徵工程 (使用 Pipeline 統一執行)
        print(">> Step 3 : 資料清理與特徵工程 (使用 Pipeline 統一執行)...")      
        # 分離原始特徵和標籤
        X_all_raw = df.drop('loan_repaid', axis=1) # 原始特徵
        y_all_raw = df['loan_repaid']              # 原始目標標籤
        
        # 使用 Pipeline.fit_transform 統一執行所有預處理步驟
        X_processed = self.pipeline.fit_transform(X_all_raw, y_all_raw)
        
        '''
         * 重要 : 開始同步 特徵矩陣 X_all_raw 和標籤向量 y_all_raw 的樣本
         * 以解決在 Pipeline 內部執行 dropna() 導致 X_all_raw, y_all_raw 行數不匹配問題
        '''
        # 獲取 FeatureEngineer 執行 dropna() 後留下的原始索引，並應用於 y_all
        # 為了解決 FeatureEngineer 內部 dropna() 導致 X 和 y 樣本數不一致的問題
        fe = self.pipeline.named_steps['feature_engineer']
        
        # 執行 FeatureEngineer 內部的清理邏輯 (FeatureEngineer.transform 的前兩步)，以取得正確的索引
        X_temp_check = X_all_raw.copy()
        
        # 移除要 drop 的欄位
        drop_cols = [c for c in fe.COLUMNS_TO_DROP if c in X_temp_check.columns]
        X_temp_check = X_temp_check.drop(drop_cols, axis=1, errors='ignore') 
        
        # 執行 dropna(), 取得被保留的行的原始索引
        final_index = X_temp_check.dropna().index 
        
        # 使用這個索引從原始標籤中選取對應的值，並重設索引 (讓它從 0 到 N-1)
        # 這樣 y_all 就與 X_processed (它也是一個新 DataFrame, 索引從 0 開始) 完全同步
        y_all = y_all_raw.loc[final_index].reset_index(drop=True)
        
        if len(X_processed) != len(y_all):
            raise ValueError(f"特徵工程後 X 和 y 樣本數不一致：X={len(X_processed)}, Y={len(y_all)}")
            
        print(f">> 特徵工程完成後樣本數: {len(X_processed)}")    
        '''
         * 重要 : 結束同步 特徵矩陣 X_all_raw 和標籤向量 y_all_raw 的樣本
        '''                   

        # 4) 儲存最終特徵欄位 (從 Pipeline 內部取出 FeatureEngineer 學習到的欄位)
        print(">> Step 4 : 儲存最終特徵欄位 (從 Pipeline 內部取出 FeatureEngineer 學習到的欄位)")      
        self._save_to_gcs(self.pipeline.named_steps['feature_engineer'].final_columns_, self.paths['feature_cols'], is_json=True)

        '''
        第二步驟 : 資料切割(訓練集(Train), 驗證集(Validation), 測試集(Test)), 透過訓練集(Train)和驗證集(Validation) 找出模型訓練的終極超參數
        '''
        # 5) 訓練資料分成 訓練(Train)(80%), 驗證(Validation)(10%), 測試(Test)(10%)
        print(">> Step 5 : 訓練資料分成 訓練(Train)(80%), 驗證(Validation)(10%), 測試(Test)(10%)")           
        # test_size=0.1 : 指定將數據的 10% 用作最終的測試集
        # random_state=101(任一常數) : 設定隨機種子, 確保每次運行程式時, 劃分結果都是固定且可重現的
        # stratify=y_all : 執行分層抽樣
        #  這確保了劃分出來的 y_temp 和 y_test 中的標籤(還款 1/0)比例, 與原始數據 y_all 中的比例保持一致
        #  這對於處理 [類別不平衡問題] 非常重要
        #  類別不平衡問題(Class Imbalance Problem)是指在一個分類任務的數據集中, 不同類別的樣本數量存在極大的差異
        #  容易產生模型偏見 (Model Bias)
        # X_test(特徵), y_test(標籤) : 獲得了 10% 的測試集
        # X_temp(特徵), y_temp(標籤) : 剩餘的 90% 數據被暫存為臨時集合, 用於後續劃分為訓練和驗證集
        X_temp, X_test, y_temp, y_test = train_test_split(X_processed, y_all, test_size=0.1, random_state=101, stratify=y_all)
        
        # 將剩餘數據劃分成訓練集和驗證集
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, train_size=(0.8 / 0.9), random_state=101, stratify=y_temp)
        
        '''
        至此分出資料 : 
        訓練集 (X_train, y_train) 80% : 學習模型的參數和權重
        驗證集 (X_val, y_val) 10%     : 在訓練過程中調校超參數 (例如 XGBoost 的 early_stopping_rounds), 但不直接影響模型權重
        測試集 (X_test, y_test)	10%	  : 訓練完成後，用於最終評估模型的泛化能力(計算 AUC、最佳閾值等), 在訓練期間和調參期間應保持完全獨立
        '''

        # 6) 模型訓練
        print(">> Step 6 : 模型訓練...")
        
        # 6.1) 單調性約束(Monotone Constraints) 設定
        # 這在金融、風險評估等需要高可解釋性的業務場景中至關重要
        print(">>>> Step 6.1 : 單調性約束(Monotone Constraints) 設定")
        '''
        單調性約束的目的是強制模型在訓練時, 讓某些特徵的變化對目標變數的影響保持在特定的方向(正相關或負相關), 
        即使數據本身的雜訊可能試圖推翻這個方向
        防止訓練數據的不確實造成機器學習的偏差, 所以要人為介入, 有點像是父母或老師在盯著孩子的重點關鍵步驟一樣
        這在金融、風險評估等需要高可解釋性的業務場景中至關重要
         1 : 正相關 
        -1 : 負相關
        '''
        mono_constraints = {
            'loan_amnt': -1, # 貸款金額
            'annual_inc': 1, # 年收入
            'dti': -1, # 債務收入比
            'int_rate': -1, # 貸款利率
            'sub_grade': -1, # 次級信用評級
            'pub_rec_bankruptcies': -1 # 公共破產紀錄數
        }
               
        # 6.2) 類別不平衡 (Class Imbalance) 設定
        # 正樣本權重縮放比例, 模型會被迫更專注於從少數類樣本中學習特徵, 從而改善它對少數類（違約）的識別能力, 平衡兩類別的錯誤率
        print(">>>> Step 6.2 : 類別不平衡 (Class Imbalance) 設定")
        '''
        scale_pos_weight = 負樣本數(0) / 正樣本數(1)
        FN, False-Negative (將一個會成功還款 [正樣本] 的人錯誤地預測為違約) 的懲罰(Loss) 乘以 scale_pos_weight 倍
        FP, False-Positive (將一個會違約 [負樣本] 的人錯誤地預測為會成功還款 的懲罰(Loss) 乘以 1 倍
        這是在告訴模型, 預測錯少數類 比 預測錯多數類 更嚴重
        模型會被迫更專注於從少數類樣本中學習特徵, 從而改善它對少數類（違約）的識別能力, 平衡兩類別的錯誤率
        '''
        neg_count = y_train.value_counts()[0]
        pos_count = y_train.value_counts()[1]
        scale_pos_weight = neg_count / pos_count
        
        # 6.3) 網格搜索 (Grid Search) / 分層 K-Fold (StratifiedKFold) 設定
        # 我們採用網格搜索(Grid Search) 作為參數搜索策略，以窮舉測試所有參數組合.
        # 在評估這些組合的性能時，我們使用分層 K-Fold（StratifiedKFold）作為交叉驗證方法,
        # 以確保在類別不平衡的數據集上, 性能評估結果的穩定性和可靠性
        print(">>>> Step 6.3 : 網格搜索 (Grid Search) / 分層 K-Fold (StratifiedKFold) 設定")  
        
        # 6.3.1) 初始化 XGBoost 分類器, 是為接下來的 網格搜索 (Grid Search) 做準備
        # 參數目的在控制模型的複雜度、防止過度擬合（Overfitting）以及優化訓練速度
        print(">>>>>> Step 6.3.1 : 初始化 XGBoost 分類器, 是為接下來的 網格搜索 (Grid Search) 做準備")
        '''
        objective = 'binary:logistic' : 【目標函數設定】指定這是一個二元分類任務, 模型會輸出每個樣本屬於正類 (會還款) 的機率
        n_estimators = 100 : 【初始樹的數量】設定模型一開始要訓練的決策樹數量, 這個值通常作為 網格搜索（Grid Search）或早停（Early Stopping）的起點, 允許模型後續調整
        scale_pos_weight : 【處理類別不平衡】: 用於調整少數類的權重, 它確保模型在訓練時, 會對少數類的錯誤給予更高的懲罰, 從而避免模型傾向於預測多數類
        eval_metric	= 'logloss' : 【內部評估指標】設定模型在訓練過程 (特別是交叉驗證或早停)中，用來衡量效能的指標. Log Loss (對數損失) 是二元分類中常用的指標, 它對錯誤預測的機率給予較大的懲罰
        random_state = 101 【確保可重現性】設定隨機種子. 確保每次運行程式時, 模型初始化的隨機過程 (如樹的建立、數據的隨機取樣) 都是一致的, 從而保證訓練結果的可重現性
        n_jobs = -1	【加速訓練】 : 指定模型訓練時可以使用的 CPU 核心數. 設定為 -1 意味著使用電腦所有可用的核心進行並行計算, 以最大化訓練速度
        '''
        xgb_base = xgb.XGBClassifier(
            objective='binary:logistic', 
            n_estimators=100, 
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', 
            random_state=101, 
            n_jobs=-1
        )
        
        # 6.3.2) 設定 Grid-Search (網格搜索) 超參數搜索空間（Hyperparameter Search Space）
        # 目的 :
        # 系統化搜索 : 探索所有可能的超參數組合
        # 找出最佳配置 : 目標是找到一組超參數配置, 能夠讓機器學習模型在 驗證集 (Validation Set) 上獲得最佳的性能評估分數        
        print(">>>>>> Step 6.3.2 : 設定 Grid-Search (網格搜索) 超參數搜索空間（Hyperparameter Search Space）")
        '''
        max_depth : 值越大樹越深, 模型的表達能力越強, 但越容易過度擬合 (Overfitting) 訓練數據, 反之則可能擬合不足 (Underfitting)
        learning_rate : 學習率(eta), 控制每棵樹對最終預測的貢獻大小, [0.03, 0.1] 調整模型的學習速度. 較低的值 (0.03) 學習較慢、通常更穩定但需要更多的樹 (n_estimators)；較高的值 (0.1) 學習較快
        gamma : 節點分裂所需的最小損失減少量 (懲罰項)[0, 0.1] 調整模型的保守程度. 0 表示只要有任何改善就分裂；0.1 表示需要更大的改善(更保守),有助於防止過度擬合
        min_child_weight : 子節點所需的最小樣本權重和. [1, 3] 調整模型的保守程度與過擬合控制. 值越大, 分裂越困難, 模型越保守. 這有助於平滑模型、避免在數據量小的節點上過度擬合
        reg_lambda : L2 正則化 (L2 Regularization)權重. [1, 5] 控制模型的複雜度. 這在損失函數中加入懲罰項, 使葉子權重趨於更小、更平滑. 數值越大, 模型越傾向於簡單       
        '''
        param_grid = {
            'max_depth': [4, 6], 
            'learning_rate': [0.03, 0.1],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3],
            'reg_lambda': [1, 10]
        }
          
        if self.mode == 'debug' :
            param_grid = {'max_depth': [4, 6]}
        '''
        以我們設定的 param_grid 總共有 2x2x2x2x2 = 32 種不同的參數組合
        窮舉與交叉驗證 (Exhaustive Search & Cross-Validation)
        網格搜索會對這個網格中的 每一個節點 (即每一個參數組合) 執行以下操作：
        1. 組合生成 : 從網格中選取一組特定的超參數值（例如：max_depth=4 和 learning_rate=0.03）
        2. 模型訓練 : 使用這組超參數初始化模型 (如 xgb_base)
        3. 性能評估(核心) : 為了可靠地評估性能, 會使用交叉驗證 (Cross-Validation, CV) [Stratified K-Fold Cross-Validation]
           將訓練數據分成 K 份 (例如 K=5). 進行 K 輪訓練, 每輪用 K-1 份數據訓練模型, 用剩下的 1 份數據作為驗證集進行評估
        4. 計算這 K 輪評估分數的平均值, 作為這組超參數的最終分數.
        5. 重複以上步驟, 直到網格中所有的組合都被測試完畢.
        '''
        
        # 6.3.3) 分層 K-Fold (StratifiedKFold) 設定
        print(">>>>>> Step 6.3.3 : 分層 K-Fold (StratifiedKFold) 設定")
        '''
        n_splits=3 : 將訓練集平均分成 3 個折疊(Folds). 使用 param_grid 交叉運行 3 輪, 每輪用 2/3 的數據訓練, 用剩下的 1/3 驗證
        shuffle=True : 在劃分之前先打亂數據, 增加隨機性
        random_state=101 : 確保劃分的結果是可重現的
        '''
        n_splits = 3
        if self.mode == 'debug' : 
            n_splits = 2       
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=101)
        
        # 6.3.4) 網格搜索物件 (GridSearchCV) 設定
        # GridSearchCV 物件就是驗證的裁判, 它接收了基礎模型、參數地圖、以及交叉驗證規則
        print(">>>>>> Step 6.3.4 : 網格搜索物件 (GridSearchCV) 設定")
        '''
        xgb_base : XGBoost 物件
        param_grid : Grid-Search (網格搜索參數)
        cv=skf : 分層 K-Fold 參數
        scoring='roc_auc' : 
        verbose : 1 (0代表過程不顯示, 1以上就會顯示)
        '''
        grid_search = GridSearchCV(xgb_base, param_grid, cv=skf, scoring='roc_auc', verbose=2)
        
        # 6.4) 開始 Grid-Search 搜索最佳超參數!
        print(">>>> Step 6.4 : 開始 Grid-Search 搜索最佳超參數!")
        '''
        開始搜索, 包括所有 32 次模型的交叉驗證訓練和最終最佳模型的訓練
        是整個調參過程中最耗時也最核心的一步
        在所有 32 種組合都測試完畢後, grid_search.fit() 會自動執行兩個最終動作：
        1. 選定最佳參數 : 找出那組導致平均 AUC-ROC 分數最高的超參數組合
        2. 最終擬合 : 使用這組最佳參數, 對整個輸入數據 (即完整的 X_train 和 y_train)重新訓練一個新的 XGBoost 模型.
           這個最終訓練好的模型會儲存在 grid_search.best_estimator_ 屬性中
        '''
        grid_search.fit(X_train, y_train)  

        # 6.4.1) 取得 Grid-Search 最佳參數    
        print(">>>>>> Step 6.4.1 : 取得 Grid-Search 最佳參數 ")        
        best_params = grid_search.best_params_
        print(f">> Grid Search 最佳參數: {best_params}")
        
        # 6.4.2) 定義最終參數
        print(">>>>>> Step 6.4.2 : 定義最終參數")      
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'seed': 101,
            'monotone_constraints': mono_constraints,
            **best_params # 把最佳參數導入 final_params(最終參數)
        }
        
        # 6.5) 定義 XGBoost, 專有的數據格式 : DMatrix
        print(">>>> Step 6.5 : 定義 XGBoost, 專有的數據格式 : DMatrix") 
        '''
        將 Pandas DataFrame (X_train, X_val) 和 Series (y_train, y_val) 轉換為 XGBoost 專有的數據格式 DMatrix
        '''
        D_train = xgb.DMatrix(X_train, label=y_train)
        D_val = xgb.DMatrix(X_val, label=y_val)
        
        # 6.5.1) 監控列表 訓練數據/驗證數據, 告訴 XGBoost 在訓練過程中應該監控哪些數據集和它們的名稱
        print(">>>>>> Step 6.5.1 : 監控列表 訓練數據/驗證數據, 告訴 XGBoost 在訓練過程中應該監控哪些數據集和它們的名稱") 
        watchlist = [(D_train, 'train'), (D_val, 'validation')]
        
        # 6.5.2) 執行原生XGBoost訓練! 取得最佳訓練結果 xgb_best
        print(">>>>>> Step 6.5.2 : 執行原生XGBoost訓練! 取得最佳訓練結果 xgb_best") 
        '''
        final_params : 最終參數
        D_train : (DMatrix)	【訓練數據】指定用於建構樹模型的訓練數據集 (DMatrix 為 XGBoost 內部的高效格式)
        num_boost_round : 3000【最大迭代次數】設定模型最多可以建立的樹的數量, 但實際的訓練將由 early_stopping_rounds 提前終止
        evals : watchlist【評估數據集】指定要監控的數據集列表. 在設定中包含了訓練集 (train) 和驗證集 (validation)
        early_stopping_rounds : 50【早停條件】(重要) : 這是防止過度擬合(Overfitting)的關鍵機制.
            如果在監控的驗證集 (validation) 上, 模型的性能指標 (由 eval_metric 定義，
            例如 AUC 連續 50 次迭代 (即 50 棵樹) 都沒有改善, 訓練將會提前停止
        verbose_eval : False 代表過程不顯示 
        '''
        num_boost_round = 3000
        early_stopping_rounds = 150
        if self.mode == 'debug' :
            num_boost_round = 300
            early_stopping_rounds = 10
        xgb_best = xgb.train(
            final_params,
            D_train,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True
        )
        
        # 6.5.3) 取得最佳迭代次數(決策樹的量) best_n_estimators
        print(">>>>>> Step 6.5.3 : 取得最佳迭代次數(決策樹的量) best_n_estimators") 
        '''
        防止過度擬合 : 確保最終模型只使用能夠帶來最佳泛化能力 (在驗證集上表現最好)的樹的數量, 避免使用早停點之後可能導致過度擬合的樹
        '''
        self.best_n_estimators = xgb_best.best_iteration + 1 # index 0 開始, 所以總數要 + 1
        print(f">> 最佳樹數量: {self.best_n_estimators}")

        # 6.5.4) 儲存最佳迭代次數(決策樹的量)
        print(">>>>>> Step 6.5.4 : 儲存最佳迭代次數(決策樹的量)") 
        self._save_to_gcs({'best_n_estimators': self.best_n_estimators}, self.paths['best_estimators'], is_json=True)

        # 6.6) 定義 Scikit-learn 兼容的 XGBClassifier 物件
        print(">>>> Step 6.6 : 定義 Scikit-learn 兼容的 XGBClassifier 物件") 
        '''
        將前面透過原生 XGBoost API 訓練出來的最佳模型（xgb_best）, 轉移並包裝到 Scikit-learn 風格的 xgb.XGBClassifier 物件 (self.model) 中
        這樣做的主要好處是, 讓這個模型具備 Scikit-learn 的標準接口 (例如 fit, predict, predict_proba),
        方便後續與其他 Scikit-learn 工具 (如 Pipeline 或 GridSearchCV) 整合
        
        *** 重要 :
        這種「先原生訓練，再轉移到 Sklearn Wrapper」的方法是為了實現
        1. 訓練階段的「效率和精準度」 (需要原生 API 的高效訓練和精確優化能力)
        2. 應用階段的「標準化和便利性」(Scikit-learn Wrapper 的優勢)
           讓模型支援 predict(), predict_proba(), fit(), get_params(), set_params() 等標準方法
           更重要的是可以 實現 pipeline 工程, 例如將 FeatureEngineer和最終模型綁定在一起進行序列化(Serialization)和部署的標準做法
        '''
        self.model = xgb.XGBClassifier(
            n_estimators=self.best_n_estimators,
            scale_pos_weight=scale_pos_weight,
            random_state=101,
            n_jobs=-1,
            **{k: v for k, v in final_params.items() if k not in ['objective', 'eval_metric', 'scale_pos_weight', 'seed', 'monotone_constraints']}
        )
        
        # 6.7) 儲存原生XGBoost機器學習模型 (JSON)
        print(">>>> Step 6.7 : 儲存原生XGBoost機器學習模型 (JSON)") 
        local_xgb = 'local_xgb.json'
        xgb_best.save_model(local_xgb)
        with open(local_xgb, 'rb') as f_in:
            self._save_to_gcs(f_in.read(), self.paths['model_json'], mode='wb')
        
        # 6.8) 產出可呼叫的機器學習模型 (Scikit-learn model)
        print(">>>> Step 6.8 : 產出可呼叫的機器學習模型 (Scikit-learn model)") 
        '''
        原生模型 (xgb_best) 和 Sklearn Wrapper (self.model) 無法直接進行記憶體中的數據轉移
        它們的內部數據結構和方法調用不同, 因此利用暫存檔案當中介
        '''
        self.model.load_model(local_xgb) # 此時 self.model = 可呼叫的機器學習模型 (Scikit-learn model)
        os.remove(local_xgb) # 移除暫存中介檔

        # 6.9) 儲存可呼叫的機器學習模型 (Scikit-learn model) (PKL)
        print(">>>> Step 6.9 : 儲存可呼叫的機器學習模型 (Scikit-learn model) (PKL)")
        '''
        我們的 client 要使用的是這支 pkl 模型
        '''
        local_pkl = 'local_xgb.pkl'
        joblib.dump(self.model, local_pkl)
        with open(local_pkl, 'rb') as f_in:
             self._save_to_gcs(f_in.read(), self.paths['model_pkl'], mode='wb')
        os.remove(local_pkl)
        
        '''
        第三步驟 : 將目前訓練出來的模型帶入 測試集(Test), 算出 最佳決策閾(Threshold)值, AUC
        '''
        # 7) 計算最佳決策閾(Threshold)值 (要搭配 Confusion Matrix 才能算出來)
        print(">> Step 7 : 計算最佳決策閾(Threshold)值 (要搭配 Confustion Matrix 才能算出來)...")
        '''
        這個最佳閾值是通過最大化業務利潤 (Profit) 來決定的, 是一種代價敏感 (Cost-Sensitive) 的決策方法
        透過將機率與最佳閾值 (而不是 0.5) 進行比較, 可以將模型的預測結果轉換為最終的 0/1 決策
        從而最大化業務利潤, 平衡錯誤接受 (FP) 和 和錯誤拒絕 (FN) 的成本
        self.model.predict_proba 會產生一個 NumPy 陣列 (NumPy array), 它包含了模型對測試集 X_test 中所有樣本的預測機率
        [違約機率, 還款機率]
        例 : 
        [
          [0.70, 0.30],  # 樣本 1: 70% 違約機率, 30% 還款機率
          [0.15, 0.85],  # 樣本 2: 15% 違約機率, 85% 還款機率
          [0.55, 0.45],  # 樣本 3: 55% 違約機率, 45% 還款機率
          [0.99, 0.01],  # 樣本 4: ...
          [0.02, 0.98]   # 樣本 5: ...
        ]
        ''' 
        
        # : => 所有列, index 1 => 取成功還款的機率        
        probs = self.model.predict_proba(X_test)[:, 1]
        
        # 創建一個 Numpy 陣列, 包含 101 個從 0.00 到 1.00 之間均勻分佈的值, 這些值是模型將要測試的所有決策閾值!
        thresholds = np.linspace(0, 1, 101)
        
        # 這是代價敏感分析的核心 : 為四種可能的預測結果定義其相應的業務價值
        '''
        COST_FP = -10.0 (False Positive) : 預測還款但實際違約(錯誤接受) ==> 高成本/損失
        COST_FN = -1.0  (False Negative) : 預測違約但實際還款(錯誤拒絕) ==> 低成本/機會成本(失去了本來可以賺取的利潤)
        PROFIT_TP = 1.0 (True Positive)  : 預測還款且實際還款(正確接受) ==> 收益
        BENEFIT_TN = 0.0 (True Negative) : 預測違約且實際違約(正確拒絕) ==> 無收益也無損失
        這些值如何定?
        我們的機器學習模型負責最大化基於這些業務成本和收益的 總利潤(profit)
        而這些成本和收益值則由 業務部門 和 財務部門 根據實際的產品利潤率和歷史違約損失數據來提供
        '''
        profits = []
        COST_FP = -10.0
        COST_FN = -1.0
        PROFIT_TP = 1.0
        BENEFIT_TN = 0.0
        
        # 7.1) 計算混淆矩陣 (Confusion Matrix)
        print(">>>> Step 7.1 : 計算混淆矩陣 (Confusion Matrix)...")
        y_test_np = y_test.values      
        for t in thresholds:
            # 將所有樣本的預測機率（probs）轉換為最終的 0 或 1
            preds_t = (probs >= t).astype(int)
            
            # 計算混淆矩陣 (Confusion Matrix)           
            FP = ((preds_t == 1) & (y_test_np == 0)).sum() # 預測還款但實際違約(錯誤接受)
            FN = ((preds_t == 0) & (y_test_np == 1)).sum() # 預測違約但實際還款(錯誤拒絕)
            TP = ((preds_t == 1) & (y_test_np == 1)).sum() # 預測還款且實際還款(正確接受)
            TN = ((preds_t == 0) & (y_test_np == 0)).sum() # 預測違約且實際違約(正確拒絕)

            profits.append((TP * PROFIT_TP) + (FP * COST_FP) + (FN * COST_FN) + (TN * BENEFIT_TN))
          
        # 7.2) 評估模型, 透過窮舉測試集上所有的候選決策閾值 (thresholds), 來找到能夠帶來最大總體業務利潤的那一個閾值
        print(">>>> Step 7.2 : 評估模型, 透過窮舉測試集上所有的候選決策閾值 (thresholds), 來找到能夠帶來最大總體業務利潤的那一個閾值")        
        best_profit_index = np.argmax(profits) # 最大利潤的 index
        max_profit = profits[best_profit_index] # 最大利潤    
        self.optimal_threshold = float(thresholds[best_profit_index])
        print("="*60)
        print(f">> 最佳決策閾值: {self.optimal_threshold:.4f}")  
        print(f">>  預期最大利潤分數: {max_profit:.2f}")
        print("="*60)
        self._save_to_gcs({'optimal_threshold': self.optimal_threshold}, self.paths['threshold'], is_json=True)
        
        # 7.3) 產生最終預測結果與評估指標
        print(">>>> Step 7.3 : 產生最終預測結果與評估指標")
        predictions = (probs >= self.optimal_threshold).astype(int)
        auc_score = roc_auc_score(y_test, probs)

        print("\n分類報告 (Classification Report):")
        print(classification_report(y_test, predictions))

        cm = confusion_matrix(y_test, predictions)
        print("="*60)
        print("      【混淆矩陣 (Confusion Matrix)】")
        print(f"       決策閾值: {self.optimal_threshold:.4f} ")
        print("="*60)

        cm_df = pd.DataFrame(
            cm,
            index=['實際違約 (0)', '實際還款 (1)'],
            columns=['預測違約 (0)', '預測還款 (1)']
        )
        print(cm_df)

        print(f"\n>> 模型 AUC 分數 (測試集): {auc_score:.4f} (使用 {self.best_n_estimators} 棵樹)")

        # 7.4) 繪圖 (利潤曲線與混淆矩陣)
        print(">>>> Step 7.4 : 產生最終預測結果與評估指標")
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, profits, label='Profit Curve', color='green')
        plt.axvline(self.optimal_threshold, color='red', linestyle='--', label=f'Optimal: {self.optimal_threshold:.2f}')
        plt.title('Business Profit Curve vs. Threshold (No Scale)')
        plt.xlabel('Threshold (Probability)')
        plt.ylabel('Profit Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['違約 (0)', '還款 (1)'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix (Threshold {self.optimal_threshold:.2f})')
            plt.show()
        except Exception as e:
            print(f"繪製混淆矩陣失敗: {e}")

        print("\n>> 模型評估完成。")        

        '''
        第四步驟 : 透過 SHAP 解釋模型判斷的特徵依據, 即 計算各特徵的貢獻度
        '''
        # 8) 產生 SHAP (SHapley Additive exPlanations) 分析
        # 夏普利值 (Shapley Value) 的原理來計算各特徵的貢獻度
        print(">> Step 8 : 產生 SHAP (SHapley Additive exPlanations) 分析...")
        '''
        執行 SHAP (SHapley Additive exPlanations) 分析, 以解釋模型是如何做出預測的
        SHAP 報告的模型行為, 是基於獨立、未參與訓練的數據, 所以要以 X_test 為參數, 
        這能更真實地反映模型在未來新數據上的表現
        '''
        self._generate_shap_summary(X_test)
               
        print(">> 訓練流程 / 檔案儲存 / SHAP 全部完成。")
      
    # 儲存完整實例 : 將包含預處理 Pipeline, 訓練好的 self.model 及 self.optimal_threshold 的 LoanPredictor 完整實例序列化 (Pickle) 後儲存至 GCS   
    def _save_pipeline(self):
        print(">> 開始將訓練結果轉移到部署 Pipeline 並儲存...")
        
        # 1. 確保 'self' 擁有所有訓練好的屬性 (如 self.pipeline, self.model)
        # 2. 創建一個新的、乾淨的 LoanPredictor 實例 (這是我們希望序列化的目標型別)
        deployment_predictor = LoanPredictor() 
        
        # 3. 將訓練好的狀態從 LoanPredictorTrainer (即 'self') 轉移到新的 LoanPredictor 實例
        deployment_predictor.pipeline = self.pipeline
        deployment_predictor.model = self.model
        deployment_predictor.optimal_threshold = self.optimal_threshold
        deployment_predictor.best_n_estimators = self.best_n_estimators
        # 確保路徑和字典也被複製
        deployment_predictor.paths = self.paths
        deployment_predictor.feature_glossary = self.feature_glossary
        
        # 4. 序列化這個新的 LoanPredictor 實例
        path = self.paths['pipeline_pkl']
        try:
            with fsspec.open(path, 'wb') as f:
                joblib.dump(deployment_predictor, f) # <-- 關鍵！序列化 LoanPredictor 實例
            print(f"✅ 成功將部署 Pipeline (型別: LoanPredictor) 儲存至: {path}")
        except Exception as e:
            print(f"❌ 儲存部署 Pipeline 失敗: {e}")
            raise