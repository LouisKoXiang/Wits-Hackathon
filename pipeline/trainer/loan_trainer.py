# =============================
# 檔案: loan_trainer.py
# 說明: 專門用於執行模型訓練的程式碼
# =============================

# 訓練檔案需要知道並匯入這兩個類別
from loan_predictor_package import LoanPredictor # 核心介面 (包含 __init__ 和 FeatureEngineer)
from loan_trainer_package import LoanTrainer # 訓練、儲存、SHAP 邏輯

class LoanPredictorTrainer(LoanPredictor, LoanTrainer):
    '''
    結合核心 LoanPredictor 介面和所有訓練邏輯，專門用於訓練環境 
    備註：這個類別不需要重新定義 __init__，它會繼承 LoanPredictor 的初始化
    '''
    
    # 這裡可以定義一個專門的公開方法來啟動訓練
    def run_training(self, data_source=None):
        # 調用繼承自 LoanTrainer 的私有方法, mode = 'debug' 加快訓練速度
        self._fit(data_source, mode='release')
        
        # 執行儲存 (現在 _save_pipeline() 
        '''
         查看 _save_pipeline, 它是使用 LoanPredictor 實例儲存 pipeline
         如此第三方只要拿到 pipeline.pkl 及 loan_predictor_package 就可以執行了.
        '''
        self._save_pipeline()  

        print('==完成訓練==')

if __name__ == '__main__':
    trainer = LoanPredictorTrainer() # 透過多重繼承，建立一個功能完整的訓練實例
    trainer.run_training()