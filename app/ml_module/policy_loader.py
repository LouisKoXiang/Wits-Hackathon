from typing import Dict, Any

# ---------------------------------------------------------
#  銀行政策定義
#  - 可依銀行（bank_id）擴充
#  - 包含：
#      emphasis（本行重視的風控面）
#      risk_strategies（依 LOW/MEDIUM/HIGH 給出不同建議）
# ---------------------------------------------------------

_BANK_POLICIES: Dict[str, Dict[str, Any]] = {
    "DEFAULT": {
        "bank_id": "DEFAULT",
        "bank_name": "Wits Bank",
        # 本行特別重視的評估觀點（可顯示於 UI 或報告）
        "emphasis": {
            "summary": "本行特別重視債務收入比（DTI）、年收入及信用歷史長度。",
            "key_metrics": [
                {
                    "metric": "dti",
                    "name_cn": "債務收入比",
                    "preferred_range": "<= 35%",
                    "description": "DTI 越低代表債務壓力較小。",
                },
                {
                    "metric": "annual_inc",
                    "name_cn": "年收入",
                    "preferred_range": ">= 600,000 NTD",
                    "description": "收入水準越高，還款能力通常越佳。",
                },
                {
                    "metric": "earliest_cr_line_year",
                    "name_cn": "最早信用年份",
                    "preferred_range": "<= 2012",
                    "description": "信用歷史越久，越容易評估其穩定度。",
                },
            ],
        },

        # 分三種風險等級給予不同策略
        "risk_strategies": {
            "HIGH": {
                "summary": "客戶屬於高風險，建議提高授信門檻或調整貸款條件。",
                "suggestions": [
                    "優先償還高利率債務，將 DTI 降至 35% 以下。",
                    "可考慮降低本次申請的貸款金額或縮短貸款年限。",
                    "維持良好還款紀錄 6–12 個月後再重新申請。",
                ],
            },
            "MEDIUM": {
                "summary": "客戶屬於中度風險，建議維持標準授信條件。",
                "suggestions": [
                    "可維持一般授信條件，持續觀察其還款能力。",
                    "建議使用較低的信貸額度以降低風險。",
                ],
            },
            "LOW": {
                "summary": "客戶屬於低風險，可提供較優惠的貸款條件。",
                "suggestions": [
                    "可提供較優惠利率，提高客戶黏著度。",
                    "可主動推薦信用卡、循環額度等其他產品。",
                    "可提供預先核准額度，方便客戶快速申貸。",
                ],
            },
        },
    }
}

#  Public API：依 bank_id 取得該銀行之政策設定
def load_policy(bank_id: str) -> Dict[str, Any]:
    """
    預設DEFAULT。
    """

    if not bank_id:
        return _BANK_POLICIES["DEFAULT"]

    bank_id = bank_id.upper()
    return _BANK_POLICIES.get(bank_id, _BANK_POLICIES["DEFAULT"])
