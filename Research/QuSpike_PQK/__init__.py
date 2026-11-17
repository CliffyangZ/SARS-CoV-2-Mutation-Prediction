"""QuSpike-PQK：投影量子核（PQK）用於 Spike 譜系分類。

設計見 Doc/QuSpike-PQK.md 與 specs/007-pqk-pennylane-ibm/。
與 QuSpike_JEPA 的關鍵差異：量子部分**不訓練線路參數**，只作為固定特徵圖，
因此量子特徵與種子無關，可快取重用（見 specs/.../research.md D2）。
"""
