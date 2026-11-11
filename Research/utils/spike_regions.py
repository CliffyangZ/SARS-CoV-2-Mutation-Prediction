"""spike 功能區註解（胺基酸座標，FR-029）。

NTD、RBD、S1/S2、HR1 取自規格；FP 取 788–806（Xia et al. 2020, Cell Mol Immunol），
HR2 1163–1213、跨膜區 1214–1237、胞內尾 1238–1273 取自文獻慣用邊界。邊界來源列入報告假設。
"""
REGIONS = [("NTD", 14, 305), ("RBD", 319, 541), ("S1/S2", 685, 686), ("FP", 788, 806),
           ("HR1", 912, 984), ("HR2", 1163, 1213), ("TM", 1214, 1237), ("CT", 1238, 1273)]

# 圓形頻率圖的四段（FR-020）
SEGMENTS = [(1, 318), (319, 636), (637, 954), (955, 1273)]


def region_of(pos):
    """spike 位置（1 起算）→ 功能區名稱；不在任何已知功能區回傳 'other'。"""
    return next((n for n, a, b in REGIONS if a <= pos <= b), "other")


def segment_of(pos):
    """位置 → 四段之一的編號 0..3。"""
    return next(i for i, (a, b) in enumerate(SEGMENTS) if a <= pos <= b)
