# QuSpike-PQK — LaTeX 編譯設定（latexmk）
#
# 輔助檔 → build/；PDF 留在 Paper/ 根目錄。
#   QuSpike-PQK.tex     （英文，IEEEtran，pdflatex）
#   QuSpike-PQK-cn.tex  （中文，article，xelatex）
$aux_dir = 'build';

mkdir 'build' unless -d 'build';
