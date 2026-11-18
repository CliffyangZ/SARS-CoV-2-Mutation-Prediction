# QuSpike-PQK 量子特徵映射：公式推導

本文件補充論文第 3.6 節（哈密頓量演化編碼）省略的推導。論文只給白話說明與結論，這裡逐步寫出每個等式與性質的來源。

符號與論文一致：$n$ 為量子位元數，$z\in[-\pi/2,\pi/2]^{2n}$ 為輸入，$T$ 為 Trotter 步數，$\tau$ 為總演化時間，$\Delta=\tau/T$，$X,Y,Z$ 為 Pauli 矩陣。實驗取 $T=4$、$\tau=0.8$。

目錄

1. [定義](#1-定義)
2. [閘分解：每個因子都是原生閘](#2-閘分解每個因子都是原生閘)
3. [部分交換形式](#3-部分交換形式)
4. [命題 1：磁化守恆](#4-命題-1磁化守恆)
5. [命題 2：無耦合時的乘積態](#5-命題-2無耦合時的乘積態)
6. [群組 D 的補零與解耦](#6-群組-d-的補零與解耦)
7. [Trotter 誤差](#7-trotter-誤差)
8. [頻率限制](#8-頻率限制)
9. [與程式碼對應](#9-與程式碼對應)

---

## 1. 定義

$n\ge3$ 時，量子位元排成環，邊集合 $E=\{(j,\ j \bmod n+1)\}_{j=1}^{n}$，$|E|=n$。$n=2$ 時只有一條邊 $E=\{(1,2)\}$，輸入 $z_4$ 不使用。

$$
H(z)=\sum_{j=1}^{n} z_j Z_j+\sum_{(j,j')\in E} z_{n+j}\,h_{jj'},\qquad
h_{jj'}=X_jX_{j'}+Y_jY_{j'}+Z_jZ_{j'} .
$$

前 $n$ 個輸入是各位元的局部場，後 $|E|$ 個是相鄰耦合強度。

一階 Trotter 分解把 $e^{-i\tau H}$ 切成 $T$ 步：

$$
U_T(z)=\prod_{t=1}^{T}\Bigg[\prod_{(j,j')\in E}e^{-i\Delta z_{n+j}h_{jj'}}\ \prod_{j=1}^{n}e^{-i\Delta z_jZ_j}\Bigg].
$$

初始態 $|{+}\rangle^{\otimes n}=H_{\mathrm d}^{\otimes n}|0\rangle^{\otimes n}$，$|\psi(z)\rangle=U_T(z)|{+}\rangle^{\otimes n}$。

讀出：每個位元的單量子位元約化密度矩陣（1-RDM）為

$$
\rho_k=\tfrac12\big(I+\phi_k\cdot\boldsymbol\sigma\big),\qquad
\phi_k=(\langle X_k\rangle,\langle Y_k\rangle,\langle Z_k\rangle),
$$

量子特徵 $\phi(z)=(\phi_1,\dots,\phi_n)\in[-1,1]^{3n}$。

> 這裡 $U_T$ 本身就是特徵映射的**定義**。它不需要忠實模擬 $e^{-i\tau H}$；第 7 節的誤差只衡量兩者有多接近。

## 2. 閘分解：每個因子都是原生閘

**局部場。** 約定 $R_Z(\theta)=e^{-i\theta Z/2}$，則

$$
e^{-i\Delta z_jZ_j}=R_Z(2\Delta z_j).
$$

**耦合。** 先驗證 $XX$、$YY$、$ZZ$ 兩兩對易。以 $XX$ 與 $YY$ 為例：

$$
(X\otimes X)(Y\otimes Y)=(XY)\otimes(XY)=(iZ)\otimes(iZ)=-Z\otimes Z,
$$
$$
(Y\otimes Y)(X\otimes X)=(YX)\otimes(YX)=(-iZ)\otimes(-iZ)=-Z\otimes Z .
$$

兩者相等，所以對易。其餘兩對同理。對易的算符，指數可以拆開且不產生誤差：

$$
e^{-i\Delta J h_{jj'}}=e^{-i\Delta J XX}\,e^{-i\Delta J YY}\,e^{-i\Delta J ZZ}
=R_{XX}(2\Delta J)\,R_{YY}(2\Delta J)\,R_{ZZ}(2\Delta J).
$$

這個拆分是**精確**的。誤差只來自場項與耦合項之間不對易（第 7 節）。

## 3. 部分交換形式

兩個量子位元的 SWAP 可寫成

$$
\mathrm{SWAP}=\tfrac12\,(I+XX+YY+ZZ)\ \Longrightarrow\ h=XX+YY+ZZ=2\,\mathrm{SWAP}-I .
$$

因為 $\mathrm{SWAP}^2=I$，$e^{-i\alpha h}=e^{i\alpha}e^{-2i\alpha\,\mathrm{SWAP}}$，再用 $e^{-i\theta S}=\cos\theta\, I-i\sin\theta\, S$（$S^2=I$）：

$$
e^{-i\alpha h_{jj'}}=e^{i\alpha}\Big[\cos(2\alpha)\,I-i\sin(2\alpha)\,\mathrm{SWAP}_{jj'}\Big].
$$

白話：每個耦合是「以 $\cos^2(2\alpha)$ 保持不動、以 $\sin^2(2\alpha)$ 交換兩個位元」的疊加。若兩位元的局部場不同（$z_j\ne z_{j'}$），交換與不交換會留下不同的相位，結果就是糾纏。

## 4. 命題 1：磁化守恆

**命題。** 對所有 $z$ 與 $T$，$\sum_{k=1}^{n}\langle Z_k\rangle=0$。

**證明。** 令 $M=\sum_k Z_k$。

1. $M$ 與每個 $Z_j$ 對易，因為 $Z$ 彼此對易。
2. $M$ 與每個 $h_{jj'}$ 對易。$h_{jj'}=2\,\mathrm{SWAP}_{jj'}-I$，而 $M$ 對位元的排列對稱，交換兩個位元不改變 $M$，所以 $[M,\mathrm{SWAP}_{jj'}]=0$。
3. 所以 $M$ 與 $U_T$ 的每個因子對易，$[M,U_T]=0$。
4. 初始態 $|{+}\rangle^{\otimes n}$ 中 $\langle Z_k\rangle=0$，故 $\langle M\rangle_0=0$。
5. $\langle\psi|M|\psi\rangle=\langle{+}|U_T^\dagger MU_T|{+}\rangle=\langle{+}|M|{+}\rangle=0$。$\blacksquare$

**後果。** $\phi$ 的 $3n$ 個分量之間有一條線性約束，有效自由維度為 $3n-1$。

## 5. 命題 2：無耦合時的乘積態

**命題。** 若所有 $z_{n+j}=0$，則對任意 $T$：

$$
\phi_k(z)=\big(\cos 2\tau z_k,\ \sin 2\tau z_k,\ 0\big).
$$

**證明。** 沒有耦合時 $U_T=\prod_k\prod_{t=1}^{T}e^{-i\Delta z_kZ_k}=\prod_k e^{-i\tau z_kZ_k}$，各位元獨立。單一位元：

$$
e^{-i\tau z Z}|{+}\rangle=\tfrac1{\sqrt2}\big(e^{-i\tau z}|0\rangle+e^{i\tau z}|1\rangle\big)=:a|0\rangle+b|1\rangle .
$$

對單量子位元態 $a|0\rangle+b|1\rangle$：$\langle X\rangle=2\,\mathrm{Re}(a^*b)$，$\langle Y\rangle=2\,\mathrm{Im}(a^*b)$，$\langle Z\rangle=|a|^2-|b|^2$。這裡 $a^*b=\tfrac12e^{2i\tau z}$、$|a|=|b|$，代入即得 $(\cos2\tau z,\ \sin2\tau z,\ 0)$。$\blacksquare$

**單射性。** $2\tau|z_k|\le 0.8\pi<\pi$，所以角度 $2\tau z_k$ 落在長度小於 $2\pi$ 的區間內，$(\cos,\sin)$ 在此區間上一對一。不同輸入不會被映射成同一個特徵。

**後果。** 對照組 C3（關掉所有耦合）的核有封閉形式，不需要量子電路。實作與此式吻合到 $2.22\times10^{-15}$。

## 6. 群組 D 的補零與解耦

群組 D 用 $n=4$，共 $2n=8$ 個輸入（4 個場、4 條環邊的耦合）。LDA 只有 $k_{\mathrm D}=6$ 個有效方向，其餘補 0，即 $z_7=z_8=0$。

環的四條邊對應：

| 邊 | 對應輸入 |
|---|---|
| (1,2) | $z_5$ |
| (2,3) | $z_6$ |
| (3,4) | $z_7=0$ |
| (4,1) | $z_8=0$ |

位元 4 的兩條邊耦合都為 0，只剩局部場，依第 5 節它與其他位元解耦，$\phi_4=(\cos2\tau z_4,\sin2\tau z_4,0)$。糾纏只發生在 1–2–3 這條鏈上。

群組 B 用 $n=2$，3 個輸入（2 個場、1 個耦合）剛好填滿，沒有這個問題。

## 7. Trotter 誤差

一階 Trotter 的標準上界（Lloyd 1996；Childs et al. 2021）：

$$
\left\lVert U_T(z)-e^{-i\tau H(z)}\right\rVert \le \frac{\tau^{2}}{2T}\sum_{a<b}\left\lVert \left[H_a,\,H_b\right]\right\rVert
$$

其中 $H=\sum_a H_a$，每個場與每個耦合各算一項。

推導重點：兩項 $H_a$、$H_b$ 的一步誤差為

$$
\left\lVert e^{-i\Delta H_b}\,e^{-i\Delta H_a}-e^{-i\Delta (H_a+H_b)}\right\rVert \le \frac{\Delta^{2}}{2}\left\lVert \left[H_a,\,H_b\right]\right\rVert
$$

$T$ 步累加後，$T\Delta^{2}=\tau^{2}/T$，即得上式。

套用到本模型：當 $\lvert z\rvert\le\pi/2$ 時，各項的範數為 $O(1)$。每一項只和 $O(1)$ 個相鄰項不對易：場項只和含該位元的兩條邊不對易，耦合項只和共用位元的耦合與兩端的場不對易。所以非零的交換子共 $O(n)$ 個，

$$
\left\lVert U_T-e^{-i\tau H}\right\rVert = O\!\left(\frac{n\,\tau^{2}}{T}\right)
$$

**實測**（$n=3$、$\tau=0.7$，40 組隨機 $z$ 的平均）：讀出各分量的最大偏差

$$
\max_i \left\lvert \phi_i-\phi_i^{\mathrm{exact}}\right\rvert
$$

隨 $T$ 按 $1/T$ 下降。

| T | 最大偏差 |
|---|---|
| 4 | $1.93\times 10^{-1}$ |
| 256 | $3.05\times 10^{-3}$ |

對數座標斜率 $-0.997$，與一階 $1/T$ 一致。

**結論。** $T=4$ 時 $U_T$ 只是連續演化的粗略離散化。論文說「哈密頓量演化」，指的是**建構方式**，不是對 $e^{-i\tau H}$ 的忠實模擬。

> 待補：在實際操作點（$n=2,4$；$\tau=0.8$）重做同一診斷。

## 8. 頻率限制

依 Schuld, Sweke, Meyer (2021)，以資料重複上傳（同一個 $z$ 在 $T$ 步中重複出現）的電路，輸出的每個期望值都是 $z$ 的**三角多項式**，可出現的頻率由資料編碼生成元的本徵值差決定。

- **場輸入 $z_j$。** 每步的生成元是 $Z_j$，本徵值 $\pm1$，差為 $2$，乘上 $\Delta$ 得每步頻率間距 $2\Delta$。$T$ 步後最高頻率 $2T\Delta=2\tau$。
- **耦合輸入 $z_{n+j}$。** 每步的生成元是 $h_{jj'}$。$h=2\,\mathrm{SWAP}-I$，SWAP 本徵值 $+1$（三重態，3 個）與 $-1$（單態，1 個），所以 $h$ 的本徵值為 $+1$ 與 $-3$，差為 $4$，每步頻率間距 $4\Delta$。最高頻率 $4\tau$。

$\tau=0.8$ 時，場輸入的最高頻率是 $1.6$、耦合輸入是 $3.2$。這是一個低頻、頻帶受限的函數族，所以古典的隨機傅立葉特徵（RFF）替代模型可以緊密近似它。這正是對照組 C2 與主模型持平的原因。

## 9. 與程式碼對應

| 內容 | 位置 |
|---|---|
| 本文使用的電路與特徵讀出（`entangle=False` 即對照組 C3） | `Research/QuSpike_PQK/pqk.py` |
| 另一版電路（v2，不屬於本文結果） | `Research/QuSpike_PQK/pqk_v2.py` |
| LDA 投影與補零 | `Research/QuSpike_PQK/features.py` |
| 解耦位元偵測 | `features.py::decoupled_qubits` |
| 電路相關測試 | `tests/unit/test_pqk_circuit.py` |

## 參考文獻

- S. Lloyd, "Universal quantum simulators," *Science* 273, 1996.
- A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, S. Zhu, "Theory of Trotter error with commutator scaling," *Phys. Rev. X* 11, 011020, 2021.
- M. Schuld, R. Sweke, J. J. Meyer, "Effect of data encoding on the expressive power of variational quantum-machine-learning models," *Phys. Rev. A* 103, 032430, 2021.
- H.-Y. Huang et al., "Power of data in quantum machine learning," *Nat. Commun.* 12, 2631, 2021.
