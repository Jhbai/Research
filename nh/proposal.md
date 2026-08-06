使用傳遞函數（Transfer Function, TF）來建模是非常實務且工業界最常用的做法（這稱為系統識別 System
Identification）。由於傳遞函數是「線性模型」，而化學反應是「非線性」的，因此你的傳遞函數必須是建立在**某個特定的操作點（Operating
Point，例如你期望的 NH_4^+ 和 pH Setpoint 附近）**上的近似模型。

要為這個系統建立傳遞函數模型，我們需要構建一個 2x2 的多變數系統 (MIMO)。以下是手把手教你如何建構這個模型的步驟：

第一步：定義 MIMO 傳遞函數矩陣架構

我們有 2 個輸入（操作變數 MV）和 2 個輸出（被控變數 CV）。在拉普拉斯域 (s-domain) 中，它們的關係可以用一個矩陣表示：

\begin{bmatrix} Y_{NH4}(s) \\ Y_{pH}(s) \end{bmatrix} = \begin{bmatrix} G_{11}(s) & G_{12}(s) \\ G_{21}(s) & G_{22}(s) \end{bmatrix} \begin{bmatrix} U_{NaOCl}(s) \\ U_{HClO}(s) \end{bmatrix}
  - Y_{NH4}(s)：NH_4^+ 濃度的變化量
  - Y_{pH}(s)：pH 值的變化量
  - U_{NaOCl}(s)：NaOCl 加藥量的變化量
  - U_{HClO}(s)：HClO 加藥量的變化量

第二步：選擇傳遞函數的模型形式 (FOPDT)

化工程序最常用的傳遞函數形式是 一階非延遲模型 (First-Order Plus Dead Time, FOPDT)。它的標準式為：

 G(s) = \frac{K}{\tau s + 1} e^{-\theta s} 

每個 G(s) 都有三個關鍵參數需要你來決定：

1.  K (Gain, 增益)： 代表「加 1 單位的藥劑，最終會讓濃度/pH 改變多少」。這是 MPC 決定用哪種藥的最關鍵參數！
2.  \tau (Time Constant, 時間常數)： 代表「反應有多快」。通常取決於你的反應槽體積 (V) 和進水流量
    (F)，大約是水力停留時間（HRT）的比例。
3.  \theta (Dead Time, 延遲時間)： 從加藥下去到感測器讀到數值變化的時間（管線距離、感測器反應時間）。

第三步：判斷各個 G(s) 的物理意義與參數方向 (正負號)

在你建 simulator 時，你可以先「假設」一組合乎物理常理的參數。我們來分析這四個方塊：

1. G_{11} (NaOCl 對 NH_4^+ 的影響)

  - 物理意義： 加次氯酸鈉能氧化氨氮，所以氨氮會減少。
  - K_{11} 符號： 負值 (-)。
  - 假設範例： G_{11}(s) = \frac{-2.0}{15s + 1} e^{-2s}

2. G_{12} (HClO 對 NH_4^+ 的影響)

  - 物理意義： 加次氯酸一樣能氧化氨氮，氨氮減少。而且在微酸性下，HClO 的氧化效率可能更好。
  - K_{12} 符號： 負值 (-)。
  - 假設範例： G_{12}(s) = \frac{-2.5}{15s + 1} e^{-2s} (假設 HClO 效率稍微好一點點，增益絕對值較大)

3. G_{21} (NaOCl 對 pH 的影響)

  - 物理意義： 次氯酸鈉是強鹼弱酸鹽，加進水裡會產生 OH^-，導致 pH 上升。
  - K_{21} 符號： 正值 (+)。
  - 假設範例： G_{21}(s) = \frac{+0.8}{10s + 1} e^{-2s} (pH 反應通常比除氨快一點，所以 \tau 設小一點)

4. G_{22} (HClO 對 pH 的影響)

  - 物理意義： 次氯酸是酸，加進水裡解離出 H^+，導致 pH 下降。
  - K_{22} 符號： 負值 (-)。
  - 假設範例： G_{22}(s) = \frac{-1.2}{10s + 1} e^{-2s}

第四步：如何在程式裡實作你的 Simulator 與 MPC？

現在你有了這個 2x2 的傳遞函數矩陣，你可以用 Python 來建立模擬器。

1. 使用 python-control 建立模擬器

如果你使用 Python，強烈建議安裝 control 套件。

import control as ctrl
import numpy as np

# 建立 s 變數
s = ctrl.tf('s')

# 根據我們上面的假設，建立四個傳遞函數 (這裡先忽略延遲 e^-2s 以簡化基礎模型，若要精確可以使用 Pade 近似)
G11 = -2.0 / (15*s + 1)
G12 = -2.5 / (15*s + 1)
G21 =  0.8 / (10*s + 1)
G22 = -1.2 / (10*s + 1)

# 組合成 MIMO 系統
Plant = ctrl.tf2io(
    [[G11, G12], 
     [G21, G22]],
    inputs=['NaOCl', 'HClO'],
    outputs=['NH4', 'pH']
)

# 為了給 MPC 使用，通常需要轉成狀態空間 (State-Space) 模型
Plant_ss = ctrl.tf2ss(Plant)

# 你可以使用 Plant_ss.A, Plant_ss.B, Plant_ss.C, Plant_ss.D 來餵給你的 MPC 演算法

2. MPC 是如何利用這個矩陣工作的？(思考邏輯)

當你把這個矩陣寫入 MPC 後，MPC 就有了「預測未來」的能力。

  - 情境： 今天進水 NH_4^+ 突然變高（超過 Setpoint）。
  - MPC 的思考： 我需要降低 NH_4^+。看了一下模型，G_{11} 和 G_{12} 都是負的，所以我可以加 NaOCl，也可以加 HClO。
  - 權衡 (Trade-off)： 如果我全部都加 NaOCl，因為 G_{21} 是正的，這會導致 pH 狂飆，超過我設定的 pH 限制。如果我全部都加
    HClO，因為 G_{22} 是負的，pH 會掉太低。
  - 最佳化計算： 於是 MPC 會自動計算出一個完美的混合比例：同時增加一點 NaOCl 和一點 HClO。這樣既能湊足總氯量把 NH_4^+ 壓下來（利用
    G_{11} 和 G_{12} 的總和），又能讓 G_{21} (升pH) 和 G_{22} (降pH) 互相抵消，維持 pH Setpoint！

給你的下一步實作建議：

1.  先寫出上述的矩陣： 在你的程式碼中把 K, \tau, \theta 當作變數（可以先用我上面提供的假設數字測試）。
2.  轉換為 Discrete State-Space： MPC 控制器都是在離散時間（如每 1 分鐘執行一次）下運作的，所以你需要用 c2d
    (Continuous to Discrete) 將傳遞函數轉成離散狀態空間矩陣。
3.  加上干擾模型 (Disturbance Model)： NH_4^+ 的進水濃度變化其實是一個干擾 (Disturbance, D)。你可以多加一個
    G_d(s) 代表「進水氨氮濃度對出水氨氮濃度的影響」，這通常是一個 K=1 但有時間延遲的傳遞函數。把這個餵給 MPC，MPC 就可以做前饋控制
    (Feedforward Control)。
