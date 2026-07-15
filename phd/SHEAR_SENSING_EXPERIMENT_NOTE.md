# 雙層編織感測器剪切實驗筆記

## 目的

評估兩片相同的 woven capacitive tactile sensor，能否透過差動訊號辨識：

- 法向按壓（normal pressing）
- 剪切／黏著狀態（shear / sticking contact）
- 整體滑動（gross sliding）

若訊號具有足夠的可重複性，再進一步估計切向力 $F_x$、$F_y$。

## 核心概念

只把兩片感測器直接疊在一起並不足以感測 shear。系統必須包含可產生、儲存並回復剪切形變的機械結構：

```text
手指或高摩擦剛性壓頭
           ↓ Fz
           → Fx, Fy
┌─────────────────────┐
│ 上層 woven sensor    │  可小幅橫向移動／變形
├─────────────────────┤
│ 柔性 silicone spacer │  提供可回復的 shear deformation
├─────────────────────┤
│ 下層 woven sensor    │  固定於剛性基座，作為參考
└─────────────────────┘
```

重要條件：

1. 下層固定。
2. 上層只能小幅彈性位移，不能任意滑走。
3. 中間 spacer 必須有穩定的剪切剛度與良好回彈。
4. 壓頭與上層之間需要足夠摩擦力，避免實驗一開始便發生 sliding。
5. 兩片電容感測器可能互相產生 electrical cross-talk，實驗前需要檢查；必要時分時讀取、增加絕緣／屏蔽層。

## 可能的感測方式 

上下兩片感測器分別計算壓力加權質心：

$$
\mathbf{c}_{\mathrm{top}}=(c_{x,t},c_{y,t}),\qquad
\mathbf{c}_{\mathrm{bottom}}=(c_{x,b},c_{y,b})
$$

差動位移：

$$
\Delta\mathbf{c}
=
\mathbf{c}_{\mathrm{top}}-\mathbf{c}_{\mathrm{bottom}}
$$

沒有剪切形變時：

$$
\Delta\mathbf{c}\approx 0
$$

有剪切形變時：

$$
\Delta\mathbf{c}\neq 0
$$

若 spacer 在工作範圍內近似線性，可透過標定建立：

$$
\begin{bmatrix}
F_x\\
F_y
\end{bmatrix}
=
\mathbf{K}
\begin{bmatrix}
\Delta c_x\\
\Delta c_y
\end{bmatrix}
+
\mathbf{b}
$$

$\mathbf{K}$ 與 $\mathbf{b}$ 必須使用參考力感測器的真值資料擬合，不能只由電容訊號自行決定。

除了質心差，也應保存：

- 上下層完整 capacitance maps
- 總訊號強度與接觸面積
- 左右／上下壓力不對稱量
- `frameDiff`
- 峰值位置
- 上下層圖形的 spatial correlation
- 釋放後的回復量與 hysteresis

## 建議設備

- 兩片相同的 woven sensor
- 一層薄 silicone／rubber spacer
- 剛性固定底板
- 高摩擦圓形壓頭
- 可控制水平位移的 linear stage
- 可控制法向預載的 Z stage 或砝碼
- 六軸 force/torque sensor；最低限度也需水平 load cell
- 相機與表面標記，用來確認接觸點是否真的沒有滑動
- 同步時間戳記錄系統

## 第一階段：可觀測性測試

先不要訓練 AI，也不要直接宣稱能測量 shear force。

1. 固定下層感測器。
2. 放置 spacer、上層感測器與壓頭。
3. 施加固定的法向預載。
4. 記錄兩層的初始電容圖。
5. 緩慢向 $+X$ 移動壓頭，但保持 sticking contact。
6. 保持數秒後釋放水平作用。
7. 重複 $-X,+Y,-Y$。
8. 在不同接觸位置、法向預載和位移幅度下重複。
9. 使用相機確認壓頭沒有相對上層表面滑動。

需要觀察：

- 差動質心 $\Delta\mathbf{c}$ 是否與施力方向一致。
- 相反方向是否產生相反符號的訊號。
- 釋放後是否回到接近初始值。
- 同一條件重複多次是否得到相似結果。
- 改變法向預載後，shear 特徵是否仍可區分。
- 上下感測器是否出現 cross-talk、漂移或飽和。

## 動作判定

### Pressing

- 接觸質心基本不變。
- 上下層總訊號與接觸面積增加。
- 鄰近 taxel 大致對稱地增強。

### Shear-like deformation / sticking

- 原始接觸區域仍保持。
- 上下層質心出現小幅、方向性的相對偏移。
- 壓力分布重新分配或變得不對稱。
- 釋放切向作用後，差動訊號自動回復。
- 相機確認沒有 macroscopic sliding。

### Sliding

- 主要激活位置由原 taxel 轉移到相鄰 taxel。
- 接觸質心持續、單向移動。
- 原始接觸區域逐漸失活。
- 只釋放切向作用時，訊號不會自動回到初始分布。

接觸區域不移動不是 shear force 的定義。Shear force 是平行於接觸面的力；不移動只代表可能處於 static shear / sticking 狀態。

## 第二階段：標定與模型

只有第一階段確認訊號可觀測後才進行。

### 簡單模型

先以線性回歸建立：

$$
[F_x,F_y]^\mathsf{T}
=
f(\Delta c_x,\Delta c_y,\text{normal-load features})
$$

必須加入 normal-load features，避免把法向壓力變化錯判成切向力。

### 時序模型

若線性模型不足，可使用 CNN--GRU：

```text
輸入：
- top sensor 時序電容圖
- bottom sensor 時序電容圖
- frameDiff
- touchMask
- 上下層質心與總訊號

輸出：
- press / shear-like / slide
或
- Fx / Fy / Fz
```

資料必須按實驗 trial、接觸位置或使用者切分，不能把同一段相鄰 frames 同時放進 training 與 validation。

## 成功判準

在聲稱系統能感測 shear 前，至少需要：

1. 方向性：$+X,-X,+Y,-Y$ 產生可區分的訊號。
2. 重複性：相同輸入的結果一致。
3. 回彈性：釋放後回到接近零點。
4. 解耦能力：不同法向預載下仍能辨識 shear。
5. 滑動區分：能區分 sticking shear 與 gross sliding。
6. 獨立驗證：在未參與標定的 trials 上仍有效。
7. 若要報告牛頓值：必須以外部 force sensor 作 ground truth。

若只能觀察到方向性變形、但沒有力值標定，應稱為：

> shear-like deformation sensing 或 shear-state classification

而不是：

> direct shear-force measurement

## 下一個最小實驗

先用一個固定法向預載，在中心位置向四個方向做小幅推動及釋放。同步畫出：

1. 上層質心軌跡
2. 下層質心軌跡
3. 差動質心 $\Delta c_x,\Delta c_y$
4. 上下層總訊號
5. 參考 $F_x,F_y,F_z$

如果差動質心具有方向性、重複性及釋放回彈，再進入完整標定與 AI 分類。
