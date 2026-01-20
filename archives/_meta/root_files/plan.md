# PhysioGen-Dental：项目执行规划（可落地版）

> 本文件是把 `plan_report.md`（研究报告）改写成“能一步步照做”的工程/科研执行计划：按阶段拆解、每一步的输入/输出/检查点、以及每个关键难题的完整解决方法（多方案备选）。
>
> 仓库根目录：`D:\dentist`（Windows）≈ `/mnt/d/dentist`（WSL）

---

## 0. 现状快照（你现在已经有什么）

### 0.1 数据目录（已解压/已整理）
- Teeth3DS（上/下颌网格 + 分割 JSON）：`data/teeth3ds/`
  - 约 1900 个 `OBJ`，约 1800 个分割 `JSON`（详情见 `DATASET_STATS.md`）
- 官方划分/名单：`data/splits/`
- Landmarks（关键点 kpt）：`data/landmarks/`（详情见 `DATASET_STATS.md`）
- 修复体 raw（CloudCompare `CCB2` bin + Excel 标签）：`raw/`
  - 253 个 `*.bin` + 4 个 `*.xlsx`（详情见 `RAW_DATASET_STATS.md`）
- raw → 通用格式（点云）：`converted/raw/`
  - 2342 个子点云导出（`*.npz` + `*.ply`）及清单/标签（详情见 `README.md`）

### 0.2 关键产物/脚本（你可以直接复用）
- 数据统计：
  - Teeth3DS：`scripts/dataset_stats.py` → `DATASET_STATS.md` / `DATASET_STATS.json`
  - raw：`scripts/raw_dataset_stats.py` → `RAW_DATASET_STATS.md` / `RAW_DATASET_STATS.json`
- CCB2 解析与导出（纯 Python）：
  - `scripts/convert_ccb2_bin.py` → `converted/raw/manifest.json`
  - `scripts/label_converted_raw.py` → `converted/raw/manifest_with_labels.json` + `converted/raw/labels.csv`

### 0.3 约束（必须先承认的现实）
- 磁盘空间接近满：`/mnt/d` 使用率约 99%，可用约 11GB（继续生成体素/SDF/缓存会很危险）。
- 你偏好：尽量用 Python；避免 `apt-get update`/大体量系统依赖。
- raw 的 CCB2 解析目前是“点云层面的启发式提取”：能导出点，但不保证恢复全部对象语义/层级/变换/网格面片。

---

## 1. 这份数据适合做什么任务（从“立刻能做”到“论文级目标”）

### 1.1 立刻能做（不需要额外标注/不需要更深 CCB2 解析）
1) **Teeth3DS：牙齿形态表征/生成先验**
   - 单齿形态编码（自监督）：autoencoder / VQ-VAE / masked modeling
   - 牙位条件生成（FDI 号条件）：conditional generation
   - 网格/点云重建与修复（holes / noise robustness）
2) **raw：修复体类型分类（多分类）**
   - 输入：每个 `Group-*.bin` 导出的点云（可合并/可筛选）
   - 输出：修复体类型（如：全冠/桩核冠/高嵌体/充填/拔除…）
3) **raw：几何分布与可解释分析**
   - 不同修复体类型的厚度/体积 proxy、曲率统计、尺度分布、点密度分布
   - 普通标注里有“磨牙/前磨牙”可做多任务学习（修复体类型 + 牙位粗分类）

### 1.2 需要补齐信息后能做（更接近“可用的修复体生成”）
1) **条件修复体生成**：给定 *预备体（stump）+ 邻牙 + 对颌* → 生成修复体（crown/inlay…）
2) **功能约束生成/优化**
   - 咬合干涉（Occlusal interference）惩罚
   - 边缘密合（Margin fidelity）约束
3) **临床级评估**
   - 边缘误差（μm 级）、接触点分布、穿透深度、最薄厚度约束

> 关键点：你当前 `converted/raw` 只有“从 bin 里导出的若干点云对象”，但“哪个是 stump/哪个是 crown/哪个是对颌/哪个是邻牙”并未完全结构化；要走到 1.2，需要进一步**对象语义识别/更深解析/补数据**。

---

## 2. 总体路线（分成三条并行主线）

### 主线 A：数据工程（把数据变成可训练、可评估、可复现）
- A1. Teeth3DS：单齿拆分 → 规范化 → 表征（点云/网格/SDF）→ 训练索引
- A2. raw：从“bin/多对象点云”构造“样本定义”（case 级/对象级）→ 标签对齐 → 训练索引
- A3. 统一评估工具：CD/HD95/接触/穿透/边缘误差等指标脚本化

### 主线 B：可发表模型（从先验到生成）
- B1. 先验：在 Teeth3DS 上训练形态先验（VQ-VAE / AE）
- B2. 生成：在先验 latent 上做 conditional generation（扩散/DiT/自回归）
- B3. 功能：引入可微咬合损失、边缘损失做训练/推理时指导

### 主线 C：快速业务价值（先把 raw 的分类/检索跑起来）
- C1. “一周内可闭环”：raw 修复体类型分类 baseline（可汇报/可演示）
- C2. “两周内可交付”：raw 的 embedding 检索 + 可解释统计（类型聚类、异常检测）

---

## 3. Phase 0：数据冻结与版本化（不做这一步，后面全会乱）

### 目标
- 把“当前有什么数据、怎么切分、怎么复现”写死，后续任何实验都可追溯。

### 输入
- 已存在目录：`data/`、`raw/`、`converted/raw/`

### 输出（建议新建，尽量轻量）
- `metadata/data_inventory.json`：记录目录大小、文件数量、关键清单文件哈希（可选）
- `metadata/splits_raw_case.json`：raw 的 case 切分（按 `*.bin` 为单位）
- `metadata/splits_teeth3ds.json`：Teeth3DS 的 case 切分（按官方 split + 你的派生拆分）

### 步骤（逐步扣细节）
1) 确认你要用的“样本单位（sample unit）”
   - Teeth3DS：`(ID, jaw)` 为 case；后续派生“单齿样本”时要把来源 case 写回去。
   - raw：以 `raw/**/Group-*.bin` 为 case（**必须 case 级切分，避免同一个病例泄漏到 train/test**）。
2) 固化清单
   - Teeth3DS：把 `data/teeth3ds/**/<ID>_<jaw>.obj/json` 路径扫一遍写入 index。
   - raw：读取 `converted/raw/manifest_with_labels.json`，把每个 `input` 作为 case key。
3) 固化 split（建议 8/1/1）
   - 把 case key 乱序（固定 seed）→ 划分 train/val/test → 写入 `metadata/splits_*.json`
4) 定义“任何派生数据的命名规则”
   - 统一使用：`processed/<dataset>/<version>/<case_key>/...`
   - 每个派生文件都要能反查到：原始输入路径 + 生成脚本版本 + 参数（写到旁路 json）

### 验收/检查点
- 任意一个样本都能从 `metadata/*` 找到：它来自哪个原始文件、属于哪个 split。

### 备选方案（完整解决方法）
- 方案 A（最省事）：只写 `splits_raw_case.json`，其余都靠现有 `*_STATS.json` + `manifest_with_labels.json`
- 方案 B（最稳）：额外计算每个关键文件的 `sha1`（防止日后文件被覆盖）
- 方案 C（最轻量）：不计算哈希，但把目录大小与文件数写死（至少能发现大改动）

---

## 4. Phase 1：raw（修复体）先做一个“能跑通的样本定义”（否则后面训练无从谈起）

### 目标
- 在不做更深 CCB2 解析的前提下，把 raw 变成一个“可训练的分类数据集”，并给后续生成任务预留升级空间。

### 输入
- `converted/raw/manifest_with_labels.json`
- `converted/raw/**/*.npz`（每个含 `points`）
- 标签：`converted/raw/labels.csv`

### 输出（建议）
- `processed/raw_cls/v1/`：
  - `index.jsonl`：每行一个样本（含路径、label、split、统计信息）
  - `samples/<case_key>.npz`：一个样本一个点云（已下采样、归一化）
  - `report.md`：数据清洗与过滤统计（删了多少、原因是什么）

### 你必须先决定：raw 的“一个样本”到底是什么？
这里给你 4 种样本定义（建议从 A 开始，能跑起来再升级）：
1) **定义 A（推荐起步）**：*case 级*（每个 `*.bin` 一个样本）
   - 做法：把该 bin 导出的所有子点云点集合并 → 下采样到固定点数 N → 用 bin 的标签训练分类
   - 优点：标签天然对齐；split 容易；最不容易错
   - 缺点：混入了“非目标对象”（比如残留网格、分割中间产物），分类更像“场景分类”
2) **定义 B（对象级）**：每个导出的子点云作为样本（2342 个）
   - 做法：每个子点云继承 bin 标签
   - 风险：标签可能只对其中某个对象成立 → 噪声标签非常大
3) **定义 C（对象级 + 自动筛选）**：每个 bin 选 1~K 个“最可能是修复体”的子点云
   - 典型规则：过滤掉超大点云（像全场景）、过滤掉超小点云（碎片），优先某些 name pattern
4) **定义 D（对象级 + 人工映射）**：做一个小工具，人工点选每个 bin 的目标对象
   - 一次性劳动换来高质量 paired 数据；后续能支持条件生成、边缘/咬合评估

### 步骤（按“定义 A：case 级”展开到可执行细节）
1) 合并点云（每个 case）
   - 从 manifest 找到该 `input` 的 `exported_clouds[].outputs.npz`
   - 逐个加载 `points`（shape: `[M, 3]`）
   - 过滤明显无效对象：
     - `M < min_points`（例如 500）直接丢弃
     - 可选：按 bbox 体积或点密度过滤离群
   - 合并：`points_all = concat(points_i)`
2) 归一化（必须写死规则，否则模型学到尺度差异）
   - 平移：`points_all -= mean(points_all, axis=0)`
   - 尺度：除以 `max(||p||)` 或 bbox 对角线（两种都行，但要固定）
   - 可选：PCA 对齐（只用于增强，不建议当作唯一对齐手段）
3) 下采样到固定 N（例如 N=4096）
   - 方案 A：随机采样（最快、足够做 baseline）
   - 方案 B：FPS（Farthest Point Sampling，更均匀，但实现/速度成本更高）
4) 写出样本与 index
   - `samples/<case_key>.npz`：至少包含 `points[N,3]`、`label`、`source_bin`
   - `index.jsonl`：写入点数、过滤比例、原始对象列表（方便回溯）
5) 划分 train/val/test（严格 case 级）
   - 直接复用 Phase 0 的 `metadata/splits_raw_case.json`

### 验收/检查点
- 253 个 case 里至少 >95% 能成功导出一个标准化样本（少量失败可接受，但要在 report 里说明原因）。
- 每个类别（全冠/桩核冠/高嵌体/充填/拔除…）在 train 集里至少有样本，否则需要合并类或重采样。

### 备选方案（完整解决方法）
- 如果“定义 A”分类效果很差：
  1) 上 “定义 C”：自动筛选更像“修复体目标”的对象（降噪）
  2) 上 “定义 D”：人工点选目标对象（高质量、可做生成）
- 如果点云太大导致内存压力：
  - 合并前先对每个子点云做下采样（例如每个最多 30k 点）
- 如果你想保留更多结构信息（例如上下颌关系）：
  - 不合并，改为“多对象输入”：每个 case 输入 K 个点云 + object name token（模型用 set encoder）

---

## 5. Phase 2：Teeth3DS → 单齿数据集（这是后续“先验预训练”的燃料）

### 目标
- 把 Teeth3DS 的“整牙列（上/下颌）+ 分割标签”拆成“单齿样本”，并提供可复用的规范化坐标系与几何表示。

### 输入
- `data/teeth3ds/**/<ID>_<jaw>.obj`
- `data/teeth3ds/**/<ID>_<jaw>.json`（有则用）
- `data/landmarks/**/<ID>/*__kpt.json`（只覆盖一部分样本）

### 输出（建议）
- `processed/teeth3ds_teeth/v1/`
  - `index.jsonl`：每颗牙一行（source case、jaw、FDI label、instance id、点数/面数、bbox）
  - `teeth/<case_key>/<tooth_key>.npz`：单齿点云（+ 可选法线）
  - 可选：`teeth_sdf/<...>.npz`：单齿 T-SDF（体素），仅在磁盘允许时生成

### 步骤（到“怎么切牙”这个层级）
1) 解析 OBJ
   - 必须支持：`v x y z [r g b]`（顶点可能带颜色）
   - 必须支持：`f i j k ...`（面片索引）
2) 解析 JSON 分割
   - 核验：`len(labels) == len(vertices)` 且 `len(instances) == len(vertices)`
   - 牙龈/背景通常是 `label=0` 或 `instance=0`
3) 定义“单齿实例”
   - 用 `instance id` 划分（`>0`）
   - 为每个 instance 统计它的主标签（FDI 号）：
     - `fdi = mode(labels[instances == inst_id])`
4) 构建单齿几何
   - 点云版本：直接取 instance 的顶点集（最快）
   - 网格版本（可选）：仅保留 3 个顶点都属于该 instance 的 faces，再 reindex 顶点
5) 基础清洗
   - 丢弃：点数过少（例如 < 500）或 bbox 异常（例如极薄/极长）
   - 可选：连通域过滤（保留最大连通分量，去掉碎片）
6) 规范化坐标系（非常关键：不然生成模型会学姿态）
   - 最低配（无 landmarks 覆盖的情况下也能跑）：
     1) 中心化：减去 centroid
     2) 尺度归一：除以 bbox 对角线
     3) PCA 对齐：主轴对齐到 (X,Y,Z)
     4) 消除翻转：用 jaw + FDI 号约束（例如上颌/下颌的“牙尖朝向”一致）
   - 高配（有 landmarks 或可训练对齐器）：
     - 用 landmarks 定义 canonical axes（长轴/近远中/颊舌向），并训练一个姿态回归器推广到全量
7) 写出 `index.jsonl` 与样本文件

### 验收/检查点
- 单齿样本数量应在 ~2.5 万量级（取决于每个 jaw 的牙数），且 FDI 分布合理（参考 `DATASET_STATS.md`）。
- 随机抽样可视化 50 颗牙，确保“大小一致、朝向大体一致、没有被切碎”。

### 备选方案（完整解决方法）
- 如果你不想处理网格拓扑/孔洞：
  - 直接用点云表示训练先验（Point-VQ-VAE / Point-AE），跳过 SDF
- 如果你必须用 SDF（为了可微碰撞/布尔运算）：
  1) 方案 A：mesh 修补后体素化（孔洞填充 + watertight）
  2) 方案 B：点云→Poisson 重建→体素化（更稳但会平滑细节）
  3) 方案 C：直接学习隐式场（Occupancy/SDF 网络），避免显式体素缓存（省磁盘）

---

## 6. Phase 3：raw 分类 baseline（最快闭环，用来验证“数据确实能学到东西”）

### 目标
- 用 Phase 1 的 `processed/raw_cls/v1/` 做一个可复现 baseline：给出可比的指标、误差分析、以及下一步升级方向。

### 输入
- `processed/raw_cls/v1/index.jsonl`

### 输出（建议）
- `runs/raw_cls_baseline/<exp_name>/`
  - `config.json`（含 seed、N 点数、augmentation）
  - `metrics.json`（acc/macro-F1/confusion）
  - `errors.csv`（错分样本列表，含回溯路径）

### 步骤（到“训练脚本怎么组织”这个层级）
1) 数据切分：严格用 case split（Phase 0）
2) 模型选择（先简单，再升级）
   - Baseline 1：PointNet（最省事、可快速验证）
   - Baseline 2：DGCNN（对局部几何更敏感）
   - Baseline 3：PointTransformer（需要更多算力）
3) 训练细节（要写进 config）
   - 输入点数 N：2048/4096（先 2048 跑通，再上 4096）
   - 增强：随机旋转（绕 Z）、缩放、抖动、dropout
   - 类别不平衡：class weight 或 focal loss（先试 class weight）
4) 评估
   - 必须输出：macro-F1（因为类别不平衡）
   - 必须输出：confusion matrix（看哪些类混淆）
5) 误差分析
   - 抽错分样本可视化：判断是“样本定义/噪声标签”问题还是“模型能力”问题

### 备选方案（完整解决方法）
- 如果噪声标签很大（定义 B/C 才会更明显）：
  - label smoothing、bootstrapping loss、cleanlab 进行噪声样本检测
- 如果类别太少导致学习不稳：
  - 合并小类（例如把“实在看不清/未知”合并到“其他”）
- 如果你需要可解释性：
  - 做 prototype retrieval：输出最相似的训练样本（embedding 余弦相似度）

---

## 7. Phase 4：论文级主线（PhysioGen-Dental）——从“形态先验”到“功能约束生成”

> 这一段是把 `plan_report.md` 的方法论拆成可以实现的工程步骤；你可以先做最小版本（先验 + 生成），再逐步加功能约束（咬合/边缘）。

### 7.1 Stage 1：形态先验预训练（VQ-VAE / AE）

**目标**
- 在 Teeth3DS 单齿样本上学到“自然牙齿形态流形”，给小样本 raw 提供强先验，减少过拟合。

**输入**
- `processed/teeth3ds_teeth/v1/`（点云或 SDF）

**输出**
- `checkpoints/morpho_prior/<run>/`：encoder/decoder/codebook
- `artifacts/morpho_embeddings/<run>.npz`：每颗牙的 embedding（用于分析/条件 token）

**工程步骤**
1) 决定几何表示（两条路线选一条先跑通）
   - 路线 A（省磁盘）：点云 VQ-VAE（PointNet encoder + folding/transformer decoder）
   - 路线 B（更适合物理约束）：SDF VQ-VAE（3D CNN encoder/decoder）
2) 损失函数（最小集合）
   - 重建：Chamfer Distance（点云）或 L1/L2（SDF）
   - codebook：commitment loss
   - 可选创新点：curvature-weight（高曲率区域加权）
3) 训练策略
   - 先 AE（不量化）跑通，再换 VQ（排错更容易）
   - 记录：重建可视化、embedding 分布（按 FDI 分组）

**备选方案（完整解决方法）**
- 如果 VQ 训练不稳定：
  - 用 EMA codebook、或先用 Gumbel-Softmax quantization
- 如果你发现 PCA 对齐不足（姿态差异太大）：
  - 加一个“姿态归一化模块”或“SE(3) 等变网络”（代价更大，但更稳）

### 7.2 Stage 2：条件生成（Latent Diffusion / DiT）

**目标**
- 给定条件（至少包括“缺失区域/预备体”）生成修复体/牙冠形态，在 latent 空间中扩散生成。

**关键前置：你得先有 paired 数据**
你目前 raw 里未必直接有 `stump → crown` 对；所以这里给出三种拿到 paired 数据的方法（按推荐顺序）：
1) **方法 P1（推荐）**：用 Teeth3DS 合成 paired（自监督 completion）
   - 从完整牙齿 T 合成预备体 P（切削/截断/加倒凹限制）→ 训练 `P → T`
   - 优点：样本量大（~2.5 万），能让条件生成先跑起来
2) **方法 P2**：从 raw 的多对象点云里“识别 stump 与 crown”
   - 需要对象语义识别（见 7.3 的 CCB2 解析升级）
3) **方法 P3**：引入你额外的临床数据（如果你说的 `D:\\dentist\\raw` 还有更全的对象）
   - 前提：能拿到 stump、对颌、邻牙或至少 stump

**工程步骤（用 P1 先跑通）**
1) 合成预备体（每颗牙）
   - 估计长轴：PCA 主轴或（有 landmarks 时）用解剖轴
   - 做切削：沿咬合方向截断一定比例（例如顶部 30%），再做边缘倒角（可选）
   - 输出：`(prep_points, target_points, margin_curve)`（margin_curve 可用切割平面与表面交线近似）
2) 训练条件生成器
   - 先做 latent AE：把 `prep` 和 `target` 都编码进同一 latent 空间（或者只编码 target，条件走 cross-attn）
   - 扩散模型选择：
     - DiT：更适合条件 token、长程依赖
     - U-Net latent diffusion：实现更成熟、门槛更低
3) 微调到 raw（如果有可用 paired 或者只做 unconditional + label token）
   - 最小版本：把 raw 的 target crown 当作无条件生成/类别条件生成（先验证“风格迁移”）

**备选方案（完整解决方法）**
- 如果你不想做扩散（工程量大）：
  - 用自回归离散 token（VQ code）生成（Transformer AR）
- 如果 paired 数据质量不足：
  - 先做“条件 inpainting”：只生成缺失局部 patch（更容易稳定）

### 7.3 Stage 3：功能约束（咬合/边缘）与可微细化

**目标**
- 让生成的修复体不仅“像”，还“能用”：不穿透对颌、边缘贴合预备体。

**你需要的额外输入（至少一项）**
- 对颌几何（opposing jaw）用于咬合干涉
- 预备体边缘线（margin line）用于边缘密合

**完整解决方法（按依赖程度排序）**
1) 方案 F1（最理想）：从 CCB2/原始 CAD 流程解析出 stump + opposing + margin
   - 需要升级 CCB2 解析（见下文“CCB2 深解析路线”）
2) 方案 F2（折中）：从 raw 的导出点云里自动识别 stump/opposing
   - 规则：按对象 name pattern、bbox 位置、点数规模聚类后判别
3) 方案 F3（最低配）：只做“自约束”
   - 边缘：用合成 prep 的切割交线当 margin
   - 咬合：用 Teeth3DS 的对颌牙列作为模拟环境（用同一病例上下颌构成）

**损失/约束实现细节（要写成可复现模块）**
- 咬合穿透（Occlusal）
  - 方案 A：对颌构建 SDF 网格，生成表面点查询 SDF 值并 ReLU 惩罚
  - 方案 B：点到对颌 mesh 的最近距离（BVH / PyTorch3D）
  - 输出指标：penetration depth（均值/最大值/95 分位）
- 边缘密合（Margin）
  - margin curve 提取（见下一节）
  - 输出指标：margin RMSE（μm）+ max error（HD95）

---

## 8. 关键难题：CCB2（CloudCompare BIN）如何“深解析”（给你完整路线图）

> 你已经有了“能提取点”的版本（`scripts/convert_ccb2_bin.py`）。如果你想把 raw 用到“条件生成/功能约束”，大概率需要更深的对象语义与变换信息。

### 你可能缺的到底是什么？
1) 对象树/层级（哪个点云属于哪个分组）
2) 每个对象的局部→全局变换矩阵（T）
3) 对象类型：点云/网格/多边形/标注线（margin 很可能是 polyline）
4) 标量场（厚度/距离）与单位（mm/μm）

### 完整解决方法（从轻到重）
1) 路线 C1（纯 Python，渐进增强，推荐先试）
   - 在现有脚本基础上：
     - 统计 `exported_clouds[].name` 的词频，建立 name→语义映射规则（例如 stump/opp/seg）
     - 尝试从 bin 中再恢复：对象数量、每个对象的变换、polyline（如果存在）
   - 优点：不需要系统安装；能逐步提升
   - 风险：CCB2 是私有格式，逆向可能需要时间
2) 路线 C2（CloudCompare 可视化导出，最稳但依赖外部软件）
   - 用 CloudCompare GUI 打开每个 bin，确认对象语义后批量导出 PLY/OBJ/Polyline
   - 优点：准确；能拿到 polyline
   - 缺点：人工成本高；不利于全自动复现
3) 路线 C3（CloudCompare CLI/插件批处理）
   - 如果能找到可用 CLI 参数（或脚本接口），可自动导出对象
   - 优点：自动化
   - 风险：环境/依赖复杂；与你“不要 update”偏好冲突

---

## 9. 关键难题：Margin Line（边缘线）怎么提取（完整方法清单）

### 你想要的输出是什么？
- 一条闭合 3D 曲线（polyline），点数可控（例如 256/512），可用于：
  - 训练条件 token（把 margin 当条件）
  - 评估 margin RMSE

### 完整解决方法（按数据类型分类）
1) 如果你有 stump 网格且它是开口边界（理想情况）
   - 方案 M1：网格边界遍历（boundary loop）
   - 方案 M2：alpha shape/concave hull 后取边界（对噪声更稳）
2) 如果你只有点云（当前 converted/raw 常见）
   - 方案 M3：估计局部法线与曲率，找“高曲率环”作为候选，再做 RANSAC 平面/圆柱拟合约束
   - 方案 M4：先重建网格（Poisson/BPA），再用 M1
3) 如果 margin 在 CCB2 里本来就有 polyline（最爽）
   - 方案 M5：直接解析 polyline（需要 CCB2 深解析或 CloudCompare 导出）

### 最小可行实现（先能用再变强）
- 在合成 paired（Teeth3DS 合成预备体）里：margin 就是“切割平面与表面交线”的离散近似 → 直接可用。

---

## 10. 关键难题：咬合约束怎么落地（完整方法清单）

### 你想要的输出是什么？
- 训练损失：生成体与对颌的穿透惩罚（可微）
- 评估指标：penetration depth / contact area

### 完整解决方法
1) 方案 O1（SDF 网格，最直观）
   - 对颌 → voxel SDF（例如 128³/256³）
   - 生成体表面采样点 → 查 SDF → ReLU(阈值 - SDF)
   - 代价：要体素缓存/或在线算 SDF；磁盘会紧张
2) 方案 O2（点-网格最近距离 + BVH）
   - 用 PyTorch3D/自建 BVH 计算点到三角面距离
   - 判别“在内/在外”较难（需要符号）；但只要不穿透可用近似
3) 方案 O3（近似接触：只做最小距离约束）
   - 不判断 inside，只惩罚距离小于阈值的点（作为“硬接触”近似）
   - 最适合你还没有稳定符号距离的阶段

---

## 11. 里程碑（建议你按这个顺序推进，避免一上来就做最难的）

### Milestone 1（1–3 天）：raw 分类闭环
- 产出：`processed/raw_cls/v1/` + baseline 指标 + 错误分析
- 目的：确认 raw 数据的“可学习性”，并反推样本定义/清洗是否合理

### Milestone 2（3–7 天）：Teeth3DS 单齿拆分 + 先验 AE 跑通
- 产出：单齿数据集 index + AE 重建可视化
- 目的：先把“形态先验”做出来，后续再谈生成

### Milestone 3（1–2 周）：Teeth3DS 合成 paired + 条件生成（最小版本）
- 产出：`prep → target` 的条件生成 baseline（不带功能约束）
- 目的：验证“条件生成管线”工程可行

### Milestone 4（2–4 周）：引入功能约束（边缘/咬合）与 raw 微调
- 产出：带功能指标的结果表 + 消融实验
- 前提：你能拿到 stump/opposing/margin（至少一部分）

---

## 12. 磁盘与算力策略（不然你会卡死在 IO 上）

### 12.1 当前空间风险
- `data/` ~30GB、`archives/` ~8.3GB、`raw/` ~4GB、`converted/` ~5.4GB
- 继续生成大体素（SDF）/缓存很容易把盘写满

### 12.2 具体建议
1) 优先用点云表示（少磁盘），把 SDF 做成“在线计算”（算力换磁盘）
2) `converted/raw` 如果必须腾空间：
   - 优先删 `*.ply` 保留 `*.npz`（通常更小，且训练更直接）
3) `archives/` 只在需要“可复现解压”时保留；否则可移到外盘/冷存

---

## 13. 你接下来要我做什么（从这个规划继续落地）

你可以从下面选一个，我就按这个规划继续往下实现：
1) 把 Phase 1 的 `processed/raw_cls/v1/` 真正生成出来（写脚本 + 输出 index/report）
2) 把 Teeth3DS 单齿拆分与规范化脚本写出来（生成 `processed/teeth3ds_teeth/v1/`）
3) 先写一个“manifest 探索脚本”：统计 raw 子点云 name 分布、点数分布、自动筛选规则（为定义 C 做准备）

