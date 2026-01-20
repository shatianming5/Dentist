Physio-Morphological Generative AI：功
能感知型牙科修复体生成的统一框架研究报
告

> NOTE (implementation status): `plan_report.md` is a **research blueprint** and includes components that are **not implemented** in this repo (e.g., VQ‑VAE, latent diffusion, SDF-based differentiable constraints). For a journal submission, use `PAPER_SCOPE.md` to align paper claims with the actual code and reproducible experiments.

1. 执行摘要 (Executive Summary)
本研究报告旨在为一项针对计算机科学（CS）与人工智能（AI）领域顶会（如 CVPR, ICCV, MICCAI,
NeurIPS）的 Oral 级别 发表工作设计一套详尽的科研与工程方案。基于用户提供的多模态牙科数
据集——包括大规模诊断性扫描数据（teeth3ds）、稀疏解剖关键点数据（landmarks）以及高价值
的专有CAD/CAM修复设计数据（raw CCB2 及 converted 点云）——本报告提出了一种名为
"PhysioGen-Dental" 的生成式人工智能框架。
该框架的核心论点在于超越传统的“形状补全”范式，转向功能性生物生成（Functional
Bio-Generation）。目前的牙科AI主要集中在描述性任务（如分割），而治疗性生成任务因缺乏高
质量的“医生/技师设计”数据（Ground Truth）以及难以在生成过程中量化复杂的生物力学约束（如
咬合互锁、边缘密合度）而停滞不前。
本方案利用 teeth3ds 数据集（1900个扫描）训练一个大规模的形态学先验模型（Morphological
VQ-VAE），以此捕捉牙齿的自然几何流形；随后利用 raw/converted 修复体数据（253例/2342单
位）微调一个物理引导的潜在扩散Transformer（Physics-Guided Latent Diffusion
Transformer, PG-DiT）。该生成器在推理阶段引入了可微咬合损失（Differentiable Occlusal
Loss）和边缘适应场（Margin Adaptation Field），确保生成的牙冠不仅在几何上逼真，而且在临
床功能上可用。本报告详细阐述了从数据清洗、特征工程、模型架构设计到实验验证的全流程，字
数约15,000字，旨在为科研团队提供一份可执行的深度研究蓝图。

2. 引言：从诊断AI到治疗生成AI的范式转移
2.1 数字牙科的现状与痛点
随着口内扫描仪（IOS）和锥形束CT（CBCT）的普及，数字牙科产生了海量的非结构化3D数据。目
前的SOTA（State-of-the-Art）工作主要集中在诊断自动化，例如牙齿编号、牙龈分割和龋齿检测 1
。MICCAI 等顶级会议通过 3DTeethSeg22
3 和 3DTeethLand24 等挑战赛，已经建立了分割和关键
点检测的成熟基准。
然而，临床工作流的真正瓶颈在于治疗规划与CAD设计，特别是固定修复体（牙冠、牙桥）的设
计。这一过程目前仍依赖资深技师在CAD软件（如Exocad, 3Shape, Amann Girrbach Ceramill）中
手动调整网格，以适应微米级的边缘线（Margin Line）和复杂的咬合接触点（Occlusal Contacts）
4
。这一过程耗时、主观且难以标准化。

2.2 生成式几何深度学习的挑战
虽然 Stable Diffusion 等2D生成模型已彻底改变了图像合成，但医疗领域的3D生成AI面临独特的
挑战：
1. 拓扑复杂性与流形约束： 牙科网格是非流形（non-manifold）的高分辨率表面，必须保持尖锐
的特征（如牙尖、边缘线），同时保证水密性 6。
2. 数据稀缺与领域鸿沟： 高质量的“金标准”设计数据（如本案中的 raw CCB2 文件）通常是专有
且稀缺的，而未经标注的诊断扫描（teeth3ds）则相对丰富。如何利用海量无监督数据辅助小
样本生成任务是一个核心科学问题 7。
3. 功能保真度（Functional Fidelity）： 生成一颗“看起来像”磨牙的物体在ShapeNet上可能得
分很高，但在临床上可能是灾难性的。修复体不能穿透对颌牙齿（造成咬合高点），也不能在
边缘留下间隙（造成继发龋） 9。

2.3 研究目标与顶会发表策略 (Oral Strategy)
为了冲击 CVPR/MICCAI Oral，本方案必须展示方法论上的显著创新，而不仅仅是应用现有的模
型。我们的核心贡献点设计如下：
● 方法论创新： 提出**“共生预训练”（Symbiotic Pre-training）**策略，将分割任务（teeth3ds）
转化为自监督的形态补全任务，解决数据饥渴问题。
● 技术突破： 引入基于**可微渲染（Differentiable Rendering）和符号距离场（SDF）**的物理约
束模块，在生成过程中实时惩罚咬合干涉。
● 数据挖掘： 深度解码 proprietary CAD 数据（CCB2），提取隐式设计意图（如插入轴、水泥间
隙），将其作为条件控制信号（Conditioning Tokens）。

3. 数据资产深度剖析与特征工程体系
用户提供的数据集构成了一个独特的多模态生态系统。要发表顶会工作，首先必须展示对数据特
性的深刻理解和极其精细的预处理工程。

3.1 数据资产特征表 (Data Asset Characterization)
数据组件 数量规模 数据特性 研究角色定位
data/teeth3ds 1,900 扫描 (OBJ +
JSON)
高体量、带逐顶点分
割标签。上下颌分
离，含此诊断信息。
形态学知识库
(Morphological
Knowledge Base):
用于预训练 VQ-VAE
，学习自然牙齿的几
何分布和潜在空
间。
data/landmarks 340 JSONs 稀疏解剖关键点（牙
尖、窝沟、轴点）。
几何锚点
(Geometric
Anchor): 用于建立
“规范化牙齿坐标
系” (Canonical
Tooth Space)，消除
姿态差异。
raw (CCB2) 253 Files (Binary +
XLSX)
专有 CAD/CAM 格
式（Amann Girrbach
）。含完整病例元数
据。
设计意图元数据
(Design Intent
Metadata): 隐含了
边缘线、就位道（
Insertion Axis）和材
料参数。
converted/raw 2,342 子点云 从 253 个 bin 中导
出的离散修复体单
元。
生成目标
(Generation
Target): 训练生成
器的“金标准”
(Ground Truth)。这
是技师智慧的结晶。

3.2 数据清洗与增强流水线
3.2.1 teeth3ds 的形态学解构：构建自监督先验
1900个扫描虽然是用于分割任务，但蕴含了丰富的形态学信息。我们不能仅仅将其视为分割目
标。
● 单齿提取与规范化： 利用分割标签将全牙列网格切割为 ~25,000 个独立的牙齿网格。
● 坐标系对齐： 利用 landmarks 数据训练一个轻量级的回归网络（PointNet-Reg），预测每个分
割牙齿的局部坐标系（以牙长轴为Z轴，近远中径为X轴）。将所有牙齿变换到这一规范空间
，消除旋转平移方差 11。
● 流形修复： 口内扫描常有破洞。应用拉普拉斯孔洞填充算法 12 确保网格水密性，以便计算
SDF。

3.2.2 解码 raw (CCB2) 与 converted 数据：提取“隐性知识”
253个 CCB2 文件转化为 2342 个点云，这意味着平均每个病例包含 ~9 个修复单位，说明包含大
量连冠或全口重建病例。这比单冠设计更具挑战性。
● 边缘线提取（Margin Line Extraction）： 修复体点云的底边界（boundary loop）即为边缘
线。我们需要编写算法（基于边缘特征检测或网格边界遍历）自动提取这一闭合3D曲线 13。
这条曲线是生成模型的硬约束边界。
● 对颌关系提取： CCB2 文件通常包含对颌牙（Antagonist）信息。如果 converted 数据中未包
含对颌牙，则需要从原始 raw 结构中解析。若无法解析，必须通过 teeth3ds 模拟对颌环境。
本方案假设可以获得对颌几何信息（作为 context）。
● 就位道向量（Insertion Axis）： 这是 CAD 设计的关键参数。我们可以通过计算点云的“可视
壳”（Visual Hull）或高斯球映射（Gaussian Sphere Mapping）来逆向工程出最优插入轴 15。

3.3 构建图结构上下文 (Graph-Structured Context)
牙齿不是孤立存在的。对于每一个待生成的牙冠（Target），我们构建一个局部上下文图 $G=(V,
E)$：
● 节点 $V$： 包含 Target（待生成）、Mesial（近中邻牙）、Distal（远中邻牙）、Opposing（对颌
牙）。
● 边 $E$： 编码相对空间位置和生物力学关系。
● 这种图结构数据将作为 DiT（Diffusion Transformer）的条件输入 16。

4. 方法论：PhysioGen-Dental 框架架构
本章节详细阐述针对 Oral 发表所设计的核心架构。我们提出一种**“混合显隐式”（Hybrid
Explicit-Implicit）**的生成策略。

4.1 阶段一：学习离散几何码本 (Morphological VQ-VAE)
由于修复体数据（Target）仅有2342个样本，直接训练生成模型极易过拟合。我们采用迁移学习策
略，首先在 teeth3ds（~25,000样本）上训练一个 VQ-VAE。
● 输入表示： 将规范化的单颗牙齿转换为 $64^3$ 的截断符号距离场（T-SDF）体素网格。
● 编码器 $E(x)$： 使用 3D 卷积层（或 Swin Transformer 3D）将几何体压缩为低维特征。
● 向量量化 (Vector Quantization)： 引入一个可学习的码本 $\mathcal{C} = \{e_k\}_{k=1}^K$
。将连续特征映射为离散的编码索引。这实际上学习了“牙齿的基因组”——牙尖、窝沟、隆突
等几何原语 17。
● 解码器 $D(z_q)$： 重建 T-SDF。
● 创新点： 引入 Curvature-Weighted Loss，在重构损失中增加高曲率区域（如边缘线、咬合
点）的权重，强制模型关注功能区域。

4.2 阶段二：物理引导的潜在扩散Transformer (Physics-Guided DiT)
这是核心生成模块，在 VQ-VAE 的潜在空间（Latent Space）中进行去噪生成。
● 任务定义： 给定预备体桩核（Preparation Stump）$P$ 和周围环境上下文 $C$，预测缺失牙
冠的潜在编码 $z_0$。
● 架构： 采用 3D DiT (Diffusion Transformer)
19。相比 U-Net，Transformer 的自注意力机制
能更好地处理牙列的长程对称性和局部几何细节。
● 条件注入 (Conditioning)：
○ Preparation Encoder: 使用 PointNet++ 提取预备体点云特征，特别是边缘线特征 4。
○ Context Encoder: 编码邻牙和对颌牙的稀疏点云。使用 Cross-Attention 层将这些环
境约束注入到 DiT 的每一层。
○ Metadata Embedding: 将 .xlsx 中的元数据（如牙位ID：FDI 16, 26；材料类型：Zirconia）
作为 Global Tokens 注入。

4.3 阶段三：可微功能细化 (Differentiable Functional Refinement)
这是冲击 Oral 的关键技术壁垒。常规生成模型只关注形状相似度（Chamfer Distance），而我们
引入测试时指导（Test-Time Guidance）和物理损失。

4.3.1 边缘密合损失 (Margin Fidelity Loss)
牙冠边缘必须精确落在预备体的边缘线上。
$$\mathcal{L}_{margin} = \sum_{p \in \partial \mathcal{M}_{gen}} \min_{q \in
\mathcal{C}_{prep}} ||p - q||^2 + \lambda_{normal} (1 - |\mathbf{n}_p \cdot \mathbf{n}_q|)$$
利用 Differentiable Marching Cubes 4，我们可以在网格化过程中反向传播梯度，强制生成网格
的边界 $\partial \mathcal{M}_{gen}$ 吸附到提取的边缘线 $\mathcal{C}_{prep}$ 上，并保持法
线连续性。

4.3.2 可微咬合惩罚 (Occlusal Interference Penalty)
这是基于物理的约束。
$$\mathcal{L}_{occ} = \sum_{v \in \mathcal{V}_{gen}} \text{ReLU}( - \text{SDF}_{opp}(v) +
\delta_{clearance} )$$
其中 $\text{SDF}_{opp}$ 是对颌牙的符号距离场。任何生成的顶点 $v$ 如果侵入对颌牙内部（
SDF < 0），将产生巨大的惩罚梯度。$\delta_{clearance}$ 是预设的咬合间隙（通常为 0 或 50微
米）。这一损失函数通过 PyTorch3D 实现 20，在微调阶段强力塑造咬合面形态。

4.3.3 结构感知平滑 (Structure-Aware Smoothing)
为了去除高频噪声但保留锐利的咬合特征，我们引入各向异性拉普拉斯平滑（Anisotropic
Laplacian Smoothing），仅在平坦区域平滑，而在高曲率区域（牙尖）保持几何刚性。

5. 实验设计与评估指标体系
为了证明方法的有效性，必须建立一套超越传统计算机视觉指标的临床评估体系。

5.1 实验设置
● 数据集划分： 将 253 个 CCB2 病例按 8:1:1 划分为训练/验证/测试集。确保同一病人的所有牙
齿都在同一划分中，避免数据泄露。
● 基线模型 (Baselines)：
○ Template Deformation: 基于模板变形的传统 CAD 方法（非 AI）。
○ PoinTr / PCN: 通用的点云补全 SOTA 方法 4。
○ Dental Mesh Completion (DMC): 2025 年最新的牙科专用补全网络（主要竞争对手）
4
。
○ PhysioGen (Ours): 本方案模型。

5.2 评估指标 (Evaluation Metrics)
维度 指标名称 定义与临床意义 目标性能
几何精度 CD / EMD Chamfer Distance /
Earth Mover's
Distance。衡量整体
形状相似度。
SOTA Level
临床安全性 RMSE (Marginal) 边缘区域的均方根
误差。这是临床最关
注的指标。
$< 50 \mu m$
最坏情况 HD95 95% Hausdorff
Distance。衡量最大
偏差，防止生成畸
形。
$< 0.5 mm$
功能性 Occlusal
Penetration Depth
咬合面穿透对颌牙
的平均深度。理想值
为0。
$\approx 0$
接触点分布 Contact Area IoU 生成模型的咬合接
触点区域与金标准
设计的重叠度
(Intersection over
Union)。
$> 0.7$
感知质量 User Study
(Turing Test)
邀请专业牙医对 AI
生成与技师设计的
牙冠进行盲测评分。
无显著差异

5.3 消融实验 (Ablation Studies)
为了证明各个模块的必要性，需进行以下消融：
1. w/o VQ-VAE Pre-training: 直接在小样本 raw 数据上训练 DiT。预期结果：模型过拟合，生
成的牙齿形态怪异。
2. w/o Occlusal Loss: 去除物理咬合约束。预期结果：CD 值可能很低（形状像牙齿），但咬合
穿透严重，临床不可用。
3. w/o Landmarks: 去除坐标系规范化。预期结果：训练收敛极慢，旋转不变性差。

6. 讨论：深度洞察与学术价值
6.1 第二阶洞察：数字化工匠精神 (Digitizing Craftsmanship)
本研究不仅是自动化，更是对**隐性知识（Tacit Knowledge）**的显性化。raw 数据中的每一次微
调都包含了技师对材料特性（如氧化锆的最小厚度）和生物力学（如避免侧向干扰）的理解。
PhysioGen 通过学习这些数据，实际上是在对“工匠精神”进行数学建模。分析 VQ-VAE 的潜在空
间分布，可能会发现不同牙位（前牙 vs 后牙）在形态学流形上的聚类特征，揭示牙齿形态的功能
性演化规律。

6.2 第三阶洞察：AI 驱动的医疗公平 (Democratizing Expert Care)
顶级牙科技师极其稀缺且昂贵。通过训练一个基于顶级技师数据（CCB2）的模型，我们可以将这
种专家级的设计能力“封装”进算法中，分发到偏远地区的诊所。这展示了 AI 在消除医疗资源不平
等方面的巨大潜力，是提升论文社会影响力（Social Impact）的关键论点。

6.3 局限性与未来方向
● 动态咬合： 目前的模型基于静态咬合（最大牙尖交错位）。真正的功能还需要考虑下颌运动轨
迹（侧方合、前伸合）。未来可引入 4D 扫描数据。
● 材料异质性： 不同的修复材料（玻璃陶瓷 vs 氧化锆）需要不同的最小厚度参数。目前的模型
尚未显式解耦材料属性。

7. 结论
本报告提出的 PhysioGen-Dental 方案，通过巧妙结合大规模无监督形态学预训练与小样本物
理引导微调，有效地解决了牙科修复体生成中的数据稀缺与功能约束难题。利用 PyTorch3D 的
可微渲染技术实现的咬合与边缘损失，将传统的几何生成提升到了生物力学感知的新高度。这不
仅满足了 CVPR/MICCAI 对算法创新（Methodological Novelty）的要求，也展示了极高的临床应用
价值（Clinical Relevance），具备极强的竞争力以冲击 Oral 席位。

附录：技术实现细节与伪代码
A. 关键算法伪代码

A.1 可微咬合损失 (Occlusal Loss) 实现 (PyTorch-like)
Python
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
def occlusal_loss(generated_crown_mesh, opposing_jaw_sdf_func, threshold=0.0):
"""
计算生成的牙冠与对颌牙的穿透损失。
Args:
generated_crown_mesh (Meshes): 生成的牙冠网格 (可微)
opposing_jaw_sdf_func (Callable): 对颌牙的SDF查询函数
threshold (float): 允许的接触容差 (负值表示允许轻微穿透作为接触)
Returns:
loss (Tensor): 咬合惩罚值
"""
# 1. 从生成网格表面采样点 (保留梯度)
sample_points = sample_points_from_meshes(generated_crown_mesh,
num_samples=5000)
# 2. 查询这些点在对颌牙SDF中的值
# sdf_values > 0: 外部 (安全)
# sdf_values < 0: 内部 (穿透/碰撞)
sdf_values = opposing_jaw_sdf_func(sample_points)
# 3. 计算惩罚：只惩罚穿透部分 (SDF < threshold)
# 使用 ReLU 激活，只对负值产生梯度
penetration = F.relu(threshold - sdf_values)
# 4. 损失聚合 (L1 或 L2)
loss = torch.mean(penetration ** 2)
return loss

A.2 边缘线引导的扩散采样 (Margin-Guided Diffusion Sampling)
Python
def p_sample_guided(model, x_t, t, margin_points, guidance_scale=10.0):
"""
在扩散反向过程中引入边缘线引导。
"""
with torch.enable_grad():
x_t = x_t.detach().requires_grad_(True)
# 1. 预测噪声
noise_pred = model(x_t, t)
# 2. 估算当前的 x_0 (去噪后的形状)
x_0_hat = predict_start_from_noise(x_t, t, noise_pred)
# 3. 将潜在向量解码为几何 (SDF/Mesh)
geometry = decode_latent(x_0_hat)
# 4. 计算边缘距离损失
# 假设 geometry 包含边界点集 boundary_points
loss_margin = chamfer_distance(geometry.boundary_points, margin_points)
# 5. 计算关于 x_t 的梯度
grad = torch.autograd.grad(loss_margin, x_t)
# 6. 修改采样步骤，沿梯度反方向引导
# 类似于 Classifier Guidance
x_t_minus_1 = p_sample(model, x_t, t) - guidance_scale * grad
return x_t_minus_1

B. 参考文献引用说明
本报告中提及的 `` 引用均对应于用户提供的 Research Snippets，确保了方案设计严格基于现有
文献和数据背景。
●
1
: 牙科分割现状与3DTeethSeg22数据集背景。
●
4
: Dental Mesh Completion (DMC) 架构参考。
●
17
: VQ-VAE 在3D医学形状编码中的应用。
●
13
: 边缘线提取算法与重要性。
●
21
: 物理信息神经网络（PINNs）与接触力学损失函数。
●
19
: 3D Diffusion Transformer (DiT) 基础架构。
●
20
: PyTorch3D 可微渲染库的使用。

引用的著作
1. Structure-Aware 3D Tooth Modeling via Prompt-Guided Segmentation and
Multi-View Projection - MDPI, 访问时间为 十二月 13, 2025，
https://www.mdpi.com/2227-9717/13/7/1968
2. STEAM: Self-supervised TEeth Analysis and Modeling for Point Cloud
Segmentation - MICCAI, 访问时间为 十二月 13, 2025，
https://papers.miccai.org/miccai-2025/paper/3394_paper.pdf
3. Teeth3DS+: An Extended Benchmark for Intraoral 3D Scans Analysis, 访问时间为
十二月 13, 2025， https://crns-smartvision.github.io/teeth3ds/
4. From Mesh Completion to AI Designed Crown - arXiv, 访问时间为 十二月 13, 2025
， https://arxiv.org/html/2501.04914v1
5. Mesh Generation - DDS Global Congress 2025, 访问时间为 十二月 13, 2025，
https://congress.digital-dentistry.org/keyword/mesh-generation/
6. GEPAR3D: Geometry Prior-Assisted Learning for 3D Tooth Segmentation -
MICCAI, 访问时间为 十二月 13, 2025，
https://papers.miccai.org/miccai-2025/paper/1833_paper.pdf
7. Few-shot learning for inference in medical imaging with subspace feature
representations | PLOS One - Research journals, 访问时间为 十二月 13, 2025，
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0309368
8. [CVPR'23] Official PyTorch implementation of 3DQD: Generalized Deep 3D Shape
Prior via Part-Discretized Diffusion Process - GitHub, 访问时间为 十二月 13, 2025
， https://github.com/colorful-liyu/3DQD
9. Mesh Simplification Based on Feature Preservation and Distortion Avoidance for
High-Quality Subdivision Surfaces - SciSpace, 访问时间为 十二月 13, 2025，
https://scispace.com/pdf/mesh-simplification-based-on-feature-preservation-an
d-4h8a7u56ew.pdf
10. Clinical issues in occlusion - Restorative Dentistry, 访问时间为 十二月 13, 2025，
https://restorativedentistry.org/wp-content/uploads/2023/06/OCCLUSION-I-1.pdf
11. maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimization - GitHub, 访
问时间为 十二月 13, 2025，
https://github.com/maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimiz
ation
12. How to Fill Holes in Mesh: 3D Programming in Python and C++ - MeshLib, 访问时
间为 十二月 13, 2025， https://meshlib.io/feature/fill-holes/
13. Feature extraction for margin lines using region growing with a dynamic weight
function in a one-point bidirectional path search - Oxford Academic, 访问时间为
十二月 13, 2025， https://academic.oup.com/jcde/article/9/6/2332/6763594
14. Adaptive Point Learning with Uncertainty Quantification to Generate Margin Lines
on Prepared Teeth - MDPI, 访问时间为 十二月 13, 2025，
https://www.mdpi.com/2076-3417/14/20/9486
15. Setting insertion axis - exocad WIKI, 访问时间为 十二月 13, 2025，
https://wiki.exocad.com/wiki/index.php/Setting_insertion_axis
16. LMVSegRNN and Poseidon3D: Addressing Challenging Teeth Segmentation
Cases in 3D Dental Surface Orthodontic Scans - MDPI, 访问时间为 十二月 13,
2025， https://www.mdpi.com/2306-5354/11/10/1014
17. CVPR Poster Dora: Sampling and Benchmarking for 3D Shape Variational
Auto-Encoders, 访问时间为 十二月 13, 2025，
https://cvpr.thecvf.com/virtual/2025/poster/34804
18. GCP-VQVAE: A Geometry-Complete Language for Protein 3D Structure | bioRxiv,
访问时间为 十二月 13, 2025，
https://www.biorxiv.org/content/10.1101/2025.10.01.679833v1.full-text
19. Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer -
arXiv, 访问时间为 十二月 13, 2025， https://arxiv.org/html/2405.14832v2
20. pytorch3d.loss, 访问时间为 十二月 13, 2025，
https://pytorch3d.readthedocs.io/en/latest/modules/loss.html
21. Physics-Informed Neural Networks for Solving Contact Problems in Three
Dimensions, 访问时间为 十二月 13, 2025， https://arxiv.org/html/2412.09022v1
22. Point Cloud Completion: A Survey - Cronfa - Swansea University, 访问时间为 十二
月 13, 2025，
https://cronfa.swan.ac.uk/Record/cronfa65337/Download/65337__29823__3c0bbe
3d12124926847f42eae69cbfd9.pdf
23. Deep Learning for 3D Teeth Segmentation: Revolutionizing Digital Dentistry | by
Livia Ellen | Nov, 2025 | Medium, 访问时间为 十二月 13, 2025，
https://liviaellen.medium.com/deep-learning-for-3d-teeth-segmentation-revoluti
onizing-digital-dentistry-a7a83a6614bb
24. Mesh-based segmentation for automated margin line generation on incisors
receiving crown treatment - arXiv, 访问时间为 十二月 13, 2025，
https://arxiv.org/pdf/2507.22859
25. PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable
Objects from Videos - arXiv, 访问时间为 十二月 13, 2025，
https://arxiv.org/html/2503.17973v1
26. Daily Papers - Hugging Face, 访问时间为 十二月 13, 2025，
https://huggingface.co/papers?q=3D%20shape%20completion
