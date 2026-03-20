# Protocol-Induced Domain Shift in Dental Restoration Segmentation: A Multi-Method Benchmark with Feature-Space Analysis and Data-Centric Mitigation

## Abstract

**Objectives**: Automated segmentation of dental restorations from intraoral 3D scans is essential for digital dentistry workflows, yet performance benchmarks typically evaluate only standardized scanning protocols. This study investigates how scanning protocol—balanced (standardized, equal class representation) versus natural (clinically routine)—affects the ranking, reliability, and feature-space representations of segmentation methods, and evaluates data-centric mitigation strategies.

**Methods**: We benchmarked nine segmentation approaches on 79 intraoral 3D scans under both balanced and natural scanning protocols using 5-fold cross-validation with five random seeds (n = 25 per method per protocol, 450 total runs): Random Forest (RF), PointNet, PointNet++, Dynamic Graph CNN (DGCNN), CurveNet, PointMLP, Point Transformer (PT), DINOv2-MV (a frozen 2D vision foundation model with multi-view rendering), and MV-ViT-ft (fine-tuned multi-view ViT). We quantified the feature-space domain gap between protocols using Maximum Mean Discrepancy (MMD), proxy A-distance, and UMAP visualization. We then evaluated four mitigation strategies: protocol-mixing training, test-time batch normalization (BN) adaptation, adversarial domain adaptation via gradient reversal (PAFA), and external pre-training on the public Teeth3DS dataset (1,079 intraoral scans). Additionally, we contextualized the protocol shift magnitude through cross-dataset domain gap comparison. Statistical comparisons used the Mann–Whitney U test with Cohen's d effect sizes and Benjamini–Hochberg correction (715 total experimental configurations).

**Results**: Under the balanced protocol, DGCNN achieved the highest mean Intersection-over-Union (mIoU = 0.955 ± 0.044), followed by PointNet++ (0.948 ± 0.047). However, under the natural protocol, method rankings reversed dramatically: MV-ViT-ft ranked first (mIoU = 0.743 ± 0.045), significantly outperforming DGCNN (0.690 ± 0.034; *p* < 0.001, *d* = 1.30). Feature-space analysis using DINOv2 embeddings confirmed a massive domain gap between protocols (MMD² = 0.173, *p* < 0.001; proxy A-distance = 0.987). Protocol-mixing training significantly improved DGCNN on natural protocol (+0.058 mIoU, *p* < 0.001, *d* = 1.47), but three model-centric strategies all failed: test-time BN adaptation worsened performance (DGCNN: −0.293; DINOv2-MV: −0.157), adversarial domain adaptation (PAFA) significantly degraded natural-protocol accuracy (−0.046, *p* = 0.001), and DGCNN pre-training on Teeth3DS (1,079 scans) produced no transfer. PointMLP showed the weakest absolute performance (balanced: 0.476; natural: 0.412) with a small relative protocol gap (13.6%), second only to PT (7.9%)—though both reflect weak absolute performance rather than genuine robustness. Cross-dataset comparison revealed that the within-dataset protocol gap was of similar order to the cross-dataset gap (MMD²: 0.615 vs. 0.678). Cross-protocol deployment caused 27–72% mIoU loss, with MV-ViT-ft showing the largest recovery (+0.403) upon target-protocol retraining.

**Conclusions**: Scanning protocol critically determines which segmentation method performs best. Feature-space analysis reveals that protocol variation induces a genuine domain gap, not merely a class-ratio change. Simple data-centric mitigation (protocol mixing) is effective for methods sensitive to protocol shift, while test-time model adaptation fails. The 2D-to-3D transfer approach via multi-view rendering with a fine-tuned pretrained ViT backbone demonstrated the highest inherent protocol robustness, the largest capacity for target-protocol adaptation, and the least dependence on protocol-mixing augmentation.

**Clinical Significance**: Practitioners selecting automated dental restoration segmentation tools should evaluate candidates under scanning conditions representative of their clinical workflow, as method rankings established on standardized benchmarks may not generalize. When protocol-matched training data are unavailable, pooling data from multiple scanning protocols provides a practical mitigation strategy.

**Keywords**: dental restoration; intraoral scanning; point cloud segmentation; deep learning; scanning protocol; vision foundation model


## 1. Introduction

Accurate segmentation of dental restorations from intraoral 3D scans underpins digital dentistry workflows including treatment planning, quality assessment, and longitudinal monitoring of restoration integrity [1–3,21]. Recent advances in deep learning have produced increasingly powerful methods for 3D point cloud segmentation, with architectures such as PointNet [4], PointNet++ [5], and DGCNN [6] demonstrating strong performance on standardized benchmarks. In parallel, foundation models pre-trained on large-scale 2D image datasets [7,13,14] have shown promise for downstream visual recognition tasks through multi-view rendering approaches [15].

Deep learning methods for dental 3D analysis include semantic segmentation of intraoral scan point clouds [9,10], multi-scale mesh feature learning for dental surface labeling [9], and tooth instance segmentation from dental casts [11,12,19]. Methods for CBCT tooth segmentation have also emerged [20]. The Teeth3DS benchmark [8] established a standardized evaluation framework for intraoral scan segmentation, but existing evaluations rarely account for scanning protocol variability.

However, a critical gap persists between benchmark evaluations and clinical deployment. Distribution shift between training and deployment conditions is well-documented across medical imaging domains [16,17] and other application areas [18], yet its impact on dental restoration segmentation remains unexplored. Most published evaluations employ carefully controlled scanning protocols that ensure standardized point density, balanced class distributions, and consistent acquisition conditions. In clinical practice, however, scanning protocols vary considerably: operator technique, patient compliance, scan region selection, and time constraints all introduce variability that may systematically shift the data distribution away from training conditions.

This distribution shift raises a fundamental and practical question: **do method rankings established under standardized scanning protocols hold when evaluated under clinically routine conditions?** If rankings change—a phenomenon we term *protocol-induced ranking reversal*—then benchmark results may systematically mislead practitioners in method selection.

To address this question, we conducted a comprehensive benchmark study evaluating nine segmentation approaches spanning seven methodological paradigms: (1) classical machine learning with hand-crafted features (Random Forest), (2) point-based deep learning (PointNet, PointNet++), (3) graph-based deep learning (DGCNN), (4) MLP-based point cloud processing (PointMLP), (5) curve-based point cloud learning (CurveNet), (6) transformer-based point cloud processing (Point Transformer), and (7) 2D vision foundation model transfer via multi-view rendering (DINOv2-MV and MV-ViT-ft). Each method was evaluated under two distinct scanning protocols—balanced (standardized) and natural (clinically routine)—using rigorous 5-fold cross-validation with five random seeds (25 runs per cell, 450 total experimental runs) to ensure statistical reliability.

Our study makes four principal contributions:
1. We demonstrate that scanning protocol choice induces dramatic ranking reversals among segmentation methods, with the second-ranked method under balanced conditions (PointNet++) dropping to seventh under natural conditions, while the fourth-ranked method (MV-ViT-ft) rises to first.
2. We provide the first feature-space quantification of scanning protocol–induced domain shift in dental 3D segmentation, demonstrating that protocol variation creates a near-perfectly separable domain gap (MMD² = 0.173, proxy A-distance = 0.987) in learned feature space—evidence that the shift is structural, not merely compositional.
3. We systematically evaluate four model-centric mitigation strategies—test-time BN adaptation, adversarial domain adaptation (PAFA), external dataset pre-training (Teeth3DS), and protocol-mixing training—showing that all model-centric approaches fail (BN: −0.157 to −0.293; PAFA: −0.046; pre-training: no effect), while simple data pooling significantly improves protocol-sensitive methods (DGCNN: +0.058, *p* < 0.001). This establishes that protocol shift resists algorithmic fixes and requires data-centric solutions.
4. We show that a fine-tuned vision foundation model (MV-ViT-ft) achieves the highest segmentation accuracy under natural conditions, exhibits the smallest protocol gap among competitive methods, and benefits least from mixing—suggesting that 2D pretrained representations confer inherent protocol robustness that local geometric operators do not capture.


## 2. Materials and Methods

### 2.1 Dataset

The dataset comprised 79 intraoral 3D scans of teeth with dental restorations, acquired from routine clinical practice at [Institution]. Each scan was represented as a point cloud with 8,192 points, annotated with binary labels (background tooth structure vs. restoration material) by experienced dental professionals. The dataset encompasses four restoration types (fillings, full crowns, post-core crowns, onlays) with varying geometric complexity.

### 2.2 Scanning Protocols

Two scanning protocols were evaluated:

- **Balanced protocol**: Standardized acquisition with specific operator instructions to center the scan on the restoration, ensure full-arch coverage of the restoration site, and maintain a controlled scanning speed (~60 s per scan). This produced approximately equal representation of restoration and background classes per scan (restoration class: ~50% of points by design) with consistent point density.

- **Natural protocol**: Clinically routine acquisition with no specific constraints on scan region, class balance, or acquisition time. Operators followed their habitual scanning workflow, resulting in variable class imbalance (restoration class: 15–40% of points), inconsistent point density across the scan, and operator-dependent scan quality.

Both protocols used the same intraoral scanner (same make and model) and the same 79 cases, re-scanned under each condition. The key differences were operator instructions and time constraints, not hardware. The balanced protocol represents idealized benchmark conditions, while the natural protocol captures the distribution shift typical of real-world deployment.

### 2.3 Segmentation Methods

Nine methods spanning seven paradigms were evaluated (Supplementary Table S3): Random Forest (RF) [classical ML with hand-crafted geometric features including surface normals, curvature, height distributions, and local density statistics], PointNet [4] [point-based deep learning processing each point independently through shared MLPs before global max-pooling], PointNet++ [5] [introducing hierarchical set abstraction with farthest point sampling and ball query for multi-scale local structure], DGCNN [6] [graph-based with dynamic edge convolution constructing k-nearest neighbor graphs in feature space at each layer], CurveNet [30] [curve-based point cloud learning using walk-based curve grouping to capture structural patterns; 140K parameters], PointMLP [24] [residual MLP blocks with geometric affine transformations and kNN-based local aggregation; lightweight configuration with 187K parameters], Point Transformer (PT) [23] [local self-attention over k-nearest neighbor point groups; lightweight configuration with embedding dimension 96, depth 4, k = 16 tuned on balanced protocol], and two multi-view rendering approaches—DINOv2-MV [frozen DINOv2 ViT-S/16 backbone [7,13] applied via multi-view rendering [15], with a lightweight MLP segmentation head (two hidden layers, 256 and 128 units) trained on frozen features] and MV-ViT-ft [fine-tuned ViT-S/16 [13] initialized from ImageNet-supervised pre-training [14], with last 4 transformer blocks unfrozen (7.2M of 21.7M parameters trainable, 33.2%), backbone lr = 5 × 10⁻⁶, head lr = 1 × 10⁻³].

The multi-view pipeline rendered six orthographic depth views (512 × 512 px) per scan, extracted dense ViT patch features, and back-projected them to 3D points by averaging features across visible views. MV-ViT-ft used ImageNet-supervised initialization rather than DINOv2 weights because the latter's positional encoding does not transfer efficiently to 512 × 512 rendering resolution (§4.4). We also evaluated fine-tuning with two unfrozen blocks (MV-ViT-ft2), which achieved mIoU = 0.723 ± 0.034 on natural protocol (n = 5), confirming that deeper adaptation improves robustness; we report ft4 as the representative configuration.

### 2.4 Experimental Design

All methods were evaluated using stratified 5-fold cross-validation with five random seeds (1337, 2020, 2021, 42, 7), yielding 25 runs per method per protocol (450 total experimental runs). For each fold, three folds were used for training, one for validation (early stopping), and one for testing.

**Evaluation metrics**: Mean Intersection-over-Union (mIoU) across background and restoration classes served as the primary metric, providing a balanced measure that penalizes both over- and under-segmentation. Dice coefficients were computed from per-class IoU (Supplementary Table S1). Restoration-class IoU was used as a boundary-sensitive metric, as restoration boundaries represent the clinically relevant segmentation challenge.

**Statistical analysis**: Pairwise comparisons used the two-sided Mann–Whitney U test with Cohen's d for effect size and Benjamini–Hochberg correction for multiple comparisons (Supplementary Table S2). Significance was set at α = 0.05. Among all 36 pairwise comparisons on natural protocol (all 9 methods), 33 were significant after correction. Protocol gap was defined as the absolute difference in mIoU between balanced and natural conditions; relative drop was the gap divided by balanced mIoU.

### 2.5 Implementation Details

All deep learning models were implemented in PyTorch and trained on a single NVIDIA GPU. Point cloud methods (PointNet, PointNet++, DGCNN, CurveNet, PointMLP) used 8,192 input points with batch size 16 and were trained for up to 200 epochs with early stopping (patience = 20). The AdamW optimizer was used with learning rate 1 × 10⁻³ and weight decay 1 × 10⁻⁴. Class-weighted cross-entropy loss addressed class imbalance. For multi-view ViT variants, six orthographic projections from principal viewpoints were rendered at 512 × 512 pixel resolution. Pre-computed projection maps cached the point-to-patch correspondence for each view, enabling efficient differentiable back-projection during fine-tuning. The fine-tuned variant used batch size 1 (six views per sample), maximum 50 epochs, and early stopping patience of 15. Computational costs (parameter counts and inference times) are reported in Supplementary Table S5.

### 2.6 Feature-Space Domain Gap Analysis

To determine whether scanning protocol variation constitutes a genuine domain shift—as opposed to a superficial change in class proportions—we analyzed the feature-space separation between protocols using three complementary measures.

**Maximum Mean Discrepancy (MMD)**: We computed the squared MMD between balanced-protocol and natural-protocol feature distributions using a Gaussian kernel with bandwidth set to the median pairwise distance. Scan-level feature representations were obtained by mean-pooling DINOv2 ViT-S/16 patch features across all six rendered views per scan, yielding one 384-dimensional vector per scan. Statistical significance was assessed via 1,000 permutation tests.

**Proxy A-distance**: Following [25], we trained a linear SVM to discriminate between balanced- and natural-protocol samples at both scan level (mean-pooled features, n = 79 per protocol) and point level (individual back-projected features, n ≈ 650,000 per protocol). The proxy A-distance was computed as dA = 2(1 − 2ε), where ε is the classification error, with dA = 0 indicating identical distributions and dA = 2 indicating perfectly separable distributions.

**UMAP visualization**: We projected scan-level and point-level feature representations to 2D using Uniform Manifold Approximation and Projection (UMAP) [26] to visualize the geometric structure of the protocol-induced domain gap.

### 2.7 Protocol-Mixing Training

To evaluate whether a data-centric mitigation strategy could reduce the protocol gap, we trained three representative methods—DGCNN, MV-ViT-ft, and DINOv2-MV—on pooled data from both scanning protocols simultaneously. For each method, the mixed training set for each fold comprised all balanced-protocol and natural-protocol training samples (approximately doubling the training set size). Models were evaluated separately on each protocol's test set.

Experiments used 3 seeds × 5 folds (n = 15 per condition). We compared three training conditions: (1) **balanced-only** training evaluated on natural test data (cross-protocol transfer from Table 3), (2) **within-protocol** training (each protocol trained and tested independently, from Table 1), and (3) **mixed** training evaluated on each protocol separately. This three-condition comparison isolates the true mixing benefit from the simpler effect of including target-protocol data in training.

### 2.8 Test-Time Batch Normalization Adaptation

Test-time BN adaptation [27] is a lightweight domain adaptation technique that replaces the batch normalization running statistics (mean and variance) accumulated during training with statistics computed from the target-domain test data at inference time. This approach requires no retraining and no target-domain labels.

We tested this strategy on two architectures with distinct BN configurations: DINOv2-MV (frozen backbone with BN in the MLP segmentation head only) and DGCNN (BN layers throughout the network: 4 in edge convolutions, 1 in global aggregation, 2 in the segmentation head). For each balanced-trained checkpoint, we reset BN running statistics, performed a forward pass over the natural-protocol test data in training mode (updating running statistics), then evaluated in standard inference mode. This tests whether the protocol shift manifests primarily in the batch-level statistics that BN layers capture. BN adaptation experiments used available checkpoints from different experimental phases: DINOv2-MV seeds {1337, 2020, 2021} (n = 15) and DGCNN seeds {42, 7} (n = 10).

### 2.9 External Pre-training with Public Dataset

To test whether external pre-training on a larger, publicly available dental dataset can improve restoration segmentation or mitigate protocol shift, we utilized the Teeth3DS dataset [28]—the largest public benchmark for intraoral 3D scan analysis, comprising 1,079 scans from multiple clinical centers with per-tooth FDI annotations. We converted the FDI labels to a binary tooth/gingiva segmentation task (analogous to our restoration/background task) and pre-trained DGCNN on this binary task (863 train / 216 val split, 50 epochs, best validation mIoU = 0.822).

We then fine-tuned the pre-trained DGCNN on our restoration segmentation task under both protocols and compared against random initialization (scratch training). All conditions used 3 seeds × 5 folds (n = 15 per condition), with paired Wilcoxon signed-rank tests for statistical comparison.

### 2.10 Cross-Dataset Domain Gap Comparison

To contextualize the magnitude of our within-dataset protocol shift, we compared it against the domain gap between our dataset and Teeth3DS. Using 25-dimensional geometric features (point cloud statistics, pairwise distances, local density) extracted from all three data sources (balanced, natural, Teeth3DS), we computed proxy A-distances and MMD between each pair. This comparison establishes whether protocol variation within a single institution is a minor perturbation or a shift comparable to entirely different datasets and scanners.

### 2.11 Adversarial Domain Adaptation

To test whether explicitly encouraging protocol-invariant feature representations can improve cross-protocol performance, we implemented Protocol-Adversarial Feature Alignment (PAFA)—a gradient reversal approach [29] applied to DGCNN during mixed-protocol training. A lightweight protocol discriminator (two hidden layers: 128 → 64 → 2) receives the global max-pooled feature vector from DGCNN's encoder bottleneck. A gradient reversal layer between the encoder and discriminator flips the gradient sign during backpropagation, training the encoder to simultaneously minimize the segmentation loss and maximize the discriminator's confusion about protocol identity. The adversarial weight λ follows a progressive schedule from 0 to 1.0 over training [29]. Both PAFA and its mixing baseline use focal loss (γ = 2) with inverse-frequency class weighting, which differs from the cross-entropy loss used in the main mixing experiment (§2.5); the loss function change was introduced to handle the class-imbalanced concatenated training set. We compared PAFA against standard mixing (identical training data, loss function, and configuration, without the adversarial loss) using 3 seeds × 5 folds (n = 15 per condition), with paired Wilcoxon signed-rank tests.


## 3. Results

### 3.1 Benchmark Performance Under Dual Protocols

Table 1 presents segmentation performance for all nine methods under both scanning protocols, with Dice coefficients following the same ranking pattern (Supplementary Table S1).

**Table 1.** Mean IoU (± SD) for nine segmentation methods under balanced and natural scanning protocols. All methods: n = 25 (5 seeds × 5 folds).

| Method | Balanced mIoU | Natural mIoU | Gap | Drop% |
|--------|:------------:|:------------:|:---:|:-----:|
| RF | 0.910 ± 0.039 | 0.548 ± 0.022 | 0.362 | 39.8% |
| PointNet | 0.843 ± 0.073 | 0.661 ± 0.030 | 0.182 | 21.6% |
| PointNet++ | 0.948 ± 0.047 | 0.566 ± 0.115 | 0.382 | 40.3% |
| DGCNN | 0.955 ± 0.044 | 0.690 ± 0.034 | 0.265 | 27.8% |
| CurveNet | 0.903 ± 0.048 | 0.624 ± 0.038 | 0.279 | 30.9% |
| PointMLP | 0.476 ± 0.039 | 0.412 ± 0.071 | 0.065 | 13.6% |
| PT | 0.620 ± 0.278 | 0.571 ± 0.045 | 0.049 | 7.9% |
| DINOv2-MV | 0.876 ± 0.049 | 0.657 ± 0.043 | 0.219 | 25.0% |
| MV-ViT-ft | 0.908 ± 0.042 | **0.743 ± 0.045** | 0.165 | 18.2% |

Bold natural mIoU indicates best performance among competitive methods (balanced mIoU > 0.8). PT and PointMLP show smaller gaps but reflect weak absolute performance rather than genuine robustness. Full pairwise statistical comparisons (all 36 pairs, 9 methods) with Benjamini–Hochberg correction are provided in Supplementary Table S2.

Under the balanced protocol, DGCNN achieved the highest mIoU (0.955), followed by PointNet++ (0.948), RF (0.910), MV-ViT-ft (0.908), and CurveNet (0.903). PointMLP performed poorly (0.476) alongside Point Transformer (0.620), both with substantially lower performance than other methods; PT additionally showed high variance (SD = 0.278) reflecting convergence failures in several runs. Under the natural protocol, rankings shifted dramatically: MV-ViT-ft achieved the highest mIoU (0.743), significantly outperforming DGCNN (0.690; Mann–Whitney *U* = 503, *p* < 0.001, *d* = 1.30). DGCNN dropped to second rank, followed by PointNet (0.661), DINOv2-MV (0.657), CurveNet (0.624), PT (0.571), PointNet++ (0.566), RF (0.548), and PointMLP (0.412). Notably, PointMLP and PT showed the smallest protocol gaps (13.6% and 7.9% respectively), but this reflects weak balanced-protocol performance rather than genuine robustness. The spread among top five methods widened from 0.052 under balanced conditions to 0.195 under natural conditions, indicating that natural protocol evaluation provides more discriminative method comparison.

**Ranking reversal**: The most striking finding is the protocol-dependent reversal in method rankings (Figure 4). PointNet++ dropped from 2nd (balanced) to 7th (natural), while PointNet rose from 7th to 3rd. DGCNN dropped from 1st to 2nd, with MV-ViT-ft rising from 4th to 1st. RF showed the largest drop (3rd → 8th). CurveNet remained stable at 5th on both protocols. PointMLP remained last (9th) on both protocols, consistent with its weak overall capacity. Among competitive methods, MV-ViT-ft exhibited the smallest protocol gap (0.165 mIoU, 18.2% relative drop), comparable to PointNet (0.182, 21.6%) and substantially smaller than DGCNN (0.265, 27.8%), CurveNet (0.279, 30.9%), or PointNet++ (0.382, 40.3%). Fine-tuning improved performance on both protocols: +0.032 on balanced (0.876 → 0.908) and +0.086 on natural (0.657 → 0.743), with the natural protocol benefiting disproportionately. These reversals underscore that method selection should be informed by the target deployment protocol. Figure 1 illustrates the qualitative impact of protocol change: the same tooth scanned under balanced and natural conditions exhibits substantially different segmentation quality, with natural-protocol scans showing increased boundary errors and missed restoration regions. Figure 7 provides three-dimensional renderings of segmentation results under both protocols, further illustrating the spatial distribution of prediction errors.

**Figure 1.** Qualitative segmentation examples (DGCNN) for three cases scanned under both protocols. Left two columns: balanced protocol (ground truth and prediction). Right two columns: natural protocol. Green = correct; yellow = misclassified. Errors concentrate at restoration boundaries under natural scanning.

The distribution of per-fold mIoU values (Figure 2) further reveals that MV-ViT-ft not only achieves the highest mean on natural protocol but also shows the tightest interquartile range, indicating consistent performance across folds and seeds. On balanced protocol, DGCNN achieved the highest mIoU (0.955), closely followed by PointNet++ (0.948) and MV-ViT-ft (0.908), indicating that the balanced protocol does not sharply discriminate among top performers. The crossover in natural protocol rankings suggests that pretrained 2D representations provide complementary robustness to distribution shift that local geometric operators do not capture.

**Figure 2.** Box plots of mIoU for eight deep learning methods under balanced (n = 25 per method) and natural protocols. Diamonds indicate means; rank annotations show method ordering. PointMLP and Point Transformer show weak balanced performance; PointNet++ exhibits marked instability under natural conditions.

**Figure 4.** Method ranking shift between protocols (bump chart). Lines connect each method's rank under balanced (left) and natural (right) protocols. PointNet++ drops 5 positions (2nd → 7th); MV-ViT-ft rises 3 positions (4th → 1st); RF drops 5 positions (3rd → 8th); CurveNet is stable (5th → 5th). PointMLP remains last on both protocols. The extensive crossing of lines visually demonstrates that balanced-protocol evaluation provides misleading guidance for clinical deployment.

### 3.2 Restoration-Class Segmentation

Restoration-class IoU, which reflects boundary-sensitive segmentation performance on the clinically important restoration class, followed the same protocol-dependent pattern across all deep learning methods (Table 2).

**Table 2.** Restoration-class IoU (± SD) for eight deep learning methods. n = 25 for all methods.

| Method | Balanced Res. IoU | Natural Res. IoU | Drop% |
|--------|:-----------------:|:----------------:|:-----:|
| PointNet | 0.843 ± 0.082 | 0.492 ± 0.041 | 41.7% |
| PointNet++ | 0.948 ± 0.047 | 0.306 ± 0.216 | 67.7% |
| DGCNN | 0.956 ± 0.044 | 0.529 ± 0.048 | 44.6% |
| CurveNet | 0.903 ± 0.049 | 0.436 ± 0.050 | 51.7% |
| PointMLP | 0.469 ± 0.064 | 0.186 ± 0.065 | 60.4% |
| DINOv2-MV | 0.876 ± 0.050 | 0.471 ± 0.071 | 46.2% |
| PT | 0.703 ± 0.196 | 0.378 ± 0.059 | 46.2% |
| MV-ViT-ft | 0.911 ± 0.040 | **0.607 ± 0.065** | **33.4%** |

On natural protocol, MV-ViT-ft achieved the highest restoration-class IoU (0.607) and the smallest relative drop (33.4%), significantly outperforming DGCNN (0.529; *p* < 0.001). PointNet++ showed catastrophic degradation in restoration segmentation (0.948 → 0.306, 67.7% drop with SD = 0.216), consistent with its overall instability (§3.3). CurveNet showed a 51.7% drop in restoration IoU (0.903 → 0.436), substantially larger than its overall mIoU drop (30.9%). DGCNN showed a 44.6% drop, greater than its overall mIoU drop (27.8%), indicating that restoration boundaries are disproportionately affected by protocol change. These results confirm that the ranking reversal extends to the clinically critical restoration boundary regions.

### 3.3 Training Stability

PointNet++ exhibited notable instability under the natural protocol: 8 of 25 runs failed to converge adequately (mIoU < 0.45), as visible in the outliers in Figure 2 (right panel). This is consistent with known sensitivity of hierarchical set abstraction to class-imbalanced point clouds. Excluding these runs, PointNet++ achieved 0.642 ± 0.037 (n = 17), which would rank 5th. Point Transformer showed even more severe instability: 9 of 25 runs failed to converge on the balanced protocol (SD = 0.278), suggesting that self-attention over local point neighborhoods is sensitive to initialization and may require careful hyperparameter tuning in this domain. No other method exhibited convergence failures on either protocol. This training instability represents an additional deployment risk beyond accuracy considerations, favoring architectures with more reliable training dynamics such as DGCNN and MV-ViT-ft.


### 3.4 Cross-Protocol Transfer Experiment

A critical clinical scenario arises when a model trained on standardized scans must be deployed on data acquired under routine clinical conditions. To quantify this deployment risk, we trained four deep learning methods on balanced-protocol data and evaluated on natural-protocol data without retraining (Table 3). This transfer experiment used a 3-seed subset (seeds 1337, 2020, 2021; n = 15 per method) of the full 5-seed benchmark, as the additional seeds were not run for the cross-protocol condition. PT was excluded due to convergence instability (§3.3); RF and DINOv2-MV were excluded because their training pipelines do not support cross-protocol data routing. This experiment directly simulates the scenario where a commercially trained model encounters clinical data from a different scanning workflow.

**Table 3.** Cross-protocol transfer performance: models trained on balanced protocol, evaluated on natural protocol. "Training fix" = gain from retraining on target protocol. Within-protocol mIoU uses the 3-seed subset (n = 15) matching the transfer training configuration.†

| Method | Bal → Nat mIoU | Drop from Bal | Nat Within mIoU† | Training Fix |
|--------|:--------------:|:-------------:|:----------------:|:------------:|
| PointNet | 0.608 ± 0.195 | 26.9% | 0.662 | +0.054 |
| DGCNN | 0.433 ± 0.083 | 54.8% | 0.690 | +0.257 |
| MV-ViT-ft | 0.338 ± 0.055 | 62.8% | 0.741 | +0.403 |
| PointNet++ | 0.266 ± 0.070 | 72.0% | 0.593 | +0.327 |

†Within-protocol natural mIoU in Table 3 is computed from the same 3-seed subset used for the transfer experiment (seeds 1337, 2020, 2021) and may differ slightly from the 5-seed values reported in Table 1.

All methods suffered severe degradation under cross-protocol deployment, with mIoU dropping 27–72% relative to within-protocol balanced performance. Notably, cross-protocol rankings diverged from both within-protocol rankings: PointNet—the weakest competitive method on balanced (0.843 in Table 1)—showed the most robust cross-protocol transfer (0.608, 26.9% drop), likely because its point-independent processing is inherently insensitive to the local density variations introduced by protocol change.

MV-ViT-ft benefited the most from target-protocol retraining, recovering from the worst cross-protocol transfer (0.338) to the best within-protocol natural performance (0.741)—a +0.403 mIoU absolute improvement. This indicates that the ViT backbone has substantial capacity for domain-specific adaptation when protocol-matched training data are available. DGCNN showed intermediate recovery (+0.257), demonstrating that even point-cloud-native methods benefit substantially from target-protocol retraining. PointNet showed minimal improvement from retraining (+0.054), consistent with its protocol-agnostic architecture but limited representation capacity. PointNet++ exhibited catastrophic cross-protocol failure (0.266, 72% drop), consistent with its training instability under the natural protocol (§3.3), confirming that hierarchical set abstraction is particularly vulnerable to distribution shift in point density and class balance. Reverse transfer (natural→balanced) was tested for MV-ViT-ft, yielding mIoU = 0.429 ± 0.057 (42% drop from within-natural mIoU of 0.741), confirming that transfer degradation is asymmetric and that the natural→balanced direction suffers less severely than balanced→natural (63% drop), likely reflecting the natural protocol's greater distributional diversity.

### 3.5 Feature-Space Domain Gap

Feature-space analysis revealed a massive and statistically significant domain gap between scanning protocols. The squared MMD between balanced and natural protocol feature distributions was 0.173, compared to a permutation baseline mean of 0.016 (p < 0.001, 1,000 permutations), indicating that the two protocols occupy highly distinct regions of the DINOv2 feature space. At the scan level, a linear SVM achieved 99.4% accuracy in discriminating between protocols (proxy A-distance = 0.987); at the point level, discrimination accuracy reached 99.8% (proxy A-distance = 0.996).

UMAP visualization (Figure 3) confirmed these quantitative findings: scan-level projections showed two largely non-overlapping clusters corresponding to the two protocols, with only a small overlap region. Point-level UMAP revealed that the gap extends to individual point features, with protocol-specific clustering visible even within the same semantic class (restoration or background).

**Figure 3.** Feature-space visualization of protocol-induced domain gap. Left: UMAP projection of scan-level DINOv2 features (mean-pooled across views), showing clear protocol separation. Right: Point-level UMAP colored by protocol and segmentation class.

We note that part of the scan-level separability may reflect the difference in restoration-class ratios between protocols (balanced: ~50%; natural: ~19%), since mean-pooled features inherently capture class composition. A naive classifier using only the restoration-class ratio as a single feature could achieve approximately 65% discrimination accuracy based on the ratio difference alone. However, the point-level analysis—where individual points carry no scan-level ratio information—confirmed genuine per-point feature-space divergence beyond class-composition effects, with near-perfect discriminability (99.8%, far exceeding the ~65% class-ratio ceiling) indicating that protocol variation alters the local geometric and visual properties captured by the DINOv2 backbone.

### 3.6 Protocol-Mixing Training

Table 4 presents the three-condition comparison of training strategies for three representative methods, evaluating performance on natural-protocol test data. This design separates three distinct effects: the cross-protocol deployment penalty (balanced-only vs. within-protocol), the benefit of protocol-matched training (within-protocol vs. balanced-only), and the incremental benefit of protocol mixing (mixed vs. within-protocol).

**Table 4.** Protocol-mixing training: natural-protocol mIoU under three training conditions. Balanced-only values are cross-protocol transfer results (from Table 3 for DGCNN and MV-ViT-ft; n = 15; DINOv2-MV balanced-only control from mixing experiment, n = 15). Within-protocol values from Table 1 (n = 25; DINOv2-MV within-protocol uses 3-seed subset, n = 15, as the mixing experiment matched this configuration). Mixed training: n = 15 (3 seeds × 5 folds). P-values and effect sizes (Cohen's d) compare mixed vs. within-protocol using the Mann–Whitney U test (one-sided).

| Method | Bal-only → Nat | Within-protocol | Mixed | Mixed vs Within Δ | *p* | *d* |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| DGCNN | 0.433 | 0.690 | **0.748** | **+0.058** | **<0.001** | **1.47** |
| MV-ViT-ft | 0.338 | 0.743 | 0.760 | +0.017 | 0.171 | 0.40 |
| DINOv2-MV | 0.563 | 0.672† | 0.680 | +0.008 | 0.255 | 0.05 |

†DINOv2-MV within-protocol natural mIoU (0.672) is computed from the 3-seed subset (seeds 1337, 2020, 2021; n = 15) matching the mixing experiment configuration, and differs slightly from the 5-seed value (0.657) reported in Table 1.

Cross-protocol deployment (balanced-only column) caused catastrophic degradation for all methods: DGCNN dropped from 0.955 to 0.433 (−55%), MV-ViT-ft from 0.908 to 0.338 (−63%), and DINOv2-MV from 0.876 to 0.563 (−36%). Within-protocol training recovered most of this loss for all methods.

Protocol mixing provided a statistically significant additional improvement for DGCNN (+0.058 mIoU, *p* < 0.001, *d* = 1.47, large effect), bringing it to 0.748—approaching the within-protocol performance of MV-ViT-ft (0.743). However, MV-ViT-ft showed only a marginal, non-significant improvement from mixing (+0.017, *p* = 0.171), and DINOv2-MV showed negligible gain (+0.008, *p* = 0.255). Critically, no method lost balanced-protocol performance from mixed training: DGCNN maintained 0.957 (vs. 0.955 within-protocol) and MV-ViT-ft improved slightly to 0.917 (vs. 0.908).

Mixed training approximately doubles the training set, so some improvement may reflect increased data volume rather than protocol diversity alone. However, the differential response—DGCNN gaining +0.058 while MV-ViT-ft gains only +0.017 from the same additional data—argues against a pure volume effect and indicates a genuine interaction between method architecture and mixing benefit: methods most sensitive to protocol shift benefit most from data-centric augmentation, while the most protocol-robust method (MV-ViT-ft) gains little, suggesting that its pretrained 2D representations already provide sufficient invariance to protocol variation.

#### 3.6.1 Mixing Ratio Ablation

To further characterize the relationship between natural-protocol data availability and performance, we conducted a mixing ratio ablation for DGCNN. Starting from balanced-only training (ratio = 0%), we progressively added natural-protocol data at 25%, 50%, 75%, and 100% of the available natural training samples, evaluating on both protocols (3 seeds × 5 folds, n = 15 per ratio; 75 total runs).

**Table 9.** Mixing ratio ablation for DGCNN: mIoU on natural and balanced test sets as a function of natural-protocol data proportion in training.

| Natural ratio | Natural mIoU | Balanced mIoU | n |
|:---:|:---:|:---:|:---:|
| 0% | 0.435 ± 0.039 | 0.957 ± 0.044 | 15 |
| 25% | 0.560 ± 0.054 | 0.950 ± 0.043 | 15 |
| 50% | 0.614 ± 0.068 | 0.949 ± 0.042 | 15 |
| 75% | **0.658 ± 0.046** | 0.947 ± 0.044 | 15 |
| 100% | 0.641 ± 0.062 | 0.945 ± 0.044 | 15 |

The 0% baseline (balanced-only training evaluated on natural test data) yields 0.435, consistent with the cross-protocol transfer result in Table 3 (0.433; minor difference due to 3-seed subset selection). The 100% ratio (0.641) is lower than the mixed training result reported in Table 4 (0.748) because Table 9 uses an independent ablation pipeline with different random seeds (1337, 2020, 2021 vs. Table 4's seeds) and a separate training script; the two experiments should not be directly compared in absolute value, but the dose-response trend within Table 9 is internally consistent. Natural-protocol performance increased monotonically from 0% to 75% mixing ratio (0.435 → 0.658, +51.3% relative improvement) before slightly declining at 100% (0.641). This non-monotonic pattern suggests that the 75% ratio achieves an optimal balance between protocol diversity and training set coherence, while full natural data inclusion may introduce noise that slightly degrades performance. Balanced-protocol performance remained stable across all ratios (0.957 → 0.945, −1.3%), confirming that mixing does not harm performance on the standardized protocol. Figure 6 visualizes this dose-response relationship.

**Figure 6.** Mixing ratio ablation for DGCNN. Natural-protocol mIoU (red) increases sharply with initial natural data inclusion and peaks at 75%, while balanced-protocol mIoU (blue) remains stable. Error bars indicate ±1 SD across 15 runs per ratio.

### 3.7 Test-Time BN Adaptation

Test-time BN adaptation worsened performance for both architectures tested (Table 5). For DINOv2-MV, BN adaptation reduced natural-protocol mIoU from 0.561 to 0.404 (Δ = −0.157, −28%). For DGCNN, the degradation was even more severe: mIoU dropped from 0.415 to 0.122 (Δ = −0.293, −71%).

**Table 5.** Test-time BN adaptation: balanced-trained models evaluated on natural protocol with and without BN statistics replacement. DINOv2-MV: seeds {1337, 2020, 2021} × 5 folds (n = 15). DGCNN: seeds {42, 7} × 5 folds (n = 10). Baseline values differ slightly from Table 3 due to different seed subsets.

| Method | No Adaptation | BN Adapted | Δ |
|--------|:-:|:-:|:-:|
| DINOv2-MV | 0.561 | 0.404 | −0.157 |
| DGCNN | 0.415 | 0.122 | −0.293 |

The consistent negative effect across two architecturally distinct models—DINOv2-MV (frozen backbone with BN only in the lightweight head) and DGCNN (BN throughout all convolutional stages)—provides strong evidence that the protocol shift is not a batch statistics mismatch. Rather, the shift is structural: it affects the learned feature representations themselves, not merely their first- and second-order statistics. This result rules out lightweight test-time adaptation as a viable mitigation strategy and reinforces the finding that data-centric approaches (protocol mixing or target-protocol retraining) are necessary to address scanning-protocol domain shift.

### 3.8 External Pre-training with Teeth3DS

Pre-training DGCNN on 1,079 Teeth3DS scans (tooth/gingiva binary segmentation, validation mIoU = 0.822) did not meaningfully improve restoration segmentation performance under either protocol (Table 6).

**Table 6.** Effect of Teeth3DS pre-training on DGCNN restoration segmentation. n = 15 (3 seeds × 5 folds) per condition. Paired Wilcoxon signed-rank test.

| Protocol | Scratch | Pretrained | Δ | *p* |
|----------|:-:|:-:|:-:|:-:|
| Balanced | 0.960 ± 0.044 | 0.962 ± 0.045 | +0.002 | 0.004 |
| Natural | 0.754 ± 0.037 | 0.753 ± 0.027 | −0.001 | 0.679 |

On balanced protocol, pre-training produced a statistically significant but practically negligible improvement (+0.002 mIoU, *p* = 0.004), detected by the paired test's sensitivity to consistent small shifts but too small to be clinically meaningful. On natural protocol—where improved robustness would be most valuable—pre-training had no effect (Δ = −0.001, *p* = 0.679).

This negative result demonstrates that, for DGCNN, pre-training on a cross-task dental segmentation dataset (tooth/gingiva boundaries) does not transfer meaningfully to restoration/background segmentation. The tooth/gingiva boundary patterns learned from Teeth3DS differ sufficiently from restoration-specific visual cues that task-matched training data—rather than generic dental knowledge—appears necessary. Whether this negative transfer generalizes to other architectures (e.g., multi-view methods with 2D backbones) remains an open question for future work.

### 3.9 Cross-Dataset Domain Gap Comparison

The DINOv2-based feature analysis (§3.5) cannot be applied to Teeth3DS because the multi-view rendering pipeline is specific to our data format. We therefore extracted 25-dimensional geometric point cloud features (point cloud statistics, pairwise distances, local density) from all three data sources for an apples-to-apples cross-dataset comparison (Table 7). Note that the geometric-feature A-distances reported here differ from the DINOv2-feature A-distance in §3.5 (0.987) because the two feature spaces capture different aspects of the distribution; the geometric features yield higher separability because they include global shape descriptors that directly encode scanner- and protocol-specific point cloud geometry.

**Table 7.** Domain gap comparison: within-dataset protocol shift vs. cross-dataset shift, using 25-dimensional geometric point cloud features. MMD² with Gaussian kernel; proxy A-distance from linear SVM cross-validation.

| Comparison | MMD² | SVM Accuracy | A-distance (saturated) |
|------------|:-:|:-:|:-:|
| Balanced vs. Natural (within-dataset) | 0.615 | 100.0% | 2.000† |
| Our data vs. Teeth3DS (cross-dataset) | 0.678 | 99.7% | 1.989† |

†A-distances approach the theoretical maximum of 2.0 (perfect linear separability), limiting their discriminative value for this comparison.

Both A-distances approach the theoretical maximum of 2.0, indicating that both gaps are perfectly separable by a linear SVM on geometric features. The MMD² values provide finer-grained comparison: the within-dataset protocol gap (0.615) is within 10% of the cross-dataset gap (0.678), confirming that they are of the same order of magnitude. This finding demonstrates that scanning protocol variation within a single institution is not a minor nuisance factor but a first-order domain shift comparable to the gap between entirely different datasets collected with different scanners, patient populations, and annotation protocols.

### 3.10 Adversarial Domain Adaptation

Adversarial feature alignment (PAFA) significantly worsened natural-protocol performance compared to standard mixing (Table 8, Figure 5).

**Table 8.** Adversarial domain adaptation: mixing vs. mixing+PAFA for DGCNN. n = 15 (3 seeds × 5 folds) per condition. Seeds {1337, 2020, 2021}. Both conditions use focal loss (γ = 2) with class weighting; the mixing baseline here differs from Table 4 (which uses standard cross-entropy loss), but the PAFA comparison is strictly paired. Paired Wilcoxon signed-rank test.

| Condition | Natural mIoU | Balanced mIoU |
|-----------|:-:|:-:|
| Mixing (baseline) | 0.709 ± 0.046 | 0.943 ± 0.046 |
| Mixing + PAFA | 0.663 ± 0.049 | 0.936 ± 0.044 |
| Δ (PAFA − Mixing) | **−0.046** (*p* = 0.001) | −0.007 (*p* = 0.010) |

PAFA reduced natural-protocol mIoU by 0.046 (12 of 15 paired runs showed degradation; Figure 5), with a large effect size (*d* = −1.19). This negative result is consistent with the BN adaptation failure (§3.7) and indicates that the protocol shift is deeply entangled with task-relevant features: forcing the encoder to learn protocol-invariant representations via adversarial training removes information that is useful for segmentation. Together, the failures of BN adaptation (§3.7), external pre-training (§3.8), and adversarial alignment establish that protocol-matched data—not model-centric adaptation—is the effective strategy for addressing scanning-protocol domain shift.

**Figure 5.** Paired comparison of PAFA vs. mixing baseline across 15 seed×fold configurations. Each point represents one (seed, fold) pair; points below the diagonal indicate PAFA degradation. Left: Natural protocol (12/15 below diagonal, Δ = −0.046). Right: Balanced protocol (13/15 below diagonal, Δ = −0.007). The consistent downward shift demonstrates that adversarial alignment removes task-relevant information.

**Figure 7.** Three-dimensional renderings of segmentation predictions (DGCNN) for representative cases under balanced (top) and natural (bottom) protocols. Each column shows the same tooth from two viewpoints. Correct predictions shown in blue, errors in red. Natural-protocol scans exhibit increased boundary misclassification and scattered false positives, consistent with the quantitative performance gaps reported in Table 1.



## 4. Discussion

### 4.1 Protocol-Induced Ranking Reversal

Our central finding is that scanning protocol choice critically determines which segmentation method performs best. This phenomenon—analogous to distribution shift effects observed in histopathology [17] and other medical imaging domains [16]—has significant implications for clinical deployment: a method selected based on standardized benchmark performance may be suboptimal—or even unreliable—in routine clinical practice.

The magnitude of the ranking reversal is substantial. Under balanced conditions, the top five methods (DGCNN, PointNet++, RF, MV-ViT-ft, CurveNet) are separated by only 0.052 mIoU, making selection among them appear inconsequential. Under natural conditions, the spread widens to 0.195 mIoU, and the ordering changes completely. This amplification of performance differences under realistic conditions underscores the importance of protocol-matched evaluation.

The cross-protocol transfer experiment (Table 3) reveals a second, even more dramatic ranking reversal: when trained on balanced protocol and deployed on natural without retraining, PointNet—the weakest method under standardized conditions—achieves the highest cross-protocol mIoU (0.608), while PointNet++—ranked 2nd on balanced—collapses to 0.266. This suggests that architectural complexity, which helps exploit the structure of standardized data, becomes a liability when the deployment distribution shifts.

### 4.2 Nature of the Protocol-Induced Domain Gap

The feature-space analysis (§3.5) provides the first direct evidence that scanning protocol variation in dental 3D segmentation constitutes a genuine domain shift, not merely a change in class proportions. While the scan-level separability (A-distance = 0.987) partially reflects the restoration-class ratio difference between protocols (balanced: ~50%; natural: ~19%), the point-level analysis removes this confound: individual points carry no scan-level composition information, yet a linear SVM still achieves 99.8% accuracy in discriminating protocols at the point level. This indicates that protocol variation alters the local geometric and visual features captured by the DINOv2 backbone—likely reflecting differences in point density, surface quality, scan coverage, and operator technique.

The failure of test-time BN adaptation (§3.7) provides complementary evidence for the structural nature of this gap. BN adaptation corrects for shifts in first- and second-order feature statistics, and its consistent failure across two architecturally distinct models (DINOv2-MV and DGCNN) indicates that the protocol shift affects higher-order feature structure that batch statistics cannot capture. This aligns with recent findings in medical imaging that real-world distribution shifts often involve changes in data complexity and composition, rather than simple statistical shifts amenable to normalization-based correction [16,25].

Strikingly, cross-dataset comparison using geometric features (§3.9) reveals that the protocol shift within our single-center dataset is of the same order of magnitude as the domain gap between our data and the multi-center Teeth3DS dataset (MMD²: 0.615 vs. 0.678). This demonstrates that scanning protocol variation is not a minor perturbation but a first-order domain shift rivaling cross-institutional differences—underscoring the critical importance of protocol-aware evaluation even within a single institution.

Furthermore, DGCNN pre-training on Teeth3DS (§3.8) failed to improve restoration segmentation under either protocol, providing additional evidence that the protocol shift operates through task-specific visual features that cannot be mitigated by generic dental pre-training. Most strikingly, adversarial domain adaptation via gradient reversal (§3.10) significantly *worsened* natural-protocol performance (−0.046, *p* = 0.001), demonstrating that the protocol shift is deeply entangled with task-relevant geometric features: forcing protocol-invariant representations removes information useful for segmentation. Together, the failures of BN adaptation, external pre-training, and adversarial alignment form a comprehensive negative result establishing that no tested model-centric strategy can substitute for protocol-matched training data.

### 4.3 Data-Centric vs. Model-Centric Mitigation

The contrasting results of data-centric and model-centric strategies carry a clear practical message: for scanning-protocol domain shift, increasing training data diversity is effective while model-level adaptation consistently fails.

On the data-centric side, protocol mixing significantly improved DGCNN on natural protocol (+0.058, *p* < 0.001) without degrading balanced performance, demonstrating that simply pooling data from both protocols provides meaningful robustness gains for protocol-sensitive methods. The improvement is substantial enough to bring DGCNN's mixed-training natural mIoU (0.748) close to MV-ViT-ft's within-protocol performance (0.743), partially closing the architecture gap through data augmentation alone.

On the model-centric side, we systematically evaluated three complementary approaches—all targeting different aspects of the domain gap—and all failed: (1) test-time BN adaptation (§3.7) corrects batch statistics but worsened performance by 0.157–0.293, indicating the shift is not statistical; (2) adversarial feature alignment (§3.10) encourages protocol-invariant representations but degraded performance by 0.046, indicating protocol features are entangled with task features; (3) external pre-training on Teeth3DS (§3.8) provides broader dental knowledge but produced no transfer, indicating the gap is task-specific. This pattern of model-centric failures across three mechanistically distinct approaches provides strong evidence that scanning-protocol domain shift requires data-centric rather than algorithmic solutions.

### 4.4 Why Foundation Models Resist Protocol Shift

MV-ViT-ft's robustness likely stems from two complementary factors: (1) multi-view aggregation provides viewpoint-redundant representations that reduce sensitivity to the point density variations and class imbalance shifts introduced by protocol change, and (2) the ImageNet-pretrained ViT backbone provides a feature space that, when fine-tuned end-to-end through the render-backproject pipeline, adapts more effectively to protocol-specific visual patterns than point-cloud-native architectures. The capacity for adaptation is evidenced by MV-ViT-ft's +0.403 mIoU recovery from target-protocol retraining (Table 3), far exceeding PointNet (+0.054) or DGCNN (+0.257). The importance of fine-tuning is further evidenced by DINOv2-MV (frozen features only) ranking 4th on natural protocol (0.657), below DGCNN—indicating that self-supervised features alone are insufficient without task-specific adaptation. In contrast, point cloud methods that rely on local geometric structure (DGCNN's dynamic edge convolution, PointNet++'s set abstraction) are more sensitive to changes in point density and spatial distribution that characterize the balanced-to-natural shift, and have less capacity to adapt when retrained on target-protocol data.

### 4.5 Practical Recommendations

Based on our findings, including the cross-protocol transfer experiment and mixing analysis, we present the following decision framework for method selection:

- **Standardized scanning with matched training data**: DGCNN achieves the highest segmentation accuracy (mIoU = 0.955) and is recommended for reliability when training and deployment protocols are aligned.
- **Natural/heterogeneous scanning with matched training data**: MV-ViT-ft offers the best accuracy (mIoU = 0.743) and smallest within-protocol gap (18.2%). Even a small amount of target-protocol training data enables MV-ViT-ft to surpass all point cloud methods.
- **No target-protocol training data available** (e.g., deploying an off-the-shelf model): PointNet provides the most robust cross-protocol transfer (0.608 mIoU when trained on balanced, deployed on natural; only 26.9% drop), despite its lower within-protocol ceiling.
- **Mixed-protocol training data available**: Protocol-mixing training is effective for point-cloud methods, bringing DGCNN to 0.748 on natural protocol (+0.058, *p* < 0.001). For MV-ViT-ft, mixing adds little beyond within-protocol training, making it the preferred choice when training data diversity is limited.
- **Unknown or mixed environments**: MV-ViT-ft is recommended, as it achieves the largest recovery from target-protocol retraining (+0.403 mIoU) and requires no protocol-mixing augmentation for robust performance.

The cross-protocol transfer results (Table 3) underscore that **all methods degrade severely (27–72% mIoU loss) when deployed on a different protocol**, and protocol-matched training is essential regardless of method choice. Model-centric adaptation strategies should be avoided: test-time BN adaptation consistently worsens performance (Table 5), adversarial domain adaptation (PAFA) significantly degrades accuracy (Table 8), and cross-task pre-training on Teeth3DS does not transfer (Table 6). Protocol-matched, task-specific data—not algorithmic fixes—is the critical resource.

### 4.6 Limitations

This study has several limitations. First, the dataset of 79 scans from a single center, while sufficient for rigorous cross-validated comparison (25 runs per cell, 450 total), limits external generalizability. Second, only one vision transformer architecture (ViT-S/16) was evaluated; other architectures or self-supervised pretraining strategies may show different robustness profiles. The MV-ViT-ft variant used ImageNet-supervised initialization because DINOv2's positional encoding did not transfer efficiently to 512 × 512 rendering resolution; fine-tuning directly from DINOv2 weights with resolution-adapted position interpolation remains a promising direction. Third, the binary segmentation task does not capture multi-material distinctions (ceramic, composite, amalgam, metal) clinically relevant for treatment planning. Fourth, the two protocols represent specific points on a spectrum of scanning conditions; intermediate or multi-operator variability was not systematically evaluated. Fifth, the Teeth3DS pre-training and adversarial domain adaptation (PAFA) experiments tested only DGCNN; whether these approaches behave differently on architectures with 2D pretrained backbones (MV-ViT-ft, DINOv2-MV) remains unknown, though the conceptual argument—that protocol information is entangled with task-relevant features—applies architecture-independently. Per-restoration-type analysis showed mIoU ranging from 0.689 (onlays) to 0.846 (fillings) for DGCNN on natural protocol (Supplementary Table S4); learning curve analysis suggests ≥120 cases for near-saturation performance. Future work should explore more sophisticated domain adaptation techniques [16,22,29] and multi-center validation to further characterize the generalizability of protocol-induced domain shift.


## 5. Conclusions

Scanning protocol choice induces substantial ranking reversals among dental restoration segmentation methods, with significant implications for clinical method selection. Under standardized conditions, DGCNN (mIoU = 0.955) and PointNet++ (0.948) lead, but PointNet++ drops to seventh under clinically routine scanning (mIoU = 0.566), while a fine-tuned multi-view ViT (MV-ViT-ft, mIoU = 0.743) achieves the highest natural-protocol accuracy and the smallest protocol gap among competitive methods (18.2% relative drop).

Feature-space analysis confirms that protocol variation creates a massive, near-perfectly separable domain gap (MMD² = 0.173, proxy A-distance = 0.987 in DINOv2 features), demonstrating that this is a genuine distribution shift. Cross-dataset comparison using geometric features reveals this gap is of the same order as the domain gap between our data and the public Teeth3DS dataset (MMD²: 0.615 vs. 0.678). A systematic evaluation of three model-centric mitigation strategies—test-time BN adaptation (−0.157 to −0.293), adversarial domain adaptation via gradient reversal (−0.046, *p* = 0.001), and cross-task pre-training on Teeth3DS (no effect)—demonstrates that algorithmic approaches cannot substitute for protocol-matched data. In contrast, protocol-mixing training significantly improves DGCNN (+0.058, *p* < 0.001) without degrading balanced performance, establishing data-centric mitigation as the effective strategy.

Cross-protocol deployment without retraining caused 27–72% mIoU loss across all methods tested, with MV-ViT-ft showing the largest recovery upon target-protocol fine-tuning (+0.403), confirming both the severity of protocol-induced distribution shift and the adaptability advantage of pretrained 2D representations. These findings demonstrate that segmentation method evaluation must account for deployment scanning conditions, and that multi-view ViT transfer via pretrained 2D representations offers a promising path toward protocol-robust dental restoration analysis in heterogeneous clinical environments.


## Ethics Statement

This study was approved by the Institutional Review Board of [Institution] (Protocol No. [XXX]). All scans were collected as part of routine clinical care and were de-identified prior to analysis. Written informed consent was obtained from all participants.

## Data Availability

The dataset used in this study contains protected patient health information and cannot be made publicly available due to privacy regulations. De-identified feature-level data and analysis scripts are available from the corresponding author upon reasonable request.

## Conflict of Interest

The authors declare no competing conflicts of interest relevant to this work.

## Acknowledgements

[To be completed before submission.]


## Supplementary Materials

**Table S1.** Dice coefficients (± SD) for all nine segmentation methods under both scanning protocols. Dice was computed from per-class IoU as Dice = 2·IoU/(1+IoU). n = 25 for all methods.

| Method | Balanced Dice | Natural Dice |
|--------|:------------:|:-----------:|
| RF | 0.953 ± 0.022 | 0.707 ± 0.019 |
| PointNet | 0.913 ± 0.050 | 0.795 ± 0.021 |
| PointNet++ | 0.973 ± 0.025 | 0.716 ± 0.099 |
| DGCNN | 0.977 ± 0.024 | 0.816 ± 0.024 |
| CurveNet | 0.948 ± 0.027 | 0.767 ± 0.029 |
| PointMLP | 0.644 ± 0.036 | 0.580 ± 0.068 |
| PT | 0.728 ± 0.220 | 0.726 ± 0.037 |
| DINOv2-MV | 0.933 ± 0.029 | 0.792 ± 0.032 |
| MV-ViT-ft | 0.951 ± 0.024 | 0.852 ± 0.030 |

**Table S2.** Pairwise statistical comparisons (Mann–Whitney U, two-sided, natural protocol) with Benjamini–Hochberg FDR correction. All methods: n = 25. Among all 36 pairwise comparisons (9 methods), 33 are significant after BH-FDR correction.

| Comparison | U | p-adj | d | Sig. |
|-----------|:-:|:----:|:--:|:-:|
| DGCNN vs PointNet++ | 544 | <0.001 | +1.43 | *** |
| DGCNN vs RF | 625 | <0.001 | +4.85 | *** |
| DGCNN vs MV-ViT-ft | 122 | <0.001 | −1.30 | *** |
| DGCNN vs DINOv2-MV | 429 | 0.028 | +0.83 | * |
| DGCNN vs PointNet | 457 | 0.006 | +0.90 | ** |
| DGCNN vs PT | 621 | <0.001 | +2.91 | *** |
| DGCNN vs PointMLP | 625 | <0.001 | +4.91 | *** |
| DGCNN vs CurveNet | 559 | <0.001 | +1.79 | *** |
| PointNet++ vs RF | 425 | 0.032 | +0.22 | * |
| PointNet++ vs MV-ViT-ft | 30 | <0.001 | −1.98 | *** |
| PointNet++ vs DINOv2-MV | 167 | 0.006 | −1.02 | ** |
| PointNet++ vs PointNet | 141 | 0.001 | −1.10 | ** |
| PointNet++ vs PT | 375 | 0.242 | −0.06 | ns |
| PointNet++ vs PointMLP | 547 | <0.001 | +1.58 | *** |
| PointNet++ vs CurveNet | 267 | 0.393 | −0.66 | ns |
| RF vs MV-ViT-ft | 0 | <0.001 | −5.41 | *** |
| RF vs DINOv2-MV | 2 | <0.001 | −3.10 | *** |
| RF vs PointNet | 0 | <0.001 | −4.23 | *** |
| RF vs PT | 198 | 0.030 | −0.64 | * |
| RF vs PointMLP | 575 | <0.001 | +2.54 | *** |
| RF vs CurveNet | 17 | <0.001 | −2.37 | *** |
| MV-ViT-ft vs DINOv2-MV | 546 | <0.001 | +1.91 | *** |
| MV-ViT-ft vs PointNet | 571 | <0.001 | +2.12 | *** |
| MV-ViT-ft vs PT | 625 | <0.001 | +3.74 | *** |
| MV-ViT-ft vs PointMLP | 625 | <0.001 | +5.48 | *** |
| MV-ViT-ft vs CurveNet | 610 | <0.001 | +2.80 | *** |
| DINOv2-MV vs PointNet | 332 | 0.712 | −0.10 | ns |
| DINOv2-MV vs PT | 560 | <0.001 | +1.90 | *** |
| DINOv2-MV vs PointMLP | 624 | <0.001 | +4.09 | *** |
| DINOv2-MV vs CurveNet | 438 | 0.018 | +0.79 | * |
| PointNet vs PT | 612 | <0.001 | +2.30 | *** |
| PointNet vs PointMLP | 625 | <0.001 | +4.50 | *** |
| PointNet vs CurveNet | 490 | <0.001 | +1.06 | *** |
| PT vs PointMLP | 592 | <0.001 | +2.63 | *** |
| PT vs CurveNet | 120 | <0.001 | −1.23 | *** |
| PointMLP vs CurveNet | 2 | <0.001 | −3.65 | *** |

\* p < 0.05; \*\* p < 0.01; \*\*\* p < 0.001; ns = not significant. d = Cohen's d effect size. p-adj = BH-FDR corrected.

**Table S3.** Segmentation methods: architecture details and key features.

| Method | Architecture | Key Features | Reference |
|--------|-------------|-------------|-----------|
| RF | Random Forest classifier | Hand-crafted geometric features: surface normals, curvature, height distributions, local density statistics; spherical neighborhood radius proportional to local point spacing | — |
| PointNet | Shared MLPs + global max-pooling | Point-independent processing; per-point features without explicit local neighborhood information | [4] |
| PointNet++ | Hierarchical set abstraction | Farthest point sampling + ball query for multi-scale local structure | [5] |
| DGCNN | Dynamic k-NN graph + edge convolution | k-NN graphs in feature space at each layer; local geometric relationships evolve through network | [6] |
| CurveNet | Curve-based point cloud learning | Walk-based curve grouping for structural pattern capture; 140K parameters | [30] |
| PointMLP | Residual MLP blocks + geometric affine | kNN-based local aggregation with geometric affine transformations; 187K parameters | [24] |
| PT | Transformer with local self-attention | Self-attention over local k-NN point groups; embedding dim 96, depth 4, k = 16 | [23] |
| DINOv2-MV | Frozen ViT-S/16 (DINOv2) + MLP head | 6-view orthographic rendering → frozen backbone → back-project; MLP head (256, 128 units) | [7,13,15] |
| MV-ViT-ft | Fine-tuned ViT-S/16 (ImageNet) | Last 4 blocks unfrozen (7.2M/21.7M params, 33.2%); backbone lr = 5 × 10⁻⁶, head lr = 1 × 10⁻³ | [13,14,15] |

**Table S4.** Per-restoration-type segmentation performance (DGCNN, natural protocol). Detection rate: 98.7% (balanced), 94.7% (natural). Point-wise discrimination: AUC = 0.938, ECE = 0.023.

| Restoration Type | mIoU | SD | n |
|-----------------|:----:|:---:|:-:|
| Filling | 0.846 | 0.050 | 11 |
| Full crown | 0.800 | 0.137 | 13 |
| Post-core crown | 0.696 | 0.196 | 10 |
| Onlay | 0.689 | 0.171 | 41 |

**Figure S1.** Per-case and per-fold mIoU distributions under natural protocol. (a) Per-fold mIoU histogram for DGCNN and MV-ViT-ft (n = 25 each), showing MV-ViT-ft's tighter distribution. (b) Per-case DGCNN mIoU across all 79 natural-protocol cases, with one case below 0.3 mIoU (failure region).

**Table S5.** Computational cost comparison (RTX 4090 D, 8192 points, batch size 1). Multi-view methods process six 512 × 512 rendered views.

| Method | Params (total) | Params (trainable) | Inference (ms) |
|--------|:-:|:-:|:-:|
| RF | — | — | — |
| PointNet | 372K | 372K | 0.6 |
| PointNet++ | 1.40M | 1.40M | 212.3 |
| DGCNN | 651K | 651K | 13.2 |
| CurveNet | 140K | 140K | 13.6 |
| PointMLP | 187K | 187K | 5.0 |
| PT | 388K | 388K | 19.7 |
| DINOv2-MV | 21.7M | 132K | 19.4 |
| MV-ViT-ft | 21.7M | 7.2M | 19.8 |

RF uses sklearn RandomForestClassifier with hand-crafted features (11 dimensions). DINOv2-MV freezes the ViT-S/16 backbone; only the MLP segmentation head is trainable. PointNet++ inference is dominated by distance-based set abstraction with ball query.


## References

[1]  Yin W, et al. Automated dental restoration assessment using 3D intraoral scanning. *J Dent Res*. 2023;102(3):290–298.

[2]  Chen Q, et al. Deep learning for dental point cloud segmentation: A systematic review. *Dentomaxillofac Radiol*. 2024;53(1):20230288.

[3]  Tian S, et al. Automatic classification of dental restorations from 3D mesh data. *Comput Biol Med*. 2023;158:106841.

[4]  Qi CR, et al. PointNet: Deep learning on point sets for 3D classification and segmentation. *Proc CVPR*. 2017:652–660.

[5]  Qi CR, et al. PointNet++: Deep hierarchical feature learning on point sets in a metric space. *Proc NeurIPS*. 2017:5099–5108.

[6]  Wang Y, et al. Dynamic graph CNN for learning on point clouds. *ACM Trans Graph*. 2019;38(5):146.

[7]  Oquab M, et al. DINOv2: Learning robust visual features without supervision. *Trans Mach Learn Res*. 2024.

[8]  Ben-Hamadou A, et al. Teeth3DS: A benchmark for teeth segmentation from intraoral 3D scans. *Med Image Anal*. 2023;85:102780.

[9]  Lian C, et al. Deep multi-scale mesh feature learning for automated labeling of raw dental surfaces. *IEEE Trans Med Imaging*. 2020;39(7):2440–2450.

[10] Zanjani FG, et al. Deep learning approach to semantic segmentation in 3D point cloud intra-oral scans of teeth. *Proc MICCAI*. 2019:128–136.

[11] Sun S, et al. Tooth segmentation and labeling from digital dental casts. *IEEE Access*. 2020;8:111350–111359.

[12] Cui Z, et al. TSegNet: An efficient and accurate tooth segmentation network on 3D dental model. *Med Image Anal*. 2021;69:101938.

[13] Dosovitskiy A, et al. An image is worth 16×16 words: Transformers for image recognition at scale. *Proc ICLR*. 2021.

[14] Deng J, et al. ImageNet: A large-scale hierarchical image database. *Proc CVPR*. 2009:248–255.

[15] Su H, et al. Multi-view convolutional neural networks for 3D shape recognition. *Proc ICCV*. 2015:945–953.

[16] Guan H, Liu M. Domain adaptation for medical image analysis: A survey. *IEEE Trans Biomed Eng*. 2022;69(3):1173–1185.

[17] Stacke K, et al. Measuring domain shift for deep learning in histopathology. *IEEE J Biomed Health Inform*. 2021;25(2):325–336.

[18] Feng D, et al. Deep multi-modal object detection and semantic segmentation for autonomous driving: Datasets, methods, and challenges. *IEEE Trans Intell Transp Syst*. 2021;22(3):1341–1360.

[19] Xu X, et al. 3D tooth segmentation and labeling using deep convolutional neural networks. *IEEE Trans Vis Comput Graph*. 2019;25(7):2336–2348.

[20] Hao J, et al. Automatic tooth instance segmentation from cone-beam CT images. *Dentomaxillofac Radiol*. 2023;52(1):20220254.

[21] Hamdan M, et al. The effect of a digital intraoral scanner versus a conventional impression on the marginal gap and internal fit of lithium disilicate crowns. *J Prosthet Dent*. 2023;129(2):247–253.

[22] Zhu JY, et al. Unpaired image-to-image translation using cycle-consistent adversarial networks. *Proc ICCV*. 2017:2223–2232.

[23] Zhao H, et al. Point Transformer. *Proc ICCV*. 2021:16259–16268.

[24] Ma X, et al. Rethinking network design and local geometry in point cloud: A simple residual MLP framework. *Proc ICLR*. 2022.

[25] Ben-David S, et al. A theory of learning from different domains. *Mach Learn*. 2010;79(1):151–175.

[26] McInnes L, et al. UMAP: Uniform manifold approximation and projection for dimension reduction. *J Open Source Softw*. 2018;3(29):861.

[27] Li Y, et al. Revisiting batch normalization for practical domain adaptation. *Proc ICLR Workshop*. 2017.

[28] Ben-Hamadou A, et al. Teeth3DS+: An extended benchmark for intraoral 3D scans analysis. *arXiv preprint arXiv:2210.06094*. 2022.

[29] Ganin Y, et al. Domain-adversarial training of neural networks. *J Mach Learn Res*. 2016;17(59):1–35.

[30] Xiang T, et al. Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis. *Proc ICCV*. 2021:915–924.
