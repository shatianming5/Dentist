# Journal Positioning

**Direction**: full-case dental point-cloud restoration analysis with both segmentation and classification
**Date**: 2026-03-17

## Recommended Positioning

The paper should not be framed as a generic ML benchmark. For a strong dental journal, the central claim should be:

- a full-case 3D intraoral scan can be used to both localize restoration-related regions and classify restoration type,
- localization is not just interpretability garnish; it is the structural prior that makes classification clinically usable,
- the method reports auditable evidence maps and reliability metrics, not only accuracy.

## Target Journal Ladder

1. **Primary target: Journal of Dentistry**
   - Best fit if the paper shows a clinically relevant digital workflow, rigorous validation, and clear practical benefit.
   - The journal already publishes AI and digital-dentistry work, including intraoral-scan analysis.
2. **Parallel target: Journal of Prosthetic Dentistry**
   - Strong thematic fit because restoration typing is directly relevant to prosthodontic and restorative workflows.
   - Especially suitable if the final story emphasizes restorative-status screening and scan-based chairside support.
3. **Stretch target: Journal of Dental Research**
   - Only realistic if the final package adds stronger clinical validation, external robustness, and a sharper translational narrative.

## Why This Topic Is Publishable

- Restoration classification is already a clinically meaningful endpoint in 2D imaging AI.
- Recent dental AI papers in high-level venues are actively using digital impressions / intraoral scans.
- What still appears underexplored, based on the current search set, is restoration-type classification from **full-case 3D scans** with explicit point-level localization.

## Paper Story for Dental Reviewers

### Clinical Problem

Restoration status is not encoded in a tiny cropped object in real workflow. Dentists inspect a full-case scan, mentally localize the relevant region, and then infer the restoration type. The model should mirror that pipeline rather than pretending the crop is given.

### Technical Thesis

The paper should argue that classification and segmentation must be learned together because:

- classification without localization underuses the scan,
- localization without case-level typing is clinically incomplete,
- the key engineering problem is not just segmentation quality, but how stable restoration evidence is transferred into the final case-level decision.

### Minimum Novelty Claim

Inference from the current repo evidence and literature sweep:

- this is a restoration-type classification task on full-case 3D point clouds,
- it uses explicit restoration localization as a learned evidence prior,
- it evaluates both predictive performance and evidence reliability.

That is a stronger dental-journal story than claiming a marginally better point-cloud backbone.

## Evidence the Final Paper Must Show

1. **Localization matters**
   - `gtseg_pointnet` must remain clearly above `all_pointnet`.
2. **The deployable joint model is not just a weak proxy**
   - the best promoted joint variant must close a meaningful part of the gap to oracle localization.
3. **The localization is clinically inspectable**
   - include qualitative restoration heatmaps / segmented regions.
4. **The model is reliable**
   - include calibration, source-wise breakdown, and failure examples.
5. **The claim is stable**
   - keep k-fold paired statistics, not only a single split.

## What High-Level Dental Journals Will Likely Reject

- a paper that reads like a generic computer-vision architecture sweep,
- a method with weak or inconsistent gains over the clinically motivated oracle baseline,
- a story that focuses on segmentation accuracy while classification remains unstable,
- a lack of reliability analysis or cross-source breakdown.

## Immediate Next Method Direction

The next method iteration should focus on **localization-consistent evidence transfer**:

- deployment branch uses predicted restoration localization,
- teacher branch uses GT localization during training,
- consistency is enforced between the two classification views,
- reporting includes classification, segmentation, and reliability metrics.

This is the simplest next step that is both technically grounded in the current failure analysis and easy to explain to dental reviewers.

## Reference Signals

- Automated classification of dental restorations on panoramic radiographs using deep learning.
  - https://pubmed.ncbi.nlm.nih.gov/34856747/
- Automated tooth detection and numbering in digital impressions of children using artificial intelligence.
  - https://pubmed.ncbi.nlm.nih.gov/39812496/
- CrossTooth: Cross-modal Vision Transformer for 3D Dental Model Segmentation with Image-guided Training.
  - https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CrossTooth_Cross-modal_Vision_Transformer_for_3D_Dental_Model_Segmentation_with_CVPR_2025_paper.html
