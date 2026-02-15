# HeteroRef-SR: Semantic-driven Texture Transfer for Heterochromia
Reference: https://github.com/XPixelGroup/BasicSR.git Sha:8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a
https://github.com/caojiezhang/DATSR.git SHA:32240b58d2c62b521ae390d0af31a1ed61316c80
## 🛠 Research Status: Archived (Feb 2026)
Experimental archive focused on reconstructing textures in low-res (160px) heterochromia images via high-res (1884px) reference. Project confirmed the feasibility of zero-shot semantic alignment but identified phase shift bottlenecks in extreme scale-up tasks.

---

## 🚀 Technical Methodology

* **Feature Backbone**: `dinov2-small` (ViT-S/14) for dense semantic indexing.
* **Alignment**: Hybrid **Phase Correlation** (FFT) + **Cosine Similarity**.
* **Fusion Logic**: Semantic-weighted Laplacian Pyramid blend.

The core fusion weight follows:
$$I_{fused} = I_{B\_low} + W_{sim} \cdot I_{A\_high}$$

---

## 📊 Experimental Diagnosis (Failure Analysis)

### 1. Background Signal Anomaly
Quantitative analysis detected a similarity drop to **0.3262** in non-semantic regions. This is attributed to JPEG ringing artifacts in the 160px source causing non-coherent noise.

### 2. Architectural Anomaly (392 Tokens)
Discovered that $224 \times 224$ input generates **392 patches** (14x28 grid) instead of the standard 256. Addressed via dynamic reshaping: `view(1, 14, 28, 384)`.

---

---
## 📝 Post-Mortem: Research Findings
**Archive Summary:** `unresolved high-freq texture phase shift and low-freq color fidelity loss under 11.7x scale disparity`

### Failure Analysis:
1. **Phase Incoherence**: Under extreme scale disparity (1884px vs 160px), the global phase correlation failed to align micro-textures, leading to visible high-frequency artifacts.
2. **Color Fidelity**: Deterministic Laplacian blending cannot compensate for the massive entropy loss in the 11.7x upsampled source, resulting in poor low-frequency color restoration.
3. **Architectural Insight**: DINOv2-small surprisingly yielded a 14x28 (392) patch grid instead of a symmetric 256 patches, requiring dynamic reshaping in the feature extraction pipeline.
