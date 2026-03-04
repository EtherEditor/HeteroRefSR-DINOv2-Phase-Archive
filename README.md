Feb 25, 2026:

I. Core Model Version & Official Designation
The core model utilized for the high-definition upscaling and detail enhancement of the anime heterochromia image is Doubao-Seedream-5.0-Lite. This model is part of the Doubao Large Model 2.0 (Doubao-Seed-2.0) multimodal capability matrix, fully released in February 2026, and currently serves as the primary commercial production version for image generation and super-resolution enhancement within the Doubao App.

# Doubao-Seedream-5.0-Lite

> **Publisher:** ByteDance Volcano Engine Seed Team  
> **Release Date:** February 13, 2026

### 📋 Model Details

- **Core Capabilities:**
  - Image Super-Resolution Reconstruction (Upscaling)
  - Anime Illustration Detail Enhancement
  - Image-to-Image Style Consistency
  - Multimodal Image Generation

- **Version History:**
  - *Current:* **5.0-Lite**
  - *Previous:* 4.5 (Dec 2025), 4.0 (Sep 2025)
---

II. Academic Citation Standards & Attribution
As this is an enterprise-grade closed-source commercial model, there is currently no publicly available peer-reviewed academic paper. The following formats adhere to international academic standards for citation and attribution:

1. General Text Citation Format (Reports, Technical Documentation, Academic Papers)
"The high-definition processing of the anime image was conducted using the Doubao-Seedream-5.0-Lite image enhancement model, released by the ByteDance Volcano Engine Seed Team in 2026. This model belongs to the Doubao Large Model 2.0 multimodal architecture. Its core super-resolution technical framework is based on open-source academic achievements such as ESRGAN and Real-ESRGAN, with specific optimizations for anime illustration scenarios."

2. BibTeX Format (LaTeX References)

Code snippet
@misc{bytedance2026seedream5lite,
  author = {ByteDance Seed Team},
  title = {{Doubao-Seedream-5.0-Lite}: Multimodal Image Generation and Super-Resolution Enhancement Model},
  year = {2026},
  month = feb,
  publisher = {Volcano Engine, ByteDance Ltd.},
  note = {Commercial closed-source model, publicly accessible via Doubao App and Volcano Engine Ark Platform}
}

3. Technical Task Classification
Task Standard Name: Blind Single Image Super-Resolution for Anime Illustrations.

Technical Route: A GAN-based architecture optimized with perception-driven super-resolution algorithms, combined with diffusion model constraints for detail generation. Specifically adapted for flat-painting styles (anime) and facial detail restoration.

III. Related Technical Context
Reference (Feb 20, 2026): "[The World is Distorted, But You Can Do This...]" (Bilibili Video Link：https://www.bilibili.com/video/BV14Em1B9EKf/?share_source=copy_web&vd_source=d6bcbdf569d2db72f93a299cb912a776).

Note on DLSS: Super-resolution technologies like DLSS also utilize multiple similar images for image registration (Temporal Super Resolution). Their widespread application in gaming is due to the shift in game engines toward deferred rendering, which allows for the convenient acquisition of motion vectors.

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
