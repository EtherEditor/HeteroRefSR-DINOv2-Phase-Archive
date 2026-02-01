import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
from cos_similarity import get_image_features  # 导入目标函数

def manual_pyramid_fusion(img_a_path, img_b_path):
    # 1. 使用 PIL 读取以解决中文路径和 WebP 兼容性
    # 对 A (1884x1884, 32bit WebP) 和 B (160x160, 24bit JPG) 进行标准转换
    a_pil = Image.open(img_a_path).convert('RGB')
    b_pil = Image.open(img_b_path).convert('RGB')

    # 转回 numpy 供 OpenCV 算法处理
    a_raw = cv2.cvtColor(np.array(a_pil), cv2.COLOR_RGB2BGR)
    b_raw = cv2.cvtColor(np.array(b_pil), cv2.COLOR_RGB2BGR)

    print(f"A 形状: {a_raw.shape}, B 形状: {b_raw.shape}")

    # 2. 空间对齐（既然死磕这一组图，手动强制对齐）
    # 将 B 放大到 A 的尺寸 (1884, 1884)
    b_resized = cv2.resize(b_raw, (a_raw.shape[1], a_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 3. 提取 A 的高频分量 (Laplacian 残差)
    # 高斯模糊的 kernel 必须是奇数，既然 A 有 1884px，kernel 要大一点才能滤掉细节
    a_low_freq = cv2.GaussianBlur(a_raw, (31, 31), 0)
    a_high_freq = cv2.subtract(a_raw, a_low_freq)
    #print(type(a_high_freq), a_high_freq.shape)

    # 4. 提取 B 的低频分量 (保留异色瞳颜色)
    b_low_freq = cv2.GaussianBlur(b_resized, (31, 31), 0)
    b_high_freq = cv2.subtract(b_resized, b_low_freq)

    # 5. 线性融合：B 的色块 + A 的细节
    #fused_img = cv2.add(b_low_freq, a_high_freq)
    #fused_img2 = cv2.add(b_low_freq, a_raw)

    # 6. 保存结果
    """
    cv2.imwrite("a_raw.jpg", a_raw)
    cv2.imwrite("b_raw.jpg", b_raw)
    cv2.imwrite("b_resized.jpg", b_resized)
    cv2.imwrite("a_low_freq.jpg", a_low_freq)
    cv2.imwrite("a_high_freq.jpg", a_high_freq)
    cv2.imwrite("b_low_freq.jpg", b_low_freq)
    cv2.imwrite("b_high_freq.jpg", b_high_freq)    # B放大后高频分量

    # ========== 全排列组合融合保存（新增，命名直观） ==========
    # 组合1：B低频 + A高频（核心融合：B色块+A细节）
    fused_b_low_a_high = cv2.add(b_low_freq, a_high_freq)
    cv2.imwrite("fused_b_low+a_high.jpg", fused_b_low_a_high)

    # 组合2：B低频 + A原图（B色块+A完整图）
    fused_b_low_a_raw = cv2.add(b_low_freq, a_raw)
    cv2.imwrite("fused_b_low+a_raw.jpg", fused_b_low_a_raw)

    # 组合3：B放大图 + A高频（B完整放大图+A细节）
    fused_b_resized_a_high = cv2.add(b_resized, a_high_freq)
    cv2.imwrite("fused_b_resized+a_high.jpg", fused_b_resized_a_high)

    # 组合4：B放大图 + A低频（B完整放大图+A低频色块）
    fused_b_resized_a_low = cv2.add(b_resized, a_low_freq)
    cv2.imwrite("fused_b_resized+a_low.jpg", fused_b_resized_a_low)

    # 组合5：A低频 + B高频（A色块+B细节）
    fused_a_low_b_high = cv2.add(a_low_freq, b_high_freq)
    cv2.imwrite("fused_a_low+b_high.jpg", fused_a_low_b_high)

    # 组合6：A原图 + B低频（A完整图+B色块）
    fused_a_raw_b_low = cv2.add(a_raw, b_low_freq)
    cv2.imwrite("fused_a_raw+b_low.jpg", fused_a_raw_b_low)

    # 组合7：A高频 + B高频（A细节+B细节）
    fused_a_high_b_high = cv2.add(a_high_freq, b_high_freq)
    cv2.imwrite("fused_a_high+b_high.jpg", fused_a_high_b_high)

    # 组合8：A低频 + B低频（A色块+B色块）
    fused_a_low_b_low = cv2.add(a_low_freq, b_low_freq)
    cv2.imwrite("fused_a_low+b_low.jpg", fused_a_low_b_low)

    # 组合9：A原图 + B放大图（A完整图+B完整放大图）
    fused_a_raw_b_resized = cv2.add(a_raw, b_resized)
    cv2.imwrite("fused_a_raw+b_resized.jpg", fused_a_raw_b_resized)
    """

    return a_low_freq, b_low_freq, a_high_freq, b_resized

def phase_alignment_fusion(img_a_path, img_b_path):

    """
    img_a: A图高像素原始数据 (1884x1884)
    img_b_resized: B图放大后的数据 (1884x1884)
    """
    a_pil = Image.open(img_a_path).convert('RGB')
    b_pil = Image.open(img_b_path).convert('RGB')

    # 转回 numpy 供 OpenCV 算法处理
    a_raw = cv2.cvtColor(np.array(a_pil), cv2.COLOR_RGB2BGR)
    b_raw = cv2.cvtColor(np.array(b_pil), cv2.COLOR_RGB2BGR)

    print(f"A 形状: {a_raw.shape}, B 形状: {b_raw.shape}")

    # 2. 空间对齐（既然死磕这一组图，手动强制对齐）
    # 将 B 放大到 A 的尺寸 (1884, 1884)
    b_resized = cv2.resize(b_raw, (a_raw.shape[1], a_raw.shape[0]), interpolation=cv2.INTER_CUBIC)
    img_a = a_raw
    img_b_resized = b_resized
    # 1. 转换为灰度图进行相位匹配
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. 计算亚像素位移 (Sub-pixel Registration)
    # 利用相位相关法计算 (dx, dy)
    (dx, dy), response = cv2.phaseCorrelate(gray_a, gray_b)
    print(f"检测到相位偏移: dx={dx:.4f}, dy={dy:.4f}")

    # 3. 构造仿射变换矩阵进行相位补偿
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_a_aligned = cv2.warpAffine(img_a, M, (img_a.shape[1], img_a.shape[0]))
    def gaussian_highpass_linear_phase(img, kernel_size=31, sigma=0):
    # 步骤1：生成高斯低通核（线性相位）
    # 注：cv2.getGaussianKernel返回的是1D核，扩展为2D后仍是对称核
      gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
      gaussian_2d = gaussian_1d @ gaussian_1d.T  # 2D高斯核（对称）
    
    # 步骤2：构造高斯高通核（线性相位，互补核）
    # 高通核 = 单位脉冲核（仅中心为1） - 低通核
      highpass_kernel = np.zeros_like(gaussian_2d)
      highpass_kernel[kernel_size//2, kernel_size//2] = 1.0  # 单位脉冲
      highpass_kernel -= gaussian_2d
    
    # 步骤3：线性相位滤波（cv2.filter2D默认用对称核，线性相位）
    # BORDER_REPLICATE避免边缘相位失真
      a_high = cv2.filter2D(img, -1, highpass_kernel, borderType=cv2.BORDER_REPLICATE)
      return a_high


# 新方法（线性相位高斯高通）：
    #a_high = gaussian_highpass_linear_phase(img_a_aligned, kernel_size=31, sigma=0)

    # 4. 在相位补偿后的空间重新进行拉普拉斯融合
    a_low = cv2.GaussianBlur(img_a_aligned, (31, 31), 0)
    a_high = cv2.subtract(img_a_aligned, a_low)
    b_low = cv2.GaussianBlur(img_b_resized, (31, 31), 0)
    cv2.imwrite("a_high_phase_aligned.jpg", a_high)
    
    return cv2.add(a_low, a_high)

def get_foreground_mask(img_pil):
    # 如果是 32 位 WebP，直接提取 Alpha 通道作为 Mask
    if img_pil.mode == 'RGBA':
        _, _, _, alpha = img_pil.split()
        return np.array(alpha) / 255.0
    else:
        # 如果没有 Alpha，用简单的阈值处理 (因为你的背景是纯白)
        gray = np.array(img_pil.convert('L'))
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        return mask / 255.0

def semantic_weighted_fusion(features_A, features_B, img_A_high, img_B_low):
    """
    features_A/B: DINOv2 提取的特征 [1, num_patches, 384]
    img_A_high: A图的高频分量 [1884, 1884, 3]
    img_B_low: B图放大后的低频分量 [1884, 1884, 3]
    """
    # 1. 计算局部相似度图 (Patch-level Similarity)
    N_A = features_A.shape[1] - 1 
    side_A = int(np.ceil(np.sqrt(N_A))) # 动态计算 patch 边长，比如 16 或 20
    N_B = features_B.shape[1] - 1 
    side_B = int(np.ceil(np.sqrt(N_B))) # 动态计算 patch 边长，比如 16 或 20

    f_A = F.normalize(features_A[:, 1:, :], dim=-1).reshape(1, side_A, side_A, 384).permute(0, 3, 1, 2)
    f_B = F.normalize(features_B[:, 1:, :], dim=-1).reshape(1, side_B, side_B, 384).permute(0, 3, 1, 2)
    
    # 计算局部余弦相似度并上采样到 1884x1884
    sim_map = torch.sum(f_A * f_B, dim=1, keepdim=True)
    
    #2026.1.27晚以以下代码得到结果:无clip semantic_weighted_fusion2.jpg;有clip semantic_weighted_fusion.jpg

    # 2. 权重平滑与上采样
    # 将 16x16 的权重图线性插值到 1884x1884，作为融合 Mask
    # 使用 Sigmoid 激活增强权重的判别力
    weight_mask = F.interpolate(sim_map, size=(1884, 1884), mode='bilinear', align_corners=False)
    weight_mask = torch.sigmoid((weight_mask - 0.9) * 10) # 强行抑制低相似度区域

    # 3. 执行加权语义融合 (Semantic-driven Blend)
    # 核心：将 (1884, 1884) 转换为 (1884, 1884, 1) 以便在 3 个通道上广播
    mask_np = weight_mask.squeeze().cpu().numpy()[:, :, np.newaxis]
    
    # 确保 img_A_high 和 img_B_low 是浮点数以防溢出
    # Result = B_low + A_high * weight_mask
    fused_res = img_B_low.astype(np.float32) + img_A_high.astype(np.float32) * mask_np

    # 必须 clip 到 [0, 255] 并转回 uint8 才能用 cv2 保存
    #return np.clip(fused_res, 0, 255).astype(np.uint8)

    return fused_res.astype(np.uint8)

# 定义两张待对比的图片路径
img1_path = "./data/a_raw.webp"
img2_path = "./data/b_raw.jpg"

# 提取两张图片的CLS特征、patch特征、原图【修改6】接收返回的原图
last_hidden_states1, cls1, patch1, num_p1, img1 = get_image_features(img1_path)
last_hidden_states2, cls2, patch2, num_p2, img2 = get_image_features(img2_path)

a_low_freq, b_low_freq, a_high_freq, b_resized=manual_pyramid_fusion(img1_path,img2_path)
#cv2.imwrite("phase_alignment_result2.jpg", phase_alignment_fusion(img1_path, img2_path))

cv2.imwrite("./results/semantic_weighted_fusion2.jpg",semantic_weighted_fusion(last_hidden_states1, last_hidden_states2, a_high_freq, b_low_freq))