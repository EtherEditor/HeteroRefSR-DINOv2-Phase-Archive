from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches  
import os

# 1. 动态获取脚本所在目录，确保相对路径永远有效
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 假设你的目录结构是 scripts/cos_similarity.py，而 weights 在根目录
local_model_dir = os.path.join(BASE_DIR, "..", "weights", "dinov2_small")
# 2. 增加路径存在性检查，防止再次报出难懂的 HF 错误
if not os.path.exists(local_model_dir):
    raise FileNotFoundError(f"❌ 找不到模型文件夹: {os.path.abspath(local_model_dir)}")

# 3. 加载模型
processor = AutoImageProcessor.from_pretrained(local_model_dir, local_files_only=True)
model = AutoModel.from_pretrained(local_model_dir, local_files_only=True)
model.eval()  # 【修改2】取消注释，启用推理模式

# 封装特征提取函数（同时返回CLS特征、patch特征、原图）
def get_image_features(image_path):
    """
    输入图片路径，输出：
    - cls_feat: 全局CLS特征 (384,)
    - patch_feats: patch特征 (num_patches, 384)
    - num_patches: patch数量（如224×224→256）
    - image: 原图（PIL对象，用于可视化）【修改3】新增返回原图
    """
    try:
        image = Image.open(image_path).convert("RGB")  # 读取原图
        inputs = processor(images=image, return_tensors="pt")
        print(type(inputs['pixel_values']), inputs['pixel_values'].shape)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state  # [1, num_patches+1, 384]
        
        # 提取CLS特征（全局）和patch特征（局部）
        cls_feat = last_hidden_states[:, 0, :].cpu().numpy().squeeze()  # (384,)
        patch_feats = last_hidden_states[:, 1:, :].cpu().numpy().squeeze()  # (num_patches, 384)
        num_patches = patch_feats.shape[0]
        
        return last_hidden_states, cls_feat, patch_feats, num_patches, image  # 【修改4】返回原图
    except Exception as e:
        print(f"提取{image_path}特征失败：{e}")
        return None, None, None, None  # 【修改5】对应返回None

# 定义两张待对比的图片路径
img1_path = "./data/a_raw.webp"
img2_path = "./data/b_raw.jpg"

# 提取两张图片的CLS特征、patch特征、原图【修改6】接收返回的原图
last_hidden_states1, cls1, patch1, num_p1, img1 = get_image_features(img1_path)
last_hidden_states2, cls2, patch2, num_p2, img2 = get_image_features(img2_path)

# 检查特征提取是否成功，且patch数量一致
if cls1 is None or cls2 is None:
    print("特征提取失败，请检查图片路径！")
    exit()
if num_p1 != num_p2:
    print(f"两张图片的patch数量不一致（{num_p1} vs {num_p2}），请统一预处理尺寸！")
    exit()
num_patches = num_p1
grid_size = int(np.sqrt(num_patches))  # 224×224→16
patch_size_px = 224 // grid_size      # 每个patch的像素尺寸：14px（224/16）
print(f"✅ Patch网格尺寸：{grid_size}×{grid_size}，每个Patch像素：{patch_size_px}×{patch_size_px}")

# 计算逐patch相似度
patch_similarities = np.array([
    cosine_similarity([patch1[i]], [patch2[i]])[0][0] 
    for i in range(num_patches)
])

# 定位相似度最小的Patch
min_sim_value = np.min(patch_similarities)  # 最小相似度数值
min_sim_idx = np.argmin(patch_similarities) # 最小相似度patch的索引
# 转换为网格位置（行、列）：索引=行×grid_size + 列
min_sim_row = min_sim_idx // grid_size      
min_sim_col = min_sim_idx % grid_size       
# 转换为原图像素位置（左上角坐标）
min_sim_x = min_sim_col * patch_size_px     
min_sim_y = min_sim_row * patch_size_px     

# 输出最小相似度Patch的详细信息
print("\n===== 相似度最小的Patch信息 =====")
print(f"最小相似度数值：{min_sim_value:.4f}")
print(f"最小相似度Patch索引：{min_sim_idx}（总共有{num_patches}个Patch）")
print(f"网格位置（行/列）：第{min_sim_row+1}行 × 第{min_sim_col+1}列（从1开始计数）")
print(f"原图像素位置（左上角）：x={min_sim_x}, y={min_sim_y}（像素）")
print(f"Patch覆盖区域：({min_sim_x}, {min_sim_y}) → ({min_sim_x+patch_size_px}, {min_sim_y+patch_size_px})")

# 可视化：标注最小相似度Patch（热力图+原图）
fig = plt.figure(figsize=(12, 10))

# 子图1：Patch相似度热力图（标注最小Patch）
ax1 = plt.subplot(2, 2, 1)
im = ax1.imshow(patch_similarities.reshape(grid_size, grid_size), cmap='RdYlGn', vmin=-1, vmax=1)
# 画方框标注最小Patch
rect1 = patches.Rectangle(
    (min_sim_col - 0.5, min_sim_row - 0.5), 1, 1, 
    linewidth=2, edgecolor='black', facecolor='none', label='最小相似度Patch'
)
ax1.add_patch(rect1)
# 标注数值和位置
ax1.text(
    min_sim_col, min_sim_row, f"最小\n{min_sim_value:.4f}", 
    ha='center', va='center', color='black', fontweight='bold', fontsize=8
)
plt.colorbar(im, ax=ax1, label='Patch余弦相似度')
plt.title('Patch相似度热力图（黑色框=最小相似度Patch）')
plt.axis('off')
plt.legend()

# 子图2：第一张图（标注最小Patch对应区域）
ax2 = plt.subplot(2, 2, 2)
img1_resized = img1.resize((224, 224))  # 缩放到预处理尺寸
ax2.imshow(img1_resized)
# 画方框标注最小Patch对应的像素区域
rect2 = patches.Rectangle(
    (min_sim_x, min_sim_y), patch_size_px, patch_size_px,
    linewidth=2, edgecolor='red', facecolor='none', label='最小相似度区域'
)
ax2.add_patch(rect2)
# 标注坐标
ax2.text(
    min_sim_x + 5, min_sim_y - 5, f"({min_sim_x},{min_sim_y})", 
    color='red', fontweight='bold'
)
plt.title(f'图片1：{img1_path.split("\\")[-1]}')
plt.axis('off')
plt.legend()

# 子图3：第二张图（标注最小Patch对应区域）
ax3 = plt.subplot(2, 2, 3)
img2_resized = img2.resize((224, 224))
ax3.imshow(img2_resized)
rect3 = patches.Rectangle(
    (min_sim_x, min_sim_y), patch_size_px, patch_size_px,
    linewidth=2, edgecolor='red', facecolor='none', label='最小相似度区域'
)
ax3.add_patch(rect3)
ax3.text(
    min_sim_x + 5, min_sim_y - 5, f"({min_sim_x},{min_sim_y})", 
    color='red', fontweight='bold'
)
plt.title(f'图片2：{img2_path.split("\\")[-1]}')
plt.axis('off')
plt.legend()

# 子图4：相似度统计（补充）
ax4 = plt.subplot(2, 2, 4)
ax4.hist(patch_similarities, bins=20, color='skyblue', edgecolor='black')
ax4.axvline(min_sim_value, color='red', linestyle='--', label=f'最小相似度：{min_sim_value:.4f}')
ax4.set_xlabel('Patch相似度')
ax4.set_ylabel('Patch数量')
ax4.set_title('Patch相似度分布')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./results/最小相似度Patch标注.jpg", dpi=150, bbox_inches='tight')
plt.show()

# 可选：输出相似度最小的前5个Patch
print("\n===== 相似度最小的前5个Patch =====")
top5_min_idx = np.argsort(patch_similarities)[:5]  # 升序排列，取前5
for i, idx in enumerate(top5_min_idx):
    row = idx // grid_size
    col = idx % grid_size
    print(f"第{i+1}小：索引{idx} → 行{row+1}列{col+1} → 相似度{patch_similarities[idx]:.4f}")