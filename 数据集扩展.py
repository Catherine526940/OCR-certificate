import os
import random
from PIL import Image, ImageEnhance

# 定义图像处理函数
def image_augmentation(image_path, output_folder):
    # 打开图像文件
    with Image.open(image_path) as img:

        # # 随机旋转图像（角度范围在-15到15之间）
        # rotate_angle = random.randint(-15, 15)
        # rotated_img = img.rotate(rotate_angle, expand=True)


        # 随机平移图像（平移范围在-20到20之间）
        shift_x = random.randint(-5, 5)
        shift_y = random.randint(-5, 5)
        shifted_img = img.transform(img.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))

        #随机增强图像的亮度和对比度
        #对比度
        enhancer = ImageEnhance.Brightness(shifted_img)
        bright_factor = random.uniform(0.5, 1.5)
        bright_img = enhancer.enhance(bright_factor)

        #亮度
        enhancer = ImageEnhance.Contrast(bright_img)
        contrast_factor = random.uniform(0.5, 1.5)
        contrast_img = enhancer.enhance(contrast_factor)

        # 保存处理后的图像
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        contrast_img.save(output_path)

# 指定图像文件夹和输出文件夹
input_folder = "D:/Users/95159/Desktop/1"
output_folder = "D:/Users/95159/Desktop/6"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 对文件夹中的每张图像进行增强处理
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image_augmentation(image_path, output_folder)

print("图像增强完成！")



