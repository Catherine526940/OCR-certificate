import cv2

use_preprocess = False  # 是否使用预处理

use_threshold = True  # 是否使用动态二值化方法
threshold_method = cv2.ADAPTIVE_THRESH_MEAN_C  # 使用哪种二值化方法，mean还是gaussian
threshold_kernel = 31  # 二值化的邻域大小
threshold_C = 5  # 二值化的偏移值调整量

dilate_first = True  # 膨胀先做还是侵蚀先做，膨胀是让白的更多，侵蚀是让黑的更多
dilate_kernel_size = 1  # 膨胀的核大小
dilate_time = 1  # 膨胀次数
erode_kernel_size = 1  # 侵蚀的核大小
erode_time = 1  # 侵蚀次数

use_filter = True  # 是否使用滤波算法（去噪声）
filter_type = "Gaussian"  # 使用哪种滤波，gaussian还是mean
filter_kernel_size = 5  # 滤波的核大小

use_edge_sharpen = False  # 是否使用锐化，使用了就是拉普拉斯锐化。注意边缘锐化会添加噪声。
edge_sharpen_kernel = 3  # 锐化的核大小

def setConfig(config):
    use_preprocess = config["use_preprocess"]
    use_threshold = config["use_threshold"]
    dilate_first = config["dilate_first"]
    dilate_kernel_size = config["dilate_kernel_size"]
    erode_kernel_size = config["erode_kernel_size"]
    use_filter = config["use_filter"]





