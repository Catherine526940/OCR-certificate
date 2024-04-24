import cv2

class PreprocessConfig:
    def __init__(self):
        self.use_preprocess = False  # 是否使用预处理

        self.use_threshold = False  # 是否使用动态二值化方法
        self.threshold_method = cv2.ADAPTIVE_THRESH_MEAN_C  # 使用哪种二值化方法，mean还是gaussian
        self.threshold_kernel = 31  # 二值化的邻域大小
        self.threshold_C = 5  # 二值化的偏移值调整量

        self.dilate_first = False  # 膨胀先做还是侵蚀先做，膨胀是让白的更多，侵蚀是让黑的更多
        self.dilate_kernel_size = 1  # 膨胀的核大小
        self.dilate_time = 1  # 膨胀次数
        self.erode_kernel_size = 1  # 侵蚀的核大小
        self.erode_time = 1  # 侵蚀次数

        self.use_filter = False  # 是否使用滤波算法（去噪声）
        self.filter_type = "Gaussian"  # 使用哪种滤波，gaussian还是mean
        self.filter_kernel_size = 5  # 滤波的核大小

        self.use_edge_sharpen = False  # 是否使用锐化，使用了就是拉普拉斯锐化。注意边缘锐化会添加噪声。
        self.edge_sharpen_kernel = 3  # 锐化的核大小

    def setConfig(self, config):
        self.use_preprocess = config["usePreprocess"]
        print(self.use_preprocess)
        self.use_threshold = config["useBinary"]
        self.dilate_first = config["dilateFirst"]
        self.dilate_kernel_size = int(config["dilateKernel"])
        self.erode_kernel_size = int(config["erodeKernel"])
        self.use_filter = config["useFilter"]

config = PreprocessConfig()







