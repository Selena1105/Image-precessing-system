import numpy as np
import scipy.stats 

# 均值
def mean(image):
    return round(np.mean(image), 2)

# 方差
def variance(image):
    return round(np.var(image), 2)

# 信息熵
def entropy(image):
    flattened_image = image.flatten()
    hist, _ = np.histogram(flattened_image, bins=256, range=[0, 256])
    normalized_hist = hist / float(np.sum(hist)) # 归一化
    entropy_value = scipy.stats.entropy(normalized_hist)
    return round(entropy_value, 2)

# 峰值信噪比
def psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse)
    return round(psnr, 2)

# 信息保留度
def ssim(original, processed):
    mu_x = np.mean(original)
    mu_y = np.mean(processed)
    sigma_x = np.std(original)
    sigma_y = np.std(processed)
    sigma_xy = np.cov(original.flatten(), processed.flatten())[0, 1]

    # 加入常数值避免分母为0
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)
    ssim = numerator / denominator

    return round(ssim, 2)

