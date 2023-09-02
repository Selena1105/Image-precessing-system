import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from score import mean, variance,entropy, psnr, ssim 

def save_image(path, image):
    if os.path.exists(path):
    # 删除图片
        os.remove(path)
    #保存图片
    cv2.imwrite(path, image)

# 显示6段线性变换结果
class LinearTransformGUI:
    def __init__(self, image_path):
        self.window = tk.Tk()
        self.window.title("6 Segment Linear Transform GUI")
        # 设置窗口的默认字体
        self.window.option_add("*Font", ("Helvetica", 18))

        # 设置窗口大小
        window_width = 1500
        window_height = 1000
        self.window.geometry(f"{window_width}x{window_height}")

        # 读入原图并转换为灰度图
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_tk = Image.fromarray(self.image)
        # 显示灰度图
        self.image_tk = ImageTk.PhotoImage(self.image_tk)
        self.image_show = tk.Label(self.window, image=self.image_tk)
        self.image_show.place(x=-350, y=-200)
        # 设置图片大小
        image_width = window_width - 20
        image_height = window_height - 120
        self.image_show.config(width=image_width, height=image_height)

        # 计算直方图并显示
        self.histogram = self.calculate_histogram(self.image)
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(side=tk.TOP, padx=(700, 0))
        self.plot_histogram()

        # 设置滑块大小
        slider_width = 200
        slider_height = 20

        # 设置滑块标签
        label_ori = tk.Label(self.window, text="original")
        label_ori.place(x=230, y=480)
        label_new = tk.Label(self.window, text="new")
        label_new.place(x=540, y=480)

        # 设置阈值滑块-原阈值
        self.num_segments_old = 5
        self.sliders_old = []
        for i in range(1, self.num_segments_old + 1):
            slider_label = tk.Label(self.window, text=f"Threshold {i}:")
            slider_label.place(x=60, y=490 + i * 55)
            threshold_slider_old = tk.Scale(self.window, from_=0, to=255, orient=tk.HORIZONTAL)
            threshold_slider_old.place(x=175, y=470 + i * 55)
            threshold_slider_old.config(length=slider_width, width=slider_height)
            self.sliders_old.append(threshold_slider_old)

        # 设置阈值滑块-新阈值
        self.num_segments_new = 5
        self.sliders_new = []
        for i in range(1, self.num_segments_new + 1):
            threshold_slider_new = tk.Scale(self.window, from_=0, to=255, orient=tk.HORIZONTAL)
            threshold_slider_new.place(x=470, y=470 + i * 55)
            threshold_slider_new.config(length=slider_width, width=slider_height)
            self.sliders_new.append(threshold_slider_new)

        # 设置按钮
        self.apply_button = tk.Button(self.window, text="Apply", command=self.apply_transform)
        self.apply_button.place(x=270, y=850)
        self.apply_button.config(width=20, height=2)

        # 添加指标标签
        self.label_mean = tk.Label(self.window, text="Mean: %.2f" % mean(self.image))
        self.label_mean.place(x=820, y=600)
        self.label_var = tk.Label(self.window, text="Variance: %.2f" % variance(self.image))
        self.label_var.place(x=820, y=650)
        self.label_ent = tk.Label(self.window, text="Entropy: %.2f" % entropy(self.image) )
        self.label_ent.place(x=820, y=700)
        self.label_psnr = tk.Label(self.window, text="PSNR: %.2f" % psnr(self.image, self.image)  )
        self.label_psnr.place(x=820, y=750)

        self.window.mainloop()

    def apply_transform(self):
        # 执行线性变换
        thresholds_old = [threshold.get() for threshold in self.sliders_old]
        thresholds_new = [threshold.get() for threshold in self.sliders_new]
        enhanced_image = self.linear_transform(self.image, thresholds_old, thresholds_new)

        # 更新直方图
        self.histogram = self.calculate_histogram(enhanced_image)
        self.plot_histogram()

        # 更新图片
        enhanced_image_tk = ImageTk.PhotoImage(Image.fromarray(enhanced_image))
        self.image_show.configure(image=enhanced_image_tk)
        self.image_show.image = enhanced_image_tk

        # 更新指标
        self.calculate_score(enhanced_image)

        # 保存变换后图片
        save_image("enhanced_image.jpg", enhanced_image)
        

    # 六段线性变换
    def linear_transform(self, image, thresholds_old, thresholds_new):
        enhanced_image = np.zeros_like(image)
        for i in range(6):
            if i == 0:
                low_old, high_old = 0, thresholds_old[i]
                low_new, high_new = 0, thresholds_new[i]
            elif i == 5:
                low_old, high_old = thresholds_old[i-1], 255
                low_new, high_new = thresholds_new[i-1], 255
            else:
                low_old, high_old = thresholds_old[i-1], thresholds_old[i]
                low_new, high_new = thresholds_new[i-1], thresholds_new[i]

            mask = np.logical_and(image >= low_old, image <= high_old)
            k = (high_new - low_new) / (high_old - low_old + 1)
            b = low_new - low_old
            enhanced_image[mask] = image[mask] * k + b

        return enhanced_image

    # 计算直方图
    def calculate_histogram(self, image):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        # 保存直方图
        save_image("histogram.jpg", histogram)
        return histogram

    # 画直方图
    def plot_histogram(self):
        self.ax.clear()
        self.ax.bar(range(256), self.histogram.flatten(), color='blue')
        self.ax.set_xlabel('Pixel Value')
        self.ax.set_ylabel('Frequency')
        self.ax.set_title('Image Histogram')
        self.canvas.draw()
    
    # 计算指标并显示
    def calculate_score(self, enhanced_image):
        self.label_mean.config(text="Mean: %.2f" % mean(enhanced_image))
        self.label_var.config(text="Variance: %.2f" % variance(enhanced_image))
        self.label_ent.config(text="Entropy: %.2f" % entropy(enhanced_image))
        self.label_psnr.config(text="PSNR: %.2f" % psnr(self.image, enhanced_image))


# 显示8位比特图
class GrayTo8BitGUI:
    def __init__(self, image_path):
        self.window = tk.Tk()
        self.window.title("Gray to 8-bit Bitmap GUI")

        # 读取灰度图像并转为8张位图
        gray_image = Image.open(image_path).convert("L")
        eight_bit_images = self.convert_8bit_images(np.array(gray_image))

        # 显示8张位图
        self.image_labels = []
        for i, image in enumerate(eight_bit_images):
            image = Image.fromarray(image).resize((400, 200))
            image_tk = ImageTk.PhotoImage(image)
            label = tk.Label(self.window, image=image_tk)
            label.image = image_tk
            row = i // 4
            column = i % 4
            label.grid(row=row, column=column)
            self.image_labels.append(label)

        self.window.mainloop()

    # 分割为8位图像的不同层次
    def convert_8bit_images(self, image):
        layers = []
        for i in range(8):
            layer = ((image >> i) & 1) * 255
            # 保存变换后图片
            save_path="layer_"+str(i+1)+".jpg"
            save_image(save_path, layer)
            layers.append(layer)
        return layers

# 显示浮雕化效果
class EmbossedGUI:
    def __init__(self, image_path):
        self.window = tk.Tk()
        self.window.title("Embossed Effect GUI")
        # 设置窗口的默认字体
        self.window.option_add("*Font", ("Helvetica", 16))

        # 读入原图并转为灰度图
        self.img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 创建按钮
        button_sobel = tk.Button(self.window, text="Sobel", command=self.apply_sobel)
        button_sobel.pack(side=tk.LEFT, padx=10, pady=10)
        button_robert = tk.Button(self.window, text="Robert", command=self.apply_robert)
        button_robert.pack(side=tk.LEFT, padx=10, pady=10)
        button_laplacian = tk.Button(self.window, text="Laplacian", command=self.apply_laplacian)
        button_laplacian.pack(side=tk.LEFT, padx=10, pady=10)
        button_scharr = tk.Button(self.window, text="Scharr", command=self.apply_scharr)
        button_scharr.pack(side=tk.LEFT, padx=10, pady=10)
        button_prewitt = tk.Button(self.window, text="Prewitt", command=self.apply_prewitt)
        button_prewitt.pack(side=tk.LEFT, padx=10, pady=10)
        button_defined = tk.Button(self.window, text="Defined", command=self.apply_defined)
        button_defined.pack(side=tk.LEFT, padx=10, pady=10)

        # 创建标签用于显示图像
        self.label = tk.Label(self.window)
        self.label.pack()
        # 显示原始图像
        self.show_image(self.img_gray)

        # 添加指标标签
        self.label_ssim = tk.Label(self.window, text="SSIM: %.2f" % ssim(self.img_gray, self.img_gray))
        self.label_ssim.place(x=140, y=300)
        self.label_psnr = tk.Label(self.window, text="PSNR: %.2f" % psnr(self.img_gray, self.img_gray) )
        self.label_psnr.place(x=360, y=300)

        self.window.mainloop()
    
    def apply_sobel(self):
        kernelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobelx = cv2.filter2D(self.img_gray, -1, kernelx)
        sobely = cv2.filter2D(self.img_gray, -1, kernely)
        sobel = cv2.addWeighted(sobelx, 0.4, sobely, 0.4, 0)+50
        save_image("sobel.jpg", sobel)
        self.calculate_score(sobel)
        self.show_image(sobel)

    def apply_robert(self):
        robert_x = np.array([[1, 0], [0, -1]])
        robert_y = np.array([[0, 1], [-1, 0]])
        robertx = cv2.filter2D(self.img_gray, -1, robert_x)
        roberty = cv2.filter2D(self.img_gray, -1, robert_y)
        robert = cv2.addWeighted(robertx, 0.9, roberty, 0.9, 0) +50
        save_image("robert.jpg", robert)
        self.calculate_score(robert)
        self.show_image(robert)

    def apply_laplacian(self):
        laplacian = cv2.Laplacian(self.img_gray, cv2.CV_64F)
        laplacian = np.clip(128 + 1.2* laplacian, 0, 255).astype(np.uint8)
        save_image("laplacian.jpg", laplacian)
        self.calculate_score(laplacian)
        self.show_image(laplacian)

    def apply_scharr(self):
        kernelx = np.array([[3, 0, -3],[10, 0, -10],[3, 0, -3]])
        kernely = np.array([[3, 10, 3],[0, 0, 0],[-3, -10, -3]])
        scharrx = cv2.filter2D(self.img_gray, -1, kernelx)
        scharry = cv2.filter2D(self.img_gray, -1, kernely)
        scharr = cv2.addWeighted(scharrx, 0.4, scharry, 0.4, 0) +30
        save_image("scharr.jpg", scharr)
        self.calculate_score(scharr)
        self.show_image(scharr)

    def apply_prewitt(self):
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(self.img_gray, -1, kernelx)
        prewitty = cv2.filter2D(self.img_gray, -1, kernely)
        prewitt = cv2.addWeighted(prewittx, 0.45, prewitty, 0.45, 0) +50
        save_image("prewitt.jpg", prewitt)
        self.calculate_score(prewitt)
        self.show_image(prewitt)

    def apply_defined(self):
        kernel = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
        defined = cv2.filter2D(self.img_gray, -1, kernel)+30
        save_image("defined.jpg", defined)
        self.calculate_score(defined)
        self.show_image(defined)

    # 显示图片
    def show_image(self, img):
        img = cv2.resize(img, (800, 400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        if hasattr(self, 'label'):
            self.label.configure(image=img_tk)
            self.label.image = img_tk
        else:
            self.label = tk.Label(self.window, image=img_tk)
            self.label.pack()
    
    # 计算指标
    def calculate_score(self, image):
        self.label_ssim.config(text="SSIM: %.2f" % ssim(self.img_gray,image))
        self.label_psnr.config(text="PSNR: %.2f" % psnr(self.img_gray,image))

# 绘制8比特分层图的直方图
class BitHistogramGUI:
    def __init__(self, image_path):
        self.window = tk.Tk()
        self.window.title("8-Bit Planes Histograms")

        # 读取灰度图像并转为8张位图
        self.gray_image = Image.open(image_path).convert("L")
        self.eight_bit_planes = self.convert_8bit_planes(np.array(self.gray_image))

        # 创建子图
        self.fig, self.axes = plt.subplots(2, 4, figsize=(20, 8))
        self.fig.suptitle("8-Bit Planes Histograms")

        # 绘制前四张直方图
        for i, ax in enumerate(self.axes[0, :]):
            ax.clear()
            ax.hist(self.eight_bit_planes[i].flatten(), bins=256, range=(-20, 275), color='blue')
            ax.set_xlim([-20, 275])
            ax.set_ylim([0, 10000])
            ax.set_title(f"Bit Plane {i+1}")

        # 绘制后四张直方图
        for i, ax in enumerate(self.axes[1, :]):
            ax.clear()
            ax.hist(self.eight_bit_planes[i+4].flatten(), bins=256, range=(-20, 275), color='blue')
            ax.set_xlim([-20, 275])
            ax.set_ylim([0, 10000])
            ax.set_title(f"Bit Plane {i+5}")

        # 显示界面
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.window.mainloop()

    # 将灰度图像转换为8张位图
    def convert_8bit_planes(self, image):
        planes = []
        for i in range(8):
            plane = ((image >> i) & 1)*255
            planes.append(plane)
        print(planes)
        return planes


# 执行程序
image_path = 'euro.jpg'
gui = LinearTransformGUI(image_path)
enhanced_path = "enhanced_image.jpg"
gui = GrayTo8BitGUI(enhanced_path)
gui = BitHistogramGUI(enhanced_path)
gui = EmbossedGUI(image_path)