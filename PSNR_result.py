import cv2
from util import *
from skimage.metrics import structural_similarity as ssim
img_clean = cv2.imread("D:/Pycharm/Deblurring/results/im_1.png")
img_out = cv2.imread("D:/Pycharm/Deblurring/RCTLS/result1.png")
img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)/255.0
img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)/255.0
print(compare_psnr(img_clean, img_out),ssim(img_clean, img_out))
