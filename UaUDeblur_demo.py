
from collections import namedtuple
from net import *
import scipy.io as scio
from net.losses import *
from net.noise import *
from utils.image_io import *
from utils.common_utils import *
from DCT import *
from SSIM import SSIM
from util import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
Result = namedtuple("Result", ['recon', 'psnr'])

class DeepResPrior(object):
    def __init__(self, blurry_filename, clean_filename , num_iter = 3000, load_pretrain = 1):

        self.input_depth = 1
        self.pad = 'reflection'
        self.dtype = torch.cuda.FloatTensor
        self.Lambda1 = 5e-7
        self.Lambda2 = 5e-5
        self.out_avg = 0
        self.best_result = None
        self.current_result = None
        self.L = 0.1
        self.current_result_av = None
        self.exp_weight = 0.99
        self.best_result_av = None
        self.num_iter = num_iter
        self.x_temp = None
        self.PSNR=[]
        self.MSE=[]
        self.load_pretrain = load_pretrain
        img_blurred =  cv2.imread(blurry_filename)
        img_blurred = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
        img_clean = cv2.imread(clean_filename)
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
        imgfblur = cv2.imread(clean_filename)
        imgfblur = cv2.cvtColor(imgfblur, cv2.COLOR_BGR2GRAY)
        self.img_blurred_np = img_blurred/255.0
        self.imgfblur_np = imgfblur/255.0
        self.img_clean_np = img_clean/255.0
        mat = scio.loadmat('K1.mat')
        w_mat = scio.loadmat('w1.mat')

        self.PSF= torch.real(np_to_torch(mat['PSF'])).unsqueeze(0).type(self.dtype)
        self.img_blurred_torch = np_to_torch(self.img_blurred_np).unsqueeze(0).type(self.dtype)
        self.img_out_np = torch.zeros(1, 1, img_blurred.shape[0], img_blurred.shape[1]).type(self.dtype)
        self.res_input = get_noise(self.input_depth, method='2D', spatial_size=(img_blurred.shape[0], img_blurred.shape[1])).type(self.dtype)
        self.img_input = get_noise(self.input_depth, method='2D',
                                   spatial_size=(img_blurred.shape[0], img_blurred.shape[1])).type(self.dtype)

        self.w = torch.zeros(1, 1, self.res_input.shape[2], self.res_input.shape[3]).type(self.dtype)
        self.r = np_to_torch(w_mat['w']).unsqueeze(0).type(self.dtype)
        self.res_input_vec = self.res_input.reshape(-1,1,self.res_input.shape[2]*self.res_input.shape[3]).type(self.dtype)
        self.s = torch.zeros(1, 1, self.res_input.shape[2], self.res_input.shape[3]).type(self.dtype)
        self.o = torch.zeros(1, 1, self.res_input.shape[2], self.res_input.shape[3]).type(self.dtype)

        self.img_outt_torch = np_to_torch(self.imgfblur_np).unsqueeze(0).type(self.dtype)
        self.image_net = skip(self.input_depth, 1,
                    num_channels_down = [128, 128, 128, 128, 128],
                    num_channels_up   = [128, 128, 128, 128, 128],
                    num_channels_skip = [16, 16, 16, 16, 16],
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad=self.pad, act_fun='LeakyReLU').type(self.dtype)



        self.residual_net = skip(self.input_depth, 1,
                                 num_channels_down=[16, 16, 32, 64, 64],
                                 num_channels_up=[16, 16, 32, 64, 64],
                                 num_channels_skip=[4, 4, 4, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=False, need_bias=True, pad=self.pad, act_fun='LeakyReLU').type(self.dtype)

        # self.residual_net = fcn(self.res_input.shape[2]*self.res_input.shape[3],self.res_input.shape[2]*self.res_input.shape[3], num_hidden=[200, 200]).type(self.dtype)
        # Losses
        self.mse = torch.nn.MSELoss().type(self.dtype)
        self.sploss = SPLoss().type(self.dtype)
        self.ssim = SSIM().type(self.dtype)
        self.l1 = torch.nn.L1Loss().type(self.dtype)
        self.tv = TVLoss().type(self.dtype)

        if self.load_pretrain :
            print('-------pretrain model loaded--------')
            self.residual_net.load_state_dict(torch.load('./output/res_net_paras12.pth'))

        else:
            print('--------pretrain model unloaded--------')
        self.p1 = self.image_net.parameters()
        self.p2 = self.residual_net.parameters()
    def sub(self):
        tmp = self.img_blurred_torch - self.img_est_torch - self.w/self.L
        self.s = torch.max(self.o, tmp -  self.Lambda1) + torch.min(self.o,tmp + self.Lambda1)

    def optimize(self):


        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if self.load_pretrain:
            optimizer = torch.optim.Adam([{'params':self.p1},{'params':self.p2,'lr':5e-8}],lr=self.learning_rate) # for the sake of accuracy, with a lower lr. #5e-8
        else:
            optimizer = torch.optim.Adam([{'params': self.p1, 'lr':5e-4,}, {'params': self.p2, 'lr': 5e-4}])

        # Optimization
        for i in range(self.num_iter + 1):
            optimizer.zero_grad()
            self.closure(i)
            self._obtain_current_result(i)
            self._plot_closure(i)
            optimizer.step()



    def closure(self, step):

        if step > 2:
            self.x_temp = self.img_out_torch.detach()
        self.img_out_torch = self.image_net(self.img_input)
        self.w = self.residual_net(self.res_input)
        self.img_est_torch = torch.real(torch.fft.ifft2(torch.fft.fft2(self.img_out_torch) * self.PSF))

        #
        if step > 0:
            self.sub()

        self.img_est_np = torch_to_np(self.img_est_torch).squeeze()


        self.total_loss =   self.mse(self.img_est_torch + self.s +  self.w,
                                   self.img_blurred_torch)  + self.Lambda1* self.sploss(dct_2d(self.s))  + self.Lambda2* self.sploss(self.w)+2*self.tv(self.img_out_torch)


        self.total_loss.backward()

        self.img_out_temp = self.img_out_torch.detach()
        self.img_out_np = torch_to_np(self.img_out_torch).squeeze()
        self.out_avg = self.out_avg * self.exp_weight + self.img_out_np * (1 - self.exp_weight)
        # if self.gradterm:
        #     self.total_loss = self.mse(self.img_blurred - self.blurred_est, self.res_out) + self.lambda3 * self.l1(
        #         2 * self.res_out, self.res_out) /
        #                       + self.lambda2 * self.mse(self.res_out, self.grad)
        # else:

    def _obtain_current_result(self, step):
        self.psnr = compare_psnr(self.img_clean_np, self.img_out_np)
        self.PSNR.append(self.psnr)

        if self.load_pretrain:
            scio.savemat('PSNRTrain.mat',{'trained':self.PSNR})
        else:
            scio.savemat('PSNRUntrain.mat', {'untrained': self.PSNR})
        self.psnr_av = compare_psnr(self.img_clean_np, self.out_avg)
        scio.savemat('./output/img_res.mat', {'w': torch_to_np(self.w).squeeze()})
        self.current_result = Result(recon=self.img_out_np, psnr=self.psnr)
        self.current_result_av = Result(recon=self.out_avg, psnr=self.psnr_av)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result
            self.best_ssim = ssim(self.img_clean_np, self.img_out_np)
            self.res_out_np = torch_to_np(self.s).squeeze()
            self.img_est_np = torch_to_np(self.img_est_torch).squeeze()
            scio.savemat('./output/img_art.mat', {'h': self.res_out_np})
            scio.savemat('./output/img_out.mat', {'x': self.img_out_np})
            scio.savemat('./output/img_out_av.mat',{'xav':self.out_avg})
            torch.save(self.image_net.state_dict(),'./output/img_net_paras.pth')
        if self.best_result_av is None or self.best_result_av.psnr < self.current_result_av.psnr:
            self.best_result_av = self.current_result_av

    def _plot_closure(self, step):
        """

         :param step: the number of the iteration

         :return:
         """
        if step % 10 ==0:
            print(
                'Iteration %05d  tol_loss %f    current_psnr: %f  max_psnr %f  best_ssim %f current_psnr_av: %f max_psnr_av: %f' % (
                step, self.total_loss.item(),
                self.current_result.psnr, self.best_result.psnr, self.best_ssim,
                self.current_result_av.psnr, self.best_result_av.psnr), '\r')

if __name__ == "__main__":
    blurry_filename = "blurry.png"
    clean_filename = "clean.png"
    DRP = DeepResPrior(blurry_filename, clean_filename, num_iter = 2050, load_pretrain = 0)
    DRP.optimize()
