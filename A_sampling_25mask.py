from random import betavariate
import sys
import numpy as np
from skimage import io, color
import cv2
sys.path.append('../..')
import functools
import numpy.fft as fft
import matplotlib.pyplot as plt
import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim, \
    mean_squared_error as compare_mse
import odl
import glob
import pydicom
from cv2 import imwrite, resize
from func_test import WriteInfo
from scipy.io import loadmat, savemat
from radon_utils import (create_sinogram, bp, filter_op,
                         fbp, reade_ima, write_img, sinogram_2c_to_img,
                         padding_img, unpadding_img, indicate)
from time import sleep
from models.DWT_IDWT_layer import DWT_1D, DWT_2D, IDWT_1D, IDWT_2D
import odl
import imageio
Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
Fan_detector_partition = odl.uniform_partition(-360, 360, 720)
Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition, src_radius=500, det_radius=500)
Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)
Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)
Fan_filter = odl.tomo.fbp_filter_op(Fan_ray_trafo)

_CORRECTORS = {}
_PREDICTORS = {}

def set_predict(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'EulerMaruyamaPredictor'
    elif num == 2:
        return 'ReverseDiffusionPredictor'

def set_correct(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'LangevinCorrector'
    elif num == 2:
        return 'AnnealedLangevinDynamics'

def padding_img(img):
    b, w, h = img.shape
    h1 = 768
    tmp = np.zeros([b, h1, h1])
    x_start = int((h1 - w) // 2)
    y_start = int((h1 - h) // 2)
    tmp[:, x_start:x_start + w, y_start:y_start + h] = img
    return tmp

def unpadding_img(img):
    b, w, h = img.shape[0], 720, 720
    h1 = 768
    tmp = np.zeros([b, h1, h1])
    x_start = int((h1 - w) // 2)
    y_start = int((h1 - h) // 2)
    return img[:, x_start:x_start + w, y_start:y_start + h]

def dwt_data(data):
    dwt = DWT_2D("haar")
    xll, xlh, xhl, xhh = dwt(data)
    dwt_data = torch.cat([xll, xlh, xhl, xhh], dim=1)
    dwt_data = np.squeeze(dwt_data)
    return dwt_data

def iwt_data(ll, lw, hl, hh):
    iwt = IDWT_2D("haar")
    iwt_data = iwt(ll, lw, hl, hh)
    iwt_data = iwt_data.cpu().detach().numpy()
    return iwt_data

def init_ct_op(img, r):
    batch = img.shape[0]
    sinogram = np.zeros([batch, 720, 720])
    sparse_sinogram = np.zeros([batch, 720, 720])
    ori_img = np.zeros_like(img)
    for i in range(batch):
        sinogram[i, ...] = Fan_ray_trafo(img[i, ...]).data
        ori_img[i, ...] = Fan_FBP(sinogram[i, ...]).data
        t = np.copy(sinogram[i, ::r, :])
        sparse_sinogram[i, ...] = resize(t, [720, 720])
    return ori_img, sparse_sinogram.astype(np.float32), sinogram.astype(np.float32)


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def get_predictor(name):
    return _PREDICTORS[name]

def get_corrector(name):
    return _CORRECTORS[name]

def get_sampling_fn(config, sde, shape, inverse_scaler, eps):

    sampler_name = config.sampling.method  # pc
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=config.device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):

        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):

        pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


# ===================================================================== ReverseDiffusionPredictor
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


# =====================================================================

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


# ================================================================================================== LangevinCorrector
@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


# ==================================================================================================

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x

# ========================================================================================================

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)

def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)
#=========================================================================================================

def get_pc_sampler(sde_sino, sde_wavelet, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    # Create predictor & corrector update functions
    predictor_update_fn_sino = functools.partial(shared_predictor_update_fn,
                                                 sde=sde_sino,
                                                 predictor=predictor,
                                                 probability_flow=probability_flow,
                                                 continuous=continuous)
    corrector_update_fn_sino = functools.partial(shared_corrector_update_fn,
                                                 sde=sde_sino,
                                                 corrector=corrector,
                                                 continuous=continuous,
                                                 snr=snr,
                                                 n_steps=n_steps)

    predictor_update_fn_wavelet = functools.partial(shared_predictor_update_fn,
                                                    sde=sde_wavelet,
                                                    predictor=predictor,
                                                    probability_flow=probability_flow,
                                                    continuous=continuous)
    corrector_update_fn_wavelet = functools.partial(shared_corrector_update_fn,
                                                    sde=sde_wavelet,
                                                    corrector=corrector,
                                                    continuous=continuous,
                                                    snr=snr,
                                                    n_steps=n_steps)

    def pc_sampler(sino_model, wavelet_model):
        CS=[12,8,6]
        for ii in CS:
            r = ii
            views = 720 // r
            mean_psnr1 = 0
            mean_ssim1 = 0
            mean_mse1 = 0
            with torch.no_grad():
                img = np.load('./Test_CT/batch_img.npy')
                num_img = img.shape[0]
                for k in range(num_img):
                    img1 = img[k, :, :]
                    img1 = img1[None,...]
                    ori_img, sparse_sinogram, sinogram = init_ct_op(img1, r)
                    fbp_img = Fan_FBP(sparse_sinogram[0, ...]).data
                    x0_sino = (padding_img(sparse_sinogram))[None, ...].astype(np.float32)
                    x0_sino = torch.from_numpy(x0_sino).cuda()
                    sinogram_BZ = (padding_img(sinogram))[None, ...].astype(np.float32)
                    sinogram_BZ = torch.from_numpy(sinogram_BZ).cuda()
                    sinogram_WAVE = (dwt_data(sinogram_BZ))[None,...]
                    batch_size = ori_img.shape[0]
                    timesteps_sino = torch.linspace(sde_sino.T, eps, sde_sino.N, device=device)
                    timesteps_wavelet = torch.linspace(sde_wavelet.T, eps, sde_wavelet.N, device=device)
                    max_psnr1 = np.zeros(batch_size)
                    max_ssim1 = np.zeros(batch_size)
                    min_mse1= 999 * np.ones(batch_size)
                    rec_img = np.zeros_like(ori_img)
                    best_fbp = np.zeros_like(ori_img)
                    if r>25:
                        j=700
                    elif (r<25 and r>9) :
                        j=800
                    else :
                        j=1200
                    for i in range(0, j):
                        t_sino = timesteps_sino[i]
                        t_wavelet = timesteps_wavelet[i]
                        vec_t_sino = torch.ones(x0_sino.shape[0], device=t_sino.device) * t_sino
                        x01, x0_sino = predictor_update_fn_sino(x0_sino, vec_t_sino, model=sino_model)
                        x0_sino[:, 0, ::r, :] = sinogram_BZ[:, 0, ::r, :]
                        x01, x0_sino = corrector_update_fn_sino(x0_sino, vec_t_sino, model=sino_model)
                        x0_sino[:, 0, ::r, :] = sinogram_BZ[:, 0, ::r, :]
                        x0_dwt = dwt_data(x0_sino)
                        x0_dwt_ll = x0_dwt[0, :, :]
                        x0_dwt_lh = x0_dwt[1, :, :]
                        x0_dwt_hl = x0_dwt[2, :, :]
                        x0_dwt_hh = x0_dwt[3, :, :]
                        x0_dwt_ll = x0_dwt_ll[None, None, ...]
                        x0_dwt_lh = x0_dwt_lh[None, None, ...]
                        x0_dwt_hl = x0_dwt_hl[None, None, ...]
                        x0_dwt_hh = x0_dwt_hh[None, None, ...]
                        x0_dwt_ll[:, :, ::r, :] = sinogram_WAVE[:, 0, ::r, :]
                        x0_dwt_lh[:, :, ::r, :] = sinogram_WAVE[:, 1, ::r, :]
                        x0_dwt_hl[:, :, ::r, :] = sinogram_WAVE[:, 2, ::r, :]
                        x0_dwt_hh[:, :, ::r, :] = sinogram_WAVE[:, 3, ::r, :]
                        vec_t_wavelet = torch.ones(x0_dwt_lh.shape[0], device=t_wavelet.device) * t_wavelet
                        x01, x0_dwt_lh = predictor_update_fn_wavelet(x0_dwt_lh, vec_t_wavelet,
                                                                     model=wavelet_model)
                        x01, x0_dwt_hl = predictor_update_fn_wavelet(x0_dwt_hl, vec_t_wavelet,
                                                                     model=wavelet_model)
                        x01, x0_dwt_hh = predictor_update_fn_wavelet(x0_dwt_hh, vec_t_wavelet,
                                                                     model=wavelet_model)
                        x0_dwt_lh[:, :, ::r, :] = sinogram_WAVE[:, 1, ::r, :]
                        x0_dwt_hl[:, :, ::r, :] = sinogram_WAVE[:, 2, ::r, :]
                        x0_dwt_hh[:, :, ::r, :] = sinogram_WAVE[:, 3, ::r, :]
                        x01, x0_dwt_lh = corrector_update_fn_wavelet(x0_dwt_lh, vec_t_wavelet,
                                                                     model=wavelet_model)
                        x01, x0_dwt_hl = corrector_update_fn_wavelet(x0_dwt_hl, vec_t_wavelet,
                                                                     model=wavelet_model)
                        x01, x0_dwt_hh = corrector_update_fn_wavelet(x0_dwt_hh, vec_t_wavelet,
                                                                     model=wavelet_model)
                        x0_dwt_lh[:, :, ::r, :] = sinogram_WAVE[:, 1, ::r, :]
                        x0_dwt_hl[:, :, ::r, :] = sinogram_WAVE[:, 2, ::r, :]
                        x0_dwt_hh[:, :, ::r, :] = sinogram_WAVE[:, 3, ::r, :]
                        x0_sino = iwt_data(x0_dwt_ll, x0_dwt_lh, x0_dwt_hl, x0_dwt_hh)
                        x0_sino = torch.from_numpy(x0_sino).cuda()
                        x0_sino[:, 0, ::r, :] = sinogram_BZ[:, 0, ::r, :]
                        if x0_sino is not None:
                            tmp = np.squeeze(x0_sino.detach().cpu().numpy()).astype(np.float32)
                            tmp = unpadding_img(tmp[None, ...])
                            for coil in range(batch_size):
                                rec_img[coil, ...] = Fan_FBP(tmp[coil, ...]).data
                            psnr1, ssim1, mse1 = indicate(rec_img, ori_img)
                            c1 = max_psnr1 < psnr1
                            if np.sum(c1) > 0.01:
                                max_psnr1 = max_psnr1 * (1 - c1) + psnr1 * c1
                                max_ssim1 = max_ssim1 * (1 - c1) + ssim1 * c1
                                min_mse1 = min_mse1 * (1 - c1) + mse1 * c1
                                c1 = c1[..., None, None]
                                best_fbp = best_fbp * (1 - c1) + rec_img * c1
                        print("Step: {}   Views:{}  PSNR:{} SSIM:{} MSE:{}".format(i, views, np.round(psnr1[:4], 3),
                                                                                   np.round(ssim1[:5], 4),
                                                                                   np.round(1000 * mse1[:4], 3)))
                    with open('output_swarm.txt', 'a') as f:
                        print(
                            "MAX:  Views:{} PSNR:{} SSIM:{} MSE:{}".format(views, np.round(max_psnr1[:4], 3),
                                                                                 np.round(max_ssim1[:5], 4),
                                                                                 np.round(1000 * min_mse1[:4], 3)),file=f)
                    write_img(fbp_img, save_path='./Test_CT',
                              name='picNO_{}_view_{}_img.png'.format(k, views))
                    mean_psnr1 = mean_psnr1 + max_psnr1
                    mean_ssim1 = mean_ssim1 + max_ssim1
                    mean_mse1 = mean_mse1 + min_mse1
                with open('output_swarm.txt', 'a') as f:
                    print(
                        "MEAN_MAX:  Views:{} PSNR:{} SSIM:{} MSE:{}".format(views,
                                                                          np.round((mean_psnr1 / num_img)[:4],3),
                                                                          np.round((mean_ssim1 / num_img)[:5],4),
                                                                          np.round(1000 * (mean_mse1 / num_img)[:4],3)), file=f)
    return pc_sampler

def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                    rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler

