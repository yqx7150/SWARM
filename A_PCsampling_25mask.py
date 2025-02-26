MODEL_W = [22] # 17
for i in MODEL_W:
    with open('output_swarm.txt', 'a') as f:
        print(f"Model_{i}_start:", file=f)
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import sys
    ##################################################################
    import A_sampling_25mask as sampling
    from A_sampling_25mask import ReverseDiffusionPredictor,LangevinCorrector,AnnealedLangevinDynamics ,EulerMaruyamaPredictor,AncestralSamplingPredictor
    import aapm_sin_ncsnpp_waveletmask0 as configs_wavelet
    import aapm_sin_ncsnpp_sinomask0 as configs_sino
    ####################2##############################################
    sys.path.append('../..')
    from losses import get_optimizer
    from models.ema import ExponentialMovingAverage
    import numpy as np
    from utils import restore_checkpoint
    import models
    from models import utils as mutils
    from models import ncsnv2
    from models import ncsnpp
    from models import ddpm as ddpm_model
    from models import layerspp
    from models import layers
    from models import normalization
    from sde_lib import VESDE, VPSDE, subVPSDE
    import os.path as osp


    if len(sys.argv) > 1:
      start = int(sys.argv[1])
      end = int(sys.argv[2])

    def get_predict(num):
      if num == 0:
        return None
      elif num == 1:
        return EulerMaruyamaPredictor
      elif num == 2:
        return ReverseDiffusionPredictor

    def get_correct(num):
      if num == 0:
        return None
      elif num == 1:
        return LangevinCorrector
      elif num == 2:
        return AnnealedLangevinDynamics

    checkpoint_num = [[i, 27]] #27
    predicts = [2]
    corrects = [1]


    for predict in predicts:
      for correct in corrects:
        for check_num in checkpoint_num:
          sde = 'VESDE'
          if sde.lower() == 'vesde':
            ckpt_filename_wavelet = './exp_random3h_mask0_4000/checkpoints/checkpoint_{}.pth'.format(check_num[0])
            ckpt_filename_sino = './exp_sino_mask_4000/checkpoints/checkpoint_{}.pth'.format(check_num[1])
            print("ckpt_filename_wavelet：", ckpt_filename_wavelet)
            print("ckpt_filename_sino：", ckpt_filename_sino)
            assert os.path.exists(ckpt_filename_wavelet)
            assert os.path.exists(ckpt_filename_sino)
            configs_wavelet = configs_wavelet.get_config()
            configs_sino = configs_sino.get_config()
            sde_wavelet = VESDE(sigma_min=configs_wavelet.model.sigma_min, sigma_max=configs_wavelet.model.sigma_max, N=configs_wavelet.model.num_scales)
            sde_sino = VESDE(sigma_min=configs_sino.model.sigma_min, sigma_max=configs_sino.model.sigma_max, N=configs_sino.model.num_scales)
            sampling_eps = 1e-5

          batch_size = 1 #@param {"type":"integer"}
          configs_wavelet.training.batch_size = batch_size
          configs_wavelet.eval.batch_size = batch_size
          random_seed = 0 #@param {"type": "integer"}
          sigmas = mutils.get_sigmas(configs_wavelet)
          wavelet_model = mutils.create_model(configs_wavelet)
          optimizer = get_optimizer(configs_wavelet, wavelet_model.parameters())
          ema = ExponentialMovingAverage(wavelet_model.parameters(),
                                        decay=configs_wavelet.model.ema_rate)
          state = dict(step=0, optimizer=optimizer,
                      model=wavelet_model, ema=ema)
          state = restore_checkpoint(ckpt_filename_wavelet, state, configs_wavelet.device)
          ema.copy_to(wavelet_model.parameters())

          batch_size = 1 #@param {"type":"integer"}
          configs_sino.training.batch_size = batch_size
          configs_sino.eval.batch_size = batch_size
          random_seed = 0 #@param {"type": "integer"}
          sigmas = mutils.get_sigmas(configs_sino)
          sino_model = mutils.create_model(configs_sino)
          optimizer = get_optimizer(configs_sino, sino_model.parameters())
          ema = ExponentialMovingAverage(sino_model.parameters(),
                                        decay=configs_sino.model.ema_rate)
          state = dict(step=0, optimizer=optimizer,
                      model=sino_model, ema=ema)
          state = restore_checkpoint(ckpt_filename_sino, state, configs_sino.device)
          ema.copy_to(sino_model.parameters())

          predictor = get_predict(predict)
          corrector = get_correct(correct)

          snr = 0.03
          n_steps = 1
          probability_flow = False

          sampling_fn = sampling.get_pc_sampler(sde_sino, sde_wavelet, predictor, corrector,
                                              None, snr, n_steps=n_steps,
                                              probability_flow=probability_flow,
                                              continuous=configs_sino.training.continuous,
                                              eps=sampling_eps, device=configs_sino.device) #,radon=fanBeam,gamma=0.01

          sampling_fn(sino_model,wavelet_model)


