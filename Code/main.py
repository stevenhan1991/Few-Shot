from dataio import *
import sys
import os
import torch
import random
import copy
from train import *
from omegaconf import OmegaConf
import time as clock
import yaml
from utils import *
from torch.utils.data import DataLoader
from skimage.transform import rescale, resize
from model import *
from scipy.stats import entropy
from tqdm import tqdm
from tsr import *
from skimage.exposure import match_histograms
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType,ModelMeanType, ModelVarType, get_named_beta_schedule


p = argparse.ArgumentParser()
p.add_argument('--config_file', type=str, default='vortex.yaml')
p.add_argument('--device', type=int, default=0)
p.add_argument('--mode', type=str, default='train')
p.add_argument('--num_samples', type=int, default=3)

opt = p.parse_args()


def main():

  with open(os.path.join("configs", opt.config_file), "r") as f:
    config = load_config(f,True)

  config.data.num_samples = opt.num_samples

  path, res, samples = getPath(config.data.dataset,config.data.num_samples) 
  config.data.res = res
  config.data.path = path

  config.data.training_samples = samples
  config.approach = opt.approach

  if not os.path.exists(config.model.model_path):
    os.mkdir(config.model.model_path)

  if not os.path.exists(config.model.result_path):
    os.mkdir(config.model.result_path)

  if not os.path.exists(config.model.model_path+config.task):
    os.mkdir(config.model.model_path+config.task)

  if not os.path.exists(config.model.result_path+config.task):
    os.mkdir(config.model.result_path+config.task)


  if not os.path.exists(config.model.model_path+config.task+'/{:d}/'.format(config.model.beta_schedule.train.n_timestep)):
    os.mkdir(config.model.model_path+config.task+'/{:d}/'.format(config.model.beta_schedule.train.n_timestep))

  if not os.path.exists(config.model.result_path+config.task+'/{:d}/'.format(config.model.beta_schedule.train.n_timestep)):
    os.mkdir(config.model.result_path+config.task+'/{:d}/'.format(config.model.beta_schedule.train.n_timestep))

  if config.task == 'SSR' or config.task == 'TSR':
    if not os.path.exists(config.model.model_path+config.task+'/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor)):
      os.mkdir(config.model.model_path+config.task+'/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor))

    if not os.path.exists(config.model.result_path+config.task+'/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor)):
      os.mkdir(config.model.result_path+config.task+'/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor))

    if not os.path.exists(config.model.model_path+config.task+'/{:d}/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples)):
      os.mkdir(config.model.model_path+config.task+'/{:d}/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples))

    if not os.path.exists(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples)):
      os.mkdir(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples))

    if not os.path.exists(config.model.model_path+config.task+'/{:d}/{:d}/{:d}/{:s}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling)):
      os.mkdir(config.model.model_path+config.task+'/{:d}/{:d}/{:d}/{:s}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling))

    if not os.path.exists(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/{:s}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling)):
      os.mkdir(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/{:s}/'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling))

  elif config.task == 'V2V':

    if not os.path.exists(config.model.model_path+config.task+'/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples)):
      os.mkdir(config.model.model_path+config.task+'/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples))

    if not os.path.exists(config.model.result_path+config.task+'/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples)):
      os.mkdir(config.model.result_path+config.task+'/{:d}/{:d}/'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples))

    if not os.path.exists(config.model.model_path+config.task+'/{:d}/{:d}/{:s}/'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling)):
      os.mkdir(config.model.model_path+config.task+'/{:d}/{:d}/{:s}/'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling))

    if not os.path.exists(config.model.result_path+config.task+'/{:d}/{:d}/{:s}/'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling)):
      os.mkdir(config.model.result_path+config.task+'/{:d}/{:d}/{:s}/'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling))

    if not os.path.exists(config.model.model_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2)):
      os.mkdir(config.model.model_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2))

    if not os.path.exists(config.model.result_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2)):
      os.mkdir(config.model.result_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2))

  config.device = opt.device

  torch.cuda.set_device(config.device)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


  if opt.mode == 'train':
    Volumes = VolumeDataSet(config)
    train_vol = DataLoader(Volumes,batch_size=1,shuffle=True,num_workers=1,pin_memory=True)
    print('Begin to Train Diffusion SR')
    model = TUNet(**config.model.unet)
    model = model.to(device)
    num_parms = count_parameters(model)
    print('Model Size is {:f} MB'.format(num_parms*4/(1024*1024)))
    train_diffusion(model,train_vol,config,device)
  elif opt.mode == 'inf':
    model = TUNet(**config.model.unet)
    model = model.to(device)
    if config.task == 'SSR' or config.task == 'TSR':
      model.load_state_dict(torch.load(config.model.model_path+config.task+'/{:d}/{:d}/{:d}/{:s}/{:s}_{:d}_best_model_epoch_{:d}.pth'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling,config.task,config.model.unet.ch,config.inf.epochs),map_location=device))
    elif config.task == 'V2V':
      model.load_state_dict(torch.load(config.model.model_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}/{:s}_best_model_epoch_{:d}.pth'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2,config.task,config.inf.epochs),map_location=device))
    
    timesteps = config.model.beta_schedule.inf.n_timestep
    betas = torch.tensor(np.linspace(config.model.beta_schedule.linear_start,config.model.beta_schedule.linear_end, timesteps))
      # Initialize diffusion utiities
      diff = GaussianDiffusion(betas=betas,
                               model_mean_type=ModelMeanType[config.model.diffusion.params.model_mean_type],
                               model_var_type=ModelVarType[config.model.diffusion.params.model_var_type],
                               loss_type=LossType[config.model.diffusion.params.loss_type])
      psnr_info = {'PSNR':[], 'Avg':0}

      if config.task == 'SSR' or config.task == 'V2V':

        for i in range(1,config.data.num+1):
          if config.task == 'SSR':
            gt = np.fromfile(config.data.path+config.data.var1+'/{:04d}.dat'.format(i),dtype='<f')
            gt = gt.reshape((config.data.res[2],config.data.res[1],config.data.res[0])).transpose()
            gt = 2*(gt-gt.min())/(gt.max()-gt.min())-1

            cond = resize(gt,(config.data.res[0]//config.data.factor,config.data.res[1]//config.data.factor,config.data.res[2]//config.data.factor),order=3)
            cond = resize(cond,(config.data.res[0],config.data.res[1],config.data.res[2]),order=3)
            cond = np.clip(cond,-1.0,1.0)
            
          elif config.task == 'V2V':
            gt = np.fromfile(config.data.path+config.data.var2+'/{:04d}.dat'.format(i),dtype='<f')
            gt = gt.reshape((config.data.res[2],config.data.res[1],config.data.res[0])).transpose()
            gt = 2*(gt-gt.min())/(gt.max()-gt.min())-1

            cond = np.fromfile(config.data.path+config.data.var1+'/{:04d}.dat'.format(i),dtype='<f')
            cond = cond.reshape((config.data.res[2],config.data.res[1],config.data.res[0])).transpose()
            cond = 2.0 * (cond-cond.min()) / (cond.max()-cond.min()) - 1.0

          cond = torch.from_numpy(cond).to(device)
          cond = cond[None,...]
          cond = cond[None,...]

          if config.task == 'SSR':  
            if config.data.dataset in ['Vortex','Jets','Cylinder']:
              with torch.no_grad():
                pred = diff.ddim_sample_loop(model, (1, *cond.shape[1:]), cond, clip_denoised=True).cpu().float()
                pred = pred[0][0].detach().numpy()
            elif config.data.dataset == 'Combustion':
              d = - np.ones((config.data.res[0],config.data.res[1],config.data.res[2]))
              if config.data.var1 in ['HR']:
                d = - d
              cond_ = cond[:,:,:,118:622,:,]
              pred = diff.ddim_sample_loop(model, (1, *cond_.shape[1:]), cond_, clip_denoised=True).cpu().float()
              pred = pred[0][0].detach().numpy()
              d[:,118:622:,] = pred
              pred = d
            elif config.data.dataset in ['Ionization']:
              cond_ = cond[:,:,280:600,:,:,]
              d = copy.deepcopy(cond[0][0].cpu().detach().numpy())
              pred = diff.ddim_sample_loop(model, (1, *cond_.shape[1:]), cond_, clip_denoised=True).cpu().float()
              pred = pred[0][0].detach().numpy()
              d[280:600:,:,] = pred
              pred = d
            elif config.data.dataset in ['Bubble']:
              cond_ = cond[:,:,160:480,:,:,]
              d = copy.deepcopy(cond[0][0].cpu().detach().numpy())
              pred = diff.ddim_sample_loop(model, (1, *cond_.shape[1:]), cond_, clip_denoised=True).cpu().float()
              pred = pred[0][0].detach().numpy()
              d[160:480,:,] = pred
              pred = d
          elif config.task == 'V2V':
            if config.data.dataset == 'Cylinder':
              with torch.no_grad():
                pred = diff.ddim_sample_loop(model, (1, *cond.shape[1:]), cond, clip_denoised=True).cpu().float()
                pred = pred[0][0].detach().numpy()
            elif config.data.dataset == 'Ionization':
              d = np.ones((config.data.res[0],config.data.res[1],config.data.res[2]))
              cond_ = cond[:,:,280:600,:,:,]
              pred = diff.ddim_sample_loop(model, (1, *cond_.shape[1:]), cond_, clip_denoised=True).cpu().float()
              pred = pred[0][0].detach().numpy()
              if config.data.var2 in ['H+','PD']:
                pred = - pred
              elif config.data.var2 in ['He']:
                d = - d
              d[280:600:,:,] = pred
              pred = d
            elif config.data.dataset == 'Combustion':
              d = - np.ones((config.data.res[0],config.data.res[1],config.data.res[2]))
              cond_ = cond[:,:,:,118:622,:,]
              pred = diff.ddim_sample_loop(model, (1, *cond_.shape[1:]), cond_, clip_denoised=True).cpu().float()
              pred = pred[0][0].detach().numpy()
              d[:,118:622:,] = pred
              pred = d
            elif config.data.dataset == 'Tangaroa':
              pred = concatsubvolumewdiff(model, diff, cond,[288,176,112],config)
          pred = np.clip(pred,-1,1)
          pred = np.asarray(pred,dtype='<f')
          p = PSNR(gt,pred)
          pred = pred.flatten('F')
          if config.task == 'SSR':
            pred.tofile(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/{:s}/{:04d}.dat'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling,i),format='<f')
          elif config.task == 'V2V':
            pred.tofile(config.model.result_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}/{:04d}.dat'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2,i),format='<f')
          psnr_info['PSNR'].append(float(p))
          psnr_info['Avg'] += float(p)
          print('PSNR at Time Step {:03d} is {:4f}'.format(i,p)) 
          if config.task == 'SSR':
            save_loss_info(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/{:s}/PSNR.yaml'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling), psnr_info)
          elif config.task == 'V2V':
            save_loss_info(config.model.result_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}/PSNR.yaml'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2), psnr_info)
        psnr_info['Avg'] /= config.data.num
        print('Average PSNR is {:4f}'.format(psnr_info['Avg'])) 
        if config.task == 'SSR':
          save_loss_info(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/{:s}/PSNR.yaml'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling), psnr_info)
        elif config.task == 'V2V':
          save_loss_info(config.model.result_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}/PSNR.yaml'.format(config.model.beta_schedule.train.n_timestep,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2), psnr_info)
      elif config.task == 'TSR':
        total_num = 0
        for i in range(1,config.data.num+1,config.data.factor+1):
          if (i+config.data.factor+1)<=config.data.num:
            cond = np.fromfile(config.data.path+config.data.var1+'/{:04d}.dat'.format(i),dtype='<f')
            cond = cond.reshape((config.data.res[2],config.data.res[1],config.data.res[0])).transpose()
            cond = 2*(cond-cond.min())/(cond.max()-cond.min())-1

            cond = torch.from_numpy(cond).to(device)
            cond = cond[None,...]
            cond = cond[None,...]

            cond_ = np.fromfile(config.data.path+config.data.var1+'/{:04d}.dat'.format(i+config.data.factor+1),dtype='<f')
            cond_ = cond_.reshape((config.data.res[2],config.data.res[1],config.data.res[0])).transpose()
            cond_ = 2*(cond_-cond_.min())/(cond_.max()-cond_.min())-1

            cond_ = torch.from_numpy(cond_).to(device)
            cond_ = cond_[None,...]
            cond_ = cond_[None,...]

            cond = torch.cat((cond,cond_),dim=1)

            if config.data.dataset in ['Vortex','Jets','Cylinder']:
              with torch.no_grad():
                pred = diff.ddim_sample_loop(model, (1, config.data.factor, *cond.shape[2:]), cond, clip_denoised=True).cpu().float()
                pred = pred[0].detach().numpy()
            elif config.data.dataset in ['Ionization']:
              cond_ = cond[:,:,280:600,:,:,]
              d = []
              for j in range(1,config.data.factor+1):
                w = j/(config.data.factor+1)
                d.append(w*cond[0][1] + (1-w)*cond[0][0])
              d = torch.stack(d)
              d = d.cpu().detach().numpy()
              pred = diff.ddim_sample_loop(model, (1, config.data.factor, *cond_.shape[2:]), cond_, clip_denoised=True).cpu().float()
              pred = pred[0].detach().numpy()
              d[:,280:600:,:,] = pred
              pred = d
            elif config.data.dataset in ['Bubble']:
              d = []
              for j in range(1,config.data.factor+1):
                w = j/(config.data.factor+1)
                d.append(w*cond[0][1] + (1-w)*cond[0][0])
              d = torch.stack(d)
              d = d.cpu().detach().numpy()
              cond_ = cond[:,:,160:480,:,:,]
              pred = diff.ddim_sample_loop(model, (1, config.data.factor, *cond_.shape[2:]), cond_, clip_denoised=True).cpu().float()
              pred = pred[0].detach().numpy()
              d[:,160:480:,:,] = pred
              pred = d
            elif config.data.dataset in ['Combustion']:
              d = - np.ones((config.data.factor,config.data.res[0],config.data.res[1],config.data.res[2]))
              if config.data.var1 in ['HR']:
                d = - d
              cond_ = cond[:,:,:,118:622,:,]
              pred = diff.ddim_sample_loop(model, (1, config.data.factor, *cond_.shape[2:]), cond_, clip_denoised=True).cpu().float()
              pred = pred[0].detach().numpy()
              d[:,:,118:622:,] = pred
              pred = d
            elif config.data.dataset == 'Tangaroa':
              pred = concatsubvolumewdiff(model, diff, cond,[288,176,112],config)
            pred = np.clip(pred,-1,1)
            pred = np.asarray(pred,dtype='<f')
            for j in range(1,config.data.factor+1):
              total_num += 1
              gt = np.fromfile(config.data.path+config.data.var1+'/{:04d}.dat'.format(i+j),dtype='<f')
              gt = gt.reshape((config.data.res[2],config.data.res[1],config.data.res[0])).transpose()
              gt = 2*(gt-gt.min())/(gt.max()-gt.min())-1
              p = PSNR(gt,pred[j-1])
              pred_ = pred[j-1].flatten('F')
              pred_.tofile(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/{:s}/{:04d}.dat'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling,i+j),format='<f')
              psnr_info['PSNR'].append('PSNR at Time Step {:03d} is {:4f}'.format(i+j,float(p)))
              psnr_info['Avg'] += float(p)
              print('PSNR at Time Step {:03d} is {:4f}'.format(i+j,p)) 
              save_loss_info(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/{:s}/PSNR.yaml'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling), psnr_info)
        psnr_info['Avg'] /= total_num
        print('Average PSNR is {:4f}'.format(psnr_info['Avg'])) 
        save_loss_info(config.model.result_path+config.task+'/{:d}/{:d}/{:d}/{:s}/PSNR.yaml'.format(config.model.beta_schedule.train.n_timestep,config.data.factor,config.data.num_samples,config.data.sampling), psnr_info)
  else:
    raise NotImplementedError('{:s} is not implemented!'.format(opt.mode))


if __name__== "__main__":
  main()