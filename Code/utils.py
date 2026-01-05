import os
import numpy as np
import sys
import tqdm
import time
import pdb
import copy
import configparser
import argparse
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import io
import torch
from torch.nn import Parameter
from omegaconf import OmegaConf
import os
import yaml

def concatsubvolumewdiff(model, diff, data, win_size, config):
    x,y,z = data.size()[2],data.size()[3],data.size()[4]
    w = np.zeros((win_size[0],win_size[1],win_size[2]))
    for i in range(win_size[0]):
        for j in range(win_size[1]):
            for k in range(win_size[2]):
                dx = min(i,win_size[0]-1-i)
                dy = min(j,win_size[1]-1-j)
                dz = min(k,win_size[2]-1-k)
                d = min(min(dx,dy),dz)+1
                w[i,j,k] = d
    w = w/np.max(w)
    if config.task == 'SSR' or config.task == 'V2V':
        avI = np.zeros((x,y,z))
        pmap= np.zeros((x,y,z))
    elif config.task == 'TSR':
        avI = np.zeros((config.data.factor,x,y,z))
        pmap= np.zeros((config.data.factor,x,y,z))
    avk = 4
    for i in range((avk*x-win_size[0])//win_size[0]+1):
        for j in range((avk*y-win_size[1])//win_size[1]+1):
            for k in range((avk*z-win_size[2])//win_size[2]+1):
                si = (i*win_size[0]//avk)
                ei = si+win_size[0]
                sj = (j*win_size[1]//avk)
                ej = sj+win_size[1]
                sk = (k*win_size[2]//avk)
                ek = sk+win_size[2]
                if ei>x:
                    ei= x
                    si=ei-win_size[0]
                if ej>y:
                    ej = y
                    sj = ej-win_size[1]
                if ek>z:
                    ek = z
                    sk = ek-win_size[2]
                d = data[:,:,si:ei,sj:ej,sk:ek]
                with torch.no_grad():
                    if config.task == 'SSR' or config.task == 'V2V':
                        result = diff.ddim_sample_loop(model, (1, *d.shape[1:]), d, clip_denoised=True).cpu().float()
                    elif config.task == 'TSR':
                        result = diff.ddim_sample_loop(model, (1, config.data.factor, *d.shape[2:]), d, clip_denoised=True).cpu().float()
                if config.task == 'SSR' or config.task == 'V2V':
                    k = np.multiply(result[0][0].cpu().detach().numpy(),w)
                    avI[si:ei,sj:ej,sk:ek] += w
                    pmap[si:ei,sj:ej,sk:ek] += k
                elif config.task == 'TSR':
                    k = np.multiply(result[0].cpu().detach().numpy(),w)
                    avI[:,si:ei,sj:ej,sk:ek] += w
                    pmap[:,si:ei,sj:ej,sk:ek] += k
    high = np.divide(pmap,avI)
    return high

def PSNR(vol,preds):
    mse = np.mean((vol-preds)**2)
    diff = vol.max()-vol.min()
    psnr = 20*np.log10(diff)-10*np.log10(mse)
    return psnr


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def save_loss_info(filepath, loss):
    with open(filepath,"w") as f:
        yaml.dump(loss,f)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def getPath(dataset,num_samples,data_path='/root/autodl-tmp/Data/'):
  path = None
  res = None
  if dataset == 'Vortex':
    path = data_path+'Vortex/'
    res = [256,256,256]
    samples = {1:[45],3:[15,45,75],5:[9,27,45,63,81],15:[i for i in range(1,91,6)],18:[i for i in range(1,91,5)],30:[i for i in range(1,91,3)],60:[i for i in range(1,91,3)] + [i for i in range(2,91,3)]}
  elif dataset == 'Tangaroa':
    path = data_path+'Tangaroa/'
    res = [300,180,120]
    samples = {1:[75],3:[25,75,125],5:[15,45,75,105,135]}
  elif dataset == 'Ionization':
    path = data_path+'Ionization/'
    res = [600,248,248]
    samples = {1:[50],3:[17,50,84],5:[10,30,50,70,90],7:[7,21,35,49,63,77,90],10:[i for i in range(10,101,10)],20:[i for i in range(5,101,5)],50:[i for i in range(2,101,2)]}
  elif dataset == 'Combustion':
    path = data_path+'Combustion/'
    res = [480,720,120]
    samples = {1:[50],3:[17,50,84],5:[10,30,50,70,90],7:[7,21,35,49,63,77,90],10:[i for i in range(10,101,10)],20:[i for i in range(5,101,5)],50:[i for i in range(2,101,2)]}
  elif dataset == 'Supernova':
    path = data_path
    res = [384,384,384]
    samples = {1:[30]}
  elif dataset == 'Bubble':
    path = data_path+'Bubble/'
    res = [640,256,256]
    samples = {1:[75],3:[25,75,125],5:[15,45,75,105,135]}
  elif dataset == 'Cylinder':
    path = data_path+'Cylinder/'
    res = [640,240,80]
    samples = {1:[75],3:[25,75,125],5:[15,45,75,105,135]}
  elif dataset == 'Jets':
    path = data_path+'Jets/'
    res = [128,128,128]
    samples = {1:[500],3:[167,500,833],5:[100,300,500,700,900]}
  else:
    raise NotImplementedError('Not Implemented for the '+str(dataset)+' Data Set!')

  return path, res, samples[num_samples]