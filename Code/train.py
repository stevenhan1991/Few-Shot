import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import os
from utils import *
from model import *
from dataio import *
from torch.utils.data import DataLoader, Dataset
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType,ModelMeanType, ModelVarType, get_named_beta_schedule
from resample import *
from tqdm import tqdm

def train_diffusion(model,vol_dl,config,device):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.optimizer.lr)
    info = {'MSE Loss':[],'Time':0}

    loss = 1e5

    timesteps = config.model.beta_schedule.train.n_timestep

    betas = torch.tensor(np.linspace(config.model.beta_schedule.linear_start,config.model.beta_schedule.linear_end, timesteps))

    # Initialize diffusion utiities
    diff = GaussianDiffusion(betas=betas,
                             model_mean_type=ModelMeanType[config.model.diffusion.params.model_mean_type],
                             model_var_type=ModelVarType[config.model.diffusion.params.model_var_type],
                             loss_type=LossType[config.model.diffusion.params.loss_type])

    schedule_sampler = LossSecondMomentResampler(config)


    for epoch in range(1,config.train.epochs+1):
        epoch_loss = 0
        x = time.time()
        for (batch_id, data) in enumerate(vol_dl):

            output = data['output']
            output = output.to(memory_format=torch.contiguous_format)
            output = output.float()
            output = output.to(device)

            cond = data['input']
            cond = cond.to(memory_format=torch.contiguous_format)
            cond = cond.float()
            cond = cond.to(device)

            if len(output.shape) == 4:
                output = output[None,...]

            if len(cond.shape) == 4:
                cond = cond[None,...]

            t, weights = schedule_sampler.sample(cond.shape[0], device)

            # Execute a diffusion forward pass
            loss_terms = diff.training_losses(model,output,t,cond,model_kwargs=None)
            loss_mse = loss_terms["loss"].mean()

            epoch_loss += loss_mse.item()

            optimizer.zero_grad()
            loss_mse.backward()
            optimizer.step()

            schedule_sampler.update_with_all_losses(t, loss_terms["loss"].detach())

        print('[{:02d}/{:d}] MSE Loss: {:.4f}'.format(epoch,config.train.epochs,epoch_loss))

        y = time.time()
        info['MSE Loss'].append(epoch_loss)
        info['Time'] += y-x


        if epoch % 1000 == 0:
            if config.task == 'SSR' or config.task == 'TSR':
                save_loss_info(config.model.model_path+config.task+'/{:d}/{:d}/{:d}/{:s}/{:s}_{:d}_diffusion_loss.yaml'.format(timesteps,config.data.factor,config.data.num_samples,config.data.sampling,config.task, config.model.unet.ch),info)
                torch.save(model.state_dict(),config.model.model_path+config.task+'/{:d}/{:d}/{:d}/{:s}/{:s}_{:d}_best_model_epoch_{:d}.pth'.format(timesteps,config.data.factor,config.data.num_samples,config.data.sampling,config.task,config.model.unet.ch, epoch))
            elif config.task == 'V2V':
                torch.save(model.state_dict(),config.model.model_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}/{:s}_best_model_epoch_{:d}.pth'.format(timesteps,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2,config.task,epoch))
                save_loss_info(config.model.model_path+config.task+'/{:d}/{:d}/{:s}/{:s}-{:s}/{:s}_diffusion_loss.yaml'.format(timesteps,config.data.num_samples,config.data.sampling,config.data.var1,config.data.var2,config.task),info)
