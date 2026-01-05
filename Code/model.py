import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import time
from torch.nn import Linear, Conv1d, BatchNorm1d, Conv3d, InstanceNorm3d, AdaptiveAvgPool1d, ModuleList
import math
import numpy as np

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1 , 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1, 1)
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm")!=-1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class Upsample(nn.Module):
    def __init__(self,inchannels,outchannels,factor=2.0):
        super(Upsample,self).__init__()
        self.conv = nn.Conv3d(inchannels,outchannels,kernel_size=3,stride=1,padding=1)
        self.factor = factor

    def forward(self,x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode="trilinear",align_corners=False)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self,inchannels,outchannels,factor=2):
        super(Downsample,self).__init__()
        self.conv = nn.Conv3d(inchannels,outchannels,kernel_size=4,stride=factor,padding=1)

    def forward(self,x):
        x = self.conv(x)
        return x

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 temb_channels=512,
                 use_affine_level=False):
        super(ResidualLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels//4,kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels//4, out_channels//4,kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(out_channels//4, out_channels,kernel_size=3, padding=1, bias=False)
        self.ac = nn.SiLU()
        self.temb_proj = FeatureWiseAffine(temb_channels, out_channels//4, use_affine_level)

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels,kernel_size=3, padding=1, bias=False)

    def forward(self, x, temb) :

        h = self.conv1(x)

        h = self.temb_proj(h, temb)

        h = self.ac(h)

        h = self.conv2(h)

        h = self.ac(h)

        h = self.conv3(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6/ self.in_features)/30, np.sqrt(6/ self.in_features)/30)  
        
    def forward(self, input):
        return self.linear(input)

class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_dims,temb_ch):
        super(Encoder,self).__init__()

        if hidden_dims is None:
            hidden_dims = [64,128,128]

        self.temb_ch = temb_ch
        self.ac = nn.SiLU()

        self.down = nn.ModuleList()
        # Build Encoder
        for i in range(0,len(hidden_dims)):
            block = nn.ModuleList()
            block.down = Downsample(in_channels,hidden_dims[i]),
            block.block = ResidualLayer(hidden_dims[i],hidden_dims[i],temb_channels=self.temb_ch)
            in_channels = hidden_dims[i]
            self.down.append(block)

        self.mid = nn.Module()

        self.mid.block_1 = ResidualLayer(in_channels, in_channels,temb_channels = self.temb_ch)
        self.mid.block_2 = ResidualLayer(in_channels, out_channels, temb_channels = self.temb_ch)

    def forward(self,x,t):
        for i_level in range(len(self.down)):
            x = self.down[i_level].down(x)
            x = self.ac(x)
            x = self.down[i_level].block(x,t)

        x = self.mid.block_1(x,t)
        x = self.mid.block_2(x,t)
        return x

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_dims,temb_ch):
        super(Decoder,self).__init__()

        if hidden_dims is None:
            hidden_dims = [64,128,128]
        
        self.temb_ch = temb_ch
        self.ac = nn.SiLU()

        self.up = nn.ModuleList()

        self.conv_in = nn.Conv3d(in_channels,hidden_dims[-1],kernel_size=3,stride=1,padding=1)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            block = nn.ModuleList()
            block.block = ResidualLayer(hidden_dims[i],hidden_dims[i],temb_channels = self.temb_ch)
            block.up = Upsample(hidden_dims[i],hidden_dims[i + 1])
            self.up.append(block)

        block = nn.ModuleList()
        block.block = ResidualLayer(hidden_dims[-1],hidden_dims[-1], temb_channels = self.temb_ch)
        block.up = Upsample(hidden_dims[-1],32)

        self.conv_out = nn.Conv3d(32,1,kernel_size=3,stride=1,padding=1)


    def forward(self,x,t):
        x = self.conv_in(x)
        x = self.ac(x)

        for i_level in range(len(self.up)):
            x = self.up[i_level].block(x,t)
            x = self.up[i_level].up(x)
            x = self.ac(x)

        x = self.conv_out(x)
        return x


class TUNet(nn.Module):
    def __init__(self,in_channels,out_channels,ch,scale=3,num_blocks=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch = ch
        self.temb_ch = ch
        self.num_blocks = num_blocks
        self.ac = nn.SiLU()

        self.temb = []

        self.temb.append(PositionalEncoding(ch))
        self.temb.append(nn.Linear(ch, ch * 4))
        self.temb.append(nn.SiLU())
        self.temb.append(nn.Linear(ch*4,ch))

        self.temb = nn.Sequential(*self.temb)

        self.down = nn.ModuleList()
        self.tranform = nn.ModuleList()
        block = nn.Module()
        module = [Downsample(self.in_channels,self.ch)]
        module += [ResidualLayer(self.ch,self.ch,temb_channels=self.temb_ch) for i in range(0,num_blocks)]

        block.block = nn.ModuleList(module)
        self.down.append(block)

        # Build Encoder
        for i in range(1,scale):
            block = nn.Module()
            module = [Downsample(self.ch,2*self.ch)]
            module += [ResidualLayer(2*self.ch,2*self.ch,temb_channels=self.temb_ch) for i in range(0,num_blocks)]
            block.block = nn.ModuleList(module)
            self.down.append(block)

            self.ch *= 2

        self.mid = nn.Module()

        self.mid.block_1 = ResidualLayer(self.ch,self.ch,temb_channels = self.temb_ch)
        self.mid.block_2 = ResidualLayer(self.ch,self.ch, temb_channels = self.temb_ch)

        # Build Decoder
        self.up = nn.ModuleList()
        self.conv_in = nn.Conv3d(self.ch,self.ch,kernel_size=3,stride=1,padding=1)

        for i in range(scale,1,-1):
            block = nn.Module()
            module = [ResidualLayer(2*self.ch,self.ch,temb_channels=self.temb_ch)]
            module += [ResidualLayer(self.ch,self.ch,temb_channels = self.temb_ch) for i in range(0,num_blocks)]
            module += [Upsample(self.ch, self.ch//2)]
            block.block = nn.ModuleList(module)
            self.up.append(block)
            self.ch //= 2

        block = nn.Module()
        module = [ResidualLayer(2*self.ch,self.ch,temb_channels=self.temb_ch)]
        module += [ResidualLayer(self.ch,self.ch,temb_channels = self.temb_ch) for i in range(0,num_blocks)]
        module += [Upsample(self.ch, 32)]
        block.block = nn.ModuleList(module)

        self.up.append(block)

        self.conv_out = nn.Conv3d(32,self.out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,x,t):
        temb = self.temb(t)

        hs = []

        for i_level in range(len(self.down)):
            x = self.down[i_level].block[0](x)
            x = self.ac(x)
            for j_level in range(1,self.num_blocks+1):
                x = self.down[i_level].block[j_level](x,temb)
            hs.append(x)

        x = self.mid.block_1(x,temb)
        x = self.mid.block_2(x,temb)

        x = self.conv_in(x)
        x = self.ac(x)

        for i_level in range(len(self.up)):
            features = hs.pop()
            x = self.up[i_level].block[0](torch.cat((x,features),dim=1),temb)
            for j_level in range(1,self.num_blocks+1):
                x = self.up[i_level].block[j_level](x,temb)
            x = self.up[i_level].block[-1](x)
            x = self.ac(x)

        x = self.conv_out(x)
        return x


class FCN(nn.Module):
    def __init__(self,in_channels,out_channels,ch,scale=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch = ch
        self.ac = nn.ReLU(inplace=True)
        self.down = nn.ModuleList()
        block = nn.Module()

        module = [Downsample(self.in_channels,self.ch)]

        block.block = nn.ModuleList(module)
        self.down.append(block)


        # Build Encoder
        for i in range(1,scale):
            block = nn.Module()
            module = [Downsample(self.ch,2*self.ch)]
            block.block = nn.ModuleList(module)
            self.down.append(block)
            self.ch *= 2


        # Build Decoder
        self.up = nn.ModuleList()
        self.conv_in = nn.Conv3d(self.ch,self.ch,kernel_size=3,stride=1,padding=1)

        for i in range(scale,1,-1):
            block = nn.Module()
            module = [nn.Conv3d(self.ch,self.ch,kernel_size=3,stride=1,padding=1)]
            module += [Upsample(self.ch, self.ch//2)]
            block.block = nn.ModuleList(module)
            self.up.append(block)
            self.ch //= 2

        block = nn.Module()
        module = [nn.Conv3d(self.ch,self.ch,kernel_size=3,stride=1,padding=1)]
        module += [Upsample(self.ch, 32)]
        block.block = nn.ModuleList(module)

        self.up.append(block)

        self.conv_out = nn.Conv3d(32,1,kernel_size=3,stride=1,padding=1)

    def forward(self,x):

        for i_level in range(len(self.down)):
            x = self.down[i_level].block[0](x)
            x = self.ac(x)

        x = self.conv_in(x)
        x = self.ac(x)

        for i_level in range(len(self.up)):
            x = self.up[i_level].block[0](x)
            x = self.ac(x)
            x = self.up[i_level].block[1](x)
            x = self.ac(x)

        x = self.conv_out(x)
        return x

    def encode(self, x):
        for i_level in range(len(self.down)):
            x = self.down[i_level].block[0](x)
            x = self.ac(x)

        x = self.conv_in(x)
        x = self.ac(x)

        return F.avg_pool3d(x,x.size()[2:]).view(-1)



### CoordNet
class Sine(nn.Module):
    def __init(self):
        super(Sine,self).__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        #return (torch.sin(self.omega_0 * self.linear(input))+torch.cos(self.omega_0 * self.linear(input)))/np.sqrt(2.0)
        return torch.sin(self.omega_0 * self.linear(input))


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6/ self.in_features)/30, np.sqrt(6/ self.in_features)/30)  
        
    def forward(self, input):
        return self.linear(input)

class CoordNet(nn.Module):
    #A fully connected neural network that also allows swapping out the weights when used with a hypernetwork. Can be used just as a normal neural network though, as well.

    def __init__(self, in_features, out_features, init_features=64, num_res = 10):
        super(CoordNet,self).__init__()

        self.num_res = num_res

        self.net = []

        self.net.append(SineLayer(in_features,init_features,is_first=True))
        self.net.append(ResBlock(init_features,2*init_features))
        self.net.append(ResBlock(2*init_features,4*init_features))

        for i in range(self.num_res):
            self.net.append(ResBlock(4*init_features,4*init_features))

        self.net.append(LinearLayer(4*init_features, out_features))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features):
        super(ResBlock,self).__init__()

        self.net = []

        self.net.append(SineLayer(in_features,out_features))

        self.net.append(SineLayer(out_features,out_features))

        self.flag = (in_features!=out_features)

        if self.flag:
            self.transform = SineLayer(in_features,out_features)

        self.net = nn.Sequential(*self.net)
    
    def forward(self,features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)


def BuildResidualBlock(channels,dropout,kernel,depth,bias):
    layers = []
    for i in range(int(depth)):
        layers += [nn.Conv3d(channels,channels,kernel_size=kernel,padding=kernel//2,bias=bias),
                   nn.ReLU(True)]
        if dropout:
            layers += [nn.Dropout(0.5)]
    layers += [nn.Conv3d(channels,channels,kernel_size=kernel,padding=kernel//2,bias=bias),
               ]
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self,channels,dropout,kernel,depth,bias):
        super(ResidualBlock,self).__init__()
        self.block = BuildResidualBlock(channels,dropout,kernel,depth,bias)

    def forward(self,x):
        out = x+self.block(x)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm")!=-1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class Block(nn.Module):
    def __init__(self,inchannels,outchannels,dropout,kernel,bias,depth,mode):
        super(Block,self).__init__()
        layers = []
        for i in range(int(depth)):
            layers += [
                       nn.Conv3d(inchannels,inchannels,kernel_size=kernel,padding=kernel//2,bias=bias),
                       nn.ReLU(inplace=True)]
            if dropout:
                layers += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*layers)

        if mode == 'down':
            self.conv1 = nn.Conv3d(inchannels,outchannels,4,stride=2,padding=1,bias=bias)
            self.conv2 = nn.Conv3d(inchannels,outchannels,4,stride=2,padding=1,bias=bias)
        elif mode == 'up':
            self.conv1 = nn.ConvTranspose3d(inchannels,outchannels,4,stride=2,padding=1,bias=bias)
            self.conv2 = nn.ConvTranspose3d(inchannels,outchannels,4,stride=2,padding=1,bias=bias)
        elif mode == 'same':
            self.conv1 = nn.Conv3d(inchannels,outchannels,kernel_size=3,stride=1,padding=1,bias=bias)
            self.conv2 = nn.Conv3d(inchannels,outchannels,kernel_size=3,stride=1,padding=1,bias=bias)

    def forward(self,x):
        y = self.model(x)
        y = self.conv1(y)
        x = self.conv2(x)
        return y+x

def voxel_shuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height, in_width, in_depth)

    return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)

class VoxelShuffle(nn.Module):
    def __init__(self,inchannels,outchannels,upscale_factor):
        super(VoxelShuffle,self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv3d(inchannels,outchannels*(upscale_factor**3),3,1,1)

    def forward(self,x):
        x = voxel_shuffle(self.conv(x),self.upscale_factor)
        return x

class SSR(nn.Module):
    def __init__(self,factor=4):
        super(SSR,self).__init__()

        self.factor = factor

        if factor == 2:
            init = 28
        elif factor == 4:
            init = 22
        elif factor == 8:
            init = 20


        self.b1 = Block(inchannels=1,outchannels=init,dropout=False,kernel=3,bias=True,depth=1,mode='same')
        self.b2 = Block(inchannels=init,outchannels=2*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')
        self.b3 = Block(inchannels=3*init,outchannels=init,dropout=False,kernel=3,bias=True,depth=1,mode='same')

        if self.factor == 2:
            self.deconv1 = VoxelShuffle(init,4*init,2)
            self.b4 = Block(inchannels=4*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')
            self.b5 = Block(inchannels=8*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')

        if factor == 4:
            self.deconv1 = VoxelShuffle(init,4*init,2)
            self.b4 = Block(inchannels=4*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')
            self.b5 = Block(inchannels=8*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')

            self.deconv2 = VoxelShuffle(4*init,4*init,2)
            self.b6 = Block(inchannels=4*init,outchannels=2*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')
            self.b7 = Block(inchannels=6*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')

        if factor == 8:
            self.deconv1 = VoxelShuffle(init,4*init,2)
            self.b4 = Block(inchannels=4*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')
            self.b5 = Block(inchannels=8*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')

            self.deconv2 = VoxelShuffle(4*init,4*init,2)
            self.b6 = Block(inchannels=4*init,outchannels=2*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')
            self.b7 = Block(inchannels=6*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')

            self.deconv3 = VoxelShuffle(4*init,4*init,2)
            self.b8 = Block(inchannels=4*init,outchannels=2*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')
            self.b9 = Block(inchannels=6*init,outchannels=4*init,dropout=False,kernel=3,bias=True,depth=1,mode='same')

        
        self.conv = nn.Conv3d(4*init,1,3,1,1)

    def forward(self,x):
        x1 = F.relu(self.b1(x)) # init
        x2 = F.relu(self.b2(x1)) # 2*init
        x3 = F.relu(self.b3(torch.cat((x1,x2),dim=1))) #init
        x3 = x3 + x1

        if self.factor == 2:
            x4 = F.relu(self.deconv1(x3)) # 4*init
            x5 = F.relu(self.b4(x4)) # 4*init
            x6 = F.relu(self.b5(torch.cat((x4,x5),dim=1))) # 4*init
            x6 = x6 + x4
            x = self.conv(x6)

        if self.factor == 4:
            x4 = F.relu(self.deconv1(x3)) # 4*init
            x5 = F.relu(self.b4(x4)) # 4*init
            x6 = F.relu(self.b5(torch.cat((x4,x5),dim=1))) # 4*init
            x6 = x6 + x4
            x7 = F.relu(self.deconv2(x6)) # 2*init
            x8 = F.relu(self.b6(x7)) # 2*init
            x9 = self.b7(torch.cat((x7,x8),dim=1)) # 4*init
            x9 = x7 + x9
            x = self.conv(x9)

        if self.factor == 8:
            x4 = F.relu(self.deconv1(x3)) # 4*init
            x5 = F.relu(self.b4(x4)) # 4*init
            x6 = F.relu(self.b5(torch.cat((x4,x5),dim=1))) # 4*init
            x6 = x6 + x4
            x7 = F.relu(self.deconv2(x6)) # 2*init
            x8 = F.relu(self.b6(x7)) # 2*init
            x9 = self.b7(torch.cat((x7,x8),dim=1)) # 4*init
            x9 = x7 + x9
            x10 = F.relu(self.deconv3(x9)) # 2*init
            x11 = F.relu(self.b8(x10)) # 2*init
            x12 = self.b9(torch.cat((x10,x11),dim=1)) # 4*init
            x = self.conv(x12)

        return x


class SRGAN(nn.Module):
    def __init__(self,factor=4):
        super(SRGAN,self).__init__()

        if factor == 2:
            init = 32
        else:
            init = 12

        self.head = nn.Sequential(*[nn.Conv3d(1,init,3,1,1),
                                   nn.ReLU(inplace=True)])

        self.net = nn.Sequential(*[ResidualBlock(init,False,3,1,True),
                                   ResidualBlock(init,False,3,1,True),
                                   ResidualBlock(init,False,3,1,True),
                                   ResidualBlock(init,False,3,1,True),
                                   ResidualBlock(init,False,3,1,True),
                                   ResidualBlock(init,False,3,1,True),
                                   ResidualBlock(init,False,3,1,True),
                                   nn.Conv3d(init,init,3,1,1),
                                   nn.ReLU(inplace=True)]
                                   )

        if factor == 2:
            self.upscale = nn.Sequential(*[VoxelShuffle(init,16*init,2),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(16*init,1,1,1)])
        elif factor == 4:
            self.upscale = nn.Sequential(*[VoxelShuffle(init,16*init,2),
                                           nn.ReLU(inplace=True),
                                           VoxelShuffle(16*init,8*init,2),
                                           nn.ReLU(inplace=True),
                                           nn.Conv3d(8*init,1,1,1)])
        elif factor == 8:
            self.upscale = nn.Sequential(*[VoxelShuffle(init,16*init,2),
                                           nn.ReLU(inplace=True),
                                           VoxelShuffle(16*init,8*init,2),
                                           nn.ReLU(inplace=True),
                                           VoxelShuffle(8*init,4*init,2),
                                           nn.ReLU(inplace=True),
                                           nn.Conv3d(4*init,1,1,1)])

    def forward(self,x):
        x = self.head(x)
        y = self.net(x)
        x = y+x
        output = self.upscale(x)
        return output


class ResidualLayer_(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResidualLayer_, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels,kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels,kernel_size=1, stride=1, bias=False)
        self.ac = nn.ReLU(inplace=True)
      
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels,kernel_size=3, padding=1, bias=True)

    def forward(self, x, temb=None) :

        h = x
        h = self.conv1(h)
        h = self.ac(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h

class V2V(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,ch=32,scale=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch = ch
        self.ac = nn.ReLU(inplace=True)

        # downsampling
        self.down = nn.ModuleList()
        self.tranform = nn.ModuleList()

        block = nn.Module()
        block.block = nn.ModuleList([Downsample(self.in_channels,self.ch),
                                     ResidualLayer_(self.ch,self.ch)])
        self.down.append(block)

        block = nn.Module()
        block.block = nn.ModuleList([ResidualLayer_(self.ch,self.ch)])
        self.tranform.append(block)


        # Build Encoder
        for i in range(1,scale):
            block = nn.Module()
            block.block = nn.ModuleList([Downsample(self.ch,2*self.ch),
                                         ResidualLayer_(2*self.ch,2*self.ch)])
            
            self.down.append(block)

            block = nn.Module()
            block.block = nn.ModuleList([ResidualLayer_(2*self.ch,2*self.ch)])
            self.tranform.append(block)

            self.ch *= 2

        # Build Decoder
        self.up = nn.ModuleList()
        self.conv_in = nn.Conv3d(self.ch,self.ch,kernel_size=3,stride=1,padding=1)

        for i in range(scale,1,-1):
            block = nn.Module()
            block.block = nn.ModuleList([ResidualLayer_(2*self.ch,self.ch),
                                         Upsample(self.ch, self.ch//2,2)])
            self.up.append(block)
            self.ch //= 2

        block = nn.Module()
        block.block = nn.ModuleList([ResidualLayer_(2*self.ch,self.ch),
                                     Upsample(self.ch,32)])

        self.up.append(block)

        self.conv_out = nn.Conv3d(32,self.out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,x):

        hs = []

        for i_level in range(len(self.down)):
            x = self.down[i_level].block[0](x)
            x = self.ac(x)
            x = self.down[i_level].block[1](x)
            hs.append(x)

        x = self.conv_in(x)
        x = self.ac(x)

        for i_level in range(len(self.up)):
            features = hs.pop()
            features = self.tranform[len(self.up)-i_level-1].block[0](features)
            x = self.up[i_level].block[0](torch.cat((x,features),dim=1))
            x = self.up[i_level].block[1](x)
            x = self.ac(x)

        x = self.conv_out(x)
        return x

class DVAO(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,ch=32,scale=3):
        super(DVAO,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch = ch
        self.ac = nn.ReLU(inplace=True)
        self.skip_szs = []
       

        # downsampling
        self.down = nn.ModuleList()
        block = nn.Module()
        block.block = nn.ModuleList([Downsample(self.in_channels,self.ch),
                                     nn.Conv3d(self.ch,self.ch,3,1,1),
                                     nn.Conv3d(self.ch,self.ch,3,1,1)])
        self.down.append(block)

        # Build Encoder
        for i in range(1,scale):
            block = nn.Module()
            block.block = nn.ModuleList([Downsample(self.ch,self.ch),
                                         nn.Conv3d(self.ch,2*self.ch,3,1,1),
                                         nn.Conv3d(2*self.ch,2*self.ch,3,1,1)])
            self.ch *= 2
            self.down.append(block)

        # Build Decoder
        self.up = nn.ModuleList()
        self.conv_in = nn.Conv3d(self.ch,self.ch,3,1,1)

        for i in range(scale,1,-1):
            block = nn.Module()
            block.block = nn.ModuleList([nn.Conv3d(2*self.ch, self.ch, 3,1,1),
                                         nn.Conv3d(self.ch, self.ch,3,1,1),
                                         Upsample(self.ch, self.ch//2)])
            self.up.append(block)
            self.ch //= 2

        block = nn.Module()
        block.block = nn.ModuleList([nn.Conv3d(2*self.ch,self.ch,3,1,1),
                                     nn.Conv3d(self.ch,self.ch,3,1,1),
                                     Upsample(self.ch,32)])

        self.up.append(block)

        self.conv_out = nn.Conv3d(32,self.out_channels,3,1,1)

    def forward(self,x):

        hs = []

        for i_level in range(len(self.down)):
            x = self.down[i_level].block[0](x)
            x = self.ac(x)
            x = self.down[i_level].block[1](x)
            x = self.ac(x)
            x = self.down[i_level].block[2](x)
            x = self.ac(x)
            hs.append(x)

        x = self.conv_in(x)
        x = self.ac(x)

        for i_level in range(len(self.up)):
            features = hs.pop()
            x = self.up[i_level].block[0](torch.cat((x,features),dim=1))
            x = self.ac(x)
            x = self.up[i_level].block[1](x)
            x = self.ac(x)
            x = self.up[i_level].block[2](x)
            x = self.ac(x)

        x = self.conv_out(x)
        return x


class Pix2Pix(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,ch=40,scale=3):
        super(Pix2Pix,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch = ch
        self.ac = nn.ReLU(inplace=True)
       

        # downsampling
        self.down = nn.ModuleList()
        block = nn.Module()
        block.block = nn.ModuleList([Downsample(self.in_channels,self.ch),
                                     nn.Conv3d(self.ch,self.ch,3,1,1)])
        self.down.append(block)

        # Build Encoder
        for i in range(1,scale):
            block = nn.Module()
            block.block = nn.ModuleList([Downsample(self.ch,self.ch),
                                         nn.Conv3d(self.ch,2*self.ch,3,1,1)])
            self.ch *= 2
            self.down.append(block)

        # Build Decoder
        self.up = nn.ModuleList()
        self.conv_in = nn.Conv3d(self.ch,self.ch,3,1,1)

        for i in range(scale,1,-1):
            block = nn.Module()
            block.block = nn.ModuleList([nn.Conv3d(2*self.ch, self.ch, 3,1,1),
                                         Upsample(self.ch, self.ch//2)])
            self.up.append(block)
            self.ch //= 2

        block = nn.Module()
        block.block = nn.ModuleList([nn.Conv3d(2*self.ch,self.ch,3,1,1),
                                     Upsample(self.ch,32)])

        self.up.append(block)

        self.conv_out = nn.Conv3d(32,self.out_channels,3,1,1)

    def forward(self,x):

        hs = []

        for i_level in range(len(self.down)):
            x = self.down[i_level].block[0](x)
            x = self.ac(x)
            x = self.down[i_level].block[1](x)
            x = self.ac(x)
            hs.append(x)

        x = self.conv_in(x)
        x = self.ac(x)

        for i_level in range(len(self.up)):
            features = hs.pop()
            x = self.up[i_level].block[0](torch.cat((x,features),dim=1))
            x = self.ac(x)
            x = self.up[i_level].block[1](x)
            x = self.ac(x)

        x = self.conv_out(x)
        return x


class FCN(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,ch=48,scale=3):
        super(FCN,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch = ch
        self.ac = nn.ReLU(inplace=True)
       

        # downsampling
        self.down = nn.ModuleList()
        block = nn.Module()
        block.block = nn.ModuleList([Downsample(self.in_channels,self.ch),
                                     nn.Conv3d(self.ch,self.ch,3,1,1)])
        self.down.append(block)

        # Build Encoder
        for i in range(1,scale):
            block = nn.Module()
            block.block = nn.ModuleList([Downsample(self.ch,self.ch),
                                         nn.Conv3d(self.ch,2*self.ch,3,1,1)])
            self.ch *= 2
            self.down.append(block)

        # Build Decoder
        self.up = nn.ModuleList()
        self.conv_in = nn.Conv3d(self.ch,self.ch,3,1,1)

        for i in range(scale,1,-1):
            block = nn.Module()
            block.block = nn.ModuleList([nn.Conv3d(self.ch, self.ch, 3,1,1),
                                         Upsample(self.ch, self.ch//2)])
            self.up.append(block)
            self.ch //= 2

        block = nn.Module()
        block.block = nn.ModuleList([nn.Conv3d(self.ch,self.ch,3,1,1),
                                     Upsample(self.ch,32)])

        self.up.append(block)

        self.conv_out = nn.Conv3d(32,self.out_channels,3,1,1)

    def forward(self,x):

        hs = []

        for i_level in range(len(self.down)):
            x = self.down[i_level].block[0](x)
            x = self.ac(x)
            x = self.down[i_level].block[1](x)
            x = self.ac(x)

        x = self.conv_in(x)
        x = self.ac(x)

        for i_level in range(len(self.up)):
            x = self.up[i_level].block[0](x)
            x = self.ac(x)
            x = self.up[i_level].block[1](x)
            x = self.ac(x)

        x = self.conv_out(x)
        return x

class ESPCN(nn.Module):
    def __init__(self,init=20):
        super(ESPCN,self).__init__()
        self.net = nn.Sequential(*[nn.Conv3d(1,init,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(init,2*init,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(2*init,4*init,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(4*init,8*init,3,1,1),
                                   nn.ReLU(inplace=True),
                                   VoxelShuffle(8*init,4*init,2),
                                   nn.ReLU(inplace=True),
                                   VoxelShuffle(4*init,2*init,2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(2*init,1,3,1,1)])

    def forward(self,x):
        output = self.net(x)
        return output

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = spectral_norm(nn.Conv3d(1,64,4,stride=2,padding=1),eps=1e-4)
        self.leaky = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = spectral_norm(nn.Conv3d(64,128,4,stride=2,padding=1),eps=1e-4)
        self.conv3 = spectral_norm(nn.Conv3d(128,256,4,stride=2,padding=1),eps=1e-4)
        self.conv4 = spectral_norm(nn.Conv3d(256,1,4,stride=2,padding=1),eps=1e-4)


    def forward(self,x):
        result = []
        x = self.leaky(self.conv1(x))
        result.append(x)
        x = self.leaky(self.conv2(x))
        result.append(x)
        x = self.leaky(self.conv3(x))
        result.append(x)
        x = self.leaky(self.conv4(x))
        result.append(x)
        x = F.avg_pool3d(x,x.size()[2:]).view(-1)
        result.append(x)
        return result

