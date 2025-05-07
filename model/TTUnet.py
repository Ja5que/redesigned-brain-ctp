import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class convwithactivation(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1,dilation=1):
        super(convwithactivation,self).__init__()
        self.conv=nn.Conv3d(in_ch,out_ch,kernel_size,stride,padding,dilation)
        self.lrelu=nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.lrelu(x)
        return x

class upconvwithactivation(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=2,padding=1,scale_factor=1):
        super(upconvwithactivation,self).__init__()
        self.scale_factor=scale_factor
        self.conv=nn.ConvTranspose3d(in_ch,out_ch,kernel_size,stride,padding)
        self.lrelu=nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self,x):
        x = F.interpolate(x,scale_factor=self.scale_factor,mode='nearest')
        x = self.conv(x)
        x = self.lrelu(x)
        return x
    
class deconvwithactivation(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):
        super(deconvwithactivation,self).__init__()
        self.conv=nn.ConvTranspose3d(in_ch,out_ch,kernel_size,stride,padding)
        self.lrelu=nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.lrelu(x)
        return x
    
class PatchEmbed(nn.Module):

    def __init__(self, img_size=256, patch_size=2, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.conv=nn.Conv3d(in_chans,4*in_chans,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,2,2])
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
            
    def forward(self, x):
        x=self.conv(x)
        B,C,S,H,W = x.shape
        x = x.permute(0,3,4,2,1).reshape(B,-1,C)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=256, patch_size=1, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.conv=nn.ConvTranspose3d(in_chans,int(in_chans/4),[1,3,3],padding=[0,1,1],stride=[1,1,1])
        #self.conv=nn.Conv3d(in_chans,int(in_chans/4),kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        
    def forward(self, x, x_size):
        B, HWS, C = x.shape
        x = x.view(B,x_size[0],x_size[1],-1,C).permute(0,4,3,1,2) #B*C*S*H*W
        x=F.interpolate(x,scale_factor=[1,2,2],mode='nearest')
        x=self.conv(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

   
class TransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=1, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
    
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape  #L=H*W*S

        shortcut = x
        x = self.norm1(x)
        #x = x.view(B, H, W, C)

        # partition windows
        #x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        
        #x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_windows = x.view(-1, 48, C)  # B*H*W, S, C

        x = self.attn(x_windows)  # B*H*W, S, C
        x = x.view(B,L,C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class TTUNet(nn.Module):
    def __init__(self,chl=1):
        super(TTUNet,self).__init__()
        ##input=batchsize*1*48*256*256
        self.conv1=convwithactivation(1,chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        self.conv1_2=convwithactivation(chl,chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        
        self.convt1_1=convwithactivation(chl,chl,kernel_size=[5,1,1],padding=[2,0,0],stride=[1,1,1])
        self.convt1_2=convwithactivation(chl,chl,kernel_size=[5,1,1],padding=[2,0,0],stride=[1,1,1])
        self.convt1_3=convwithactivation(chl,chl,kernel_size=[5,1,1],padding=[2,0,0],stride=[1,1,1])
        
        ##48*256*256->48*128*128
        self.conv2=convwithactivation(chl,2*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,2,2])
        self.conv3=convwithactivation(2*chl,2*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        self.conv3_2=convwithactivation(2*chl,2*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        
        self.embed1=PatchEmbed(in_chans=2*chl)
        self.trans1_1=TransformerBlock(dim=8*chl, input_resolution=[64,64], num_heads=torch.tensor(chl*32/16,dtype=int), window_size=1, shift_size=0,mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.05,act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.trans1_2=TransformerBlock(dim=8*chl, input_resolution=[64,64], num_heads=torch.tensor(chl*32/16,dtype=int), window_size=1, shift_size=0,mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.05,act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.unembed1=PatchUnEmbed(in_chans=8*chl)
        self.convt1=convwithactivation(2*chl,2*chl,kernel_size=[3,1,1],padding=[1,0,0],stride=[1,1,1])
        
        ##48*128*128->48*64*64
        self.conv4=convwithactivation(2*chl,4*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,2,2])
        self.conv5=convwithactivation(4*chl,4*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        self.conv5_2=convwithactivation(4*chl,4*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        
        self.embed2=PatchEmbed(in_chans=4*chl)
        self.trans2_1=TransformerBlock(dim=16*chl, input_resolution=[32,32], num_heads=torch.tensor(chl*16/16,dtype=int), window_size=1, shift_size=0,mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.trans2_2=TransformerBlock(dim=16*chl, input_resolution=[32,32], num_heads=torch.tensor(chl*16/16,dtype=int), window_size=1, shift_size=0,mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.unembed2=PatchUnEmbed(in_chans=16*chl)
        self.convt2=convwithactivation(4*chl,4*chl,kernel_size=[3,1,1],padding=[1,0,0],stride=[1,1,1])
        
        ##48*64*64->48*32*32
        self.conv6=convwithactivation(4*chl,8*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,2,2])
        self.conv7=convwithactivation(8*chl,8*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        self.conv7_2=convwithactivation(8*chl,8*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        
        self.embed3=PatchEmbed(in_chans=8*chl)
        self.trans3_1=TransformerBlock(dim=32*chl, input_resolution=[16,16], num_heads=torch.tensor(chl*32/16,dtype=int), window_size=1, shift_size=0,mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.trans3_2=TransformerBlock(dim=32*chl, input_resolution=[16,16], num_heads=torch.tensor(chl*32/16,dtype=int), window_size=1, shift_size=0,mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.unembed3=PatchUnEmbed(in_chans=32*chl)
        self.convt3=convwithactivation(8*chl,8*chl,kernel_size=[3,1,1],padding=[1,0,0],stride=[1,1,1])
        
        self.conv8=deconvwithactivation(8*chl,8*chl,kernel_size=[1,3,3],padding=[0,1,1],stride=[1,1,1])
        
        ##48*32*32->48*64*64
        self.conv9=upconvwithactivation(8*chl,4*chl,kernel_size=[1,3,3],stride=1,padding=[0,1,1],scale_factor=[1,2,2])
        self.conv10=deconvwithactivation(8*chl,4*chl,kernel_size=3,padding=1,stride=1)
        self.conv10_2=deconvwithactivation(4*chl,4*chl,kernel_size=3,padding=1,stride=1)
        ##48*64*64->48*128*128
        self.conv11=upconvwithactivation(4*chl,2*chl,kernel_size=[1,3,3],stride=1,padding=[0,1,1],scale_factor=[1,2,2])
        self.conv12=deconvwithactivation(4*chl,2*chl,kernel_size=3,padding=1,stride=1)
        self.conv12_2=deconvwithactivation(2*chl,2*chl,kernel_size=3,padding=1,stride=1)
        ##48*128*128->48*256*256
        self.conv13=upconvwithactivation(2*chl,1*chl,kernel_size=[1,3,3],stride=1,padding=[0,1,1],scale_factor=[1,2,2])
        ##output
        self.conv14=deconvwithactivation(2*chl,1*chl,kernel_size=3,padding=1,stride=1)
        self.conv15=deconvwithactivation(1*chl,1*chl,kernel_size=3,padding=1,stride=1)
        self.conv16=nn.Conv3d(chl,1,kernel_size=3,padding=1,stride=1)
        
        
    def forward(self,x):
        x0=self.conv1_2(self.conv1(x)) #48*256*256
        x0_t=self.convt1_3(self.convt1_2(self.convt1_1(x0)))
        
        x1=self.conv3_2(self.conv3(self.conv2(x0))) #48*128*128
        x1_t=self.convt1(self.unembed1(self.trans1_2(self.trans1_1(self.embed1(x1))),[64,64]))+x1 #48*128*128
        
        x2=self.conv5_2(self.conv5(self.conv4(x1))) #48*64*64
        x2_t=self.convt2(self.unembed2(self.trans2_2(self.trans2_1(self.embed2(x2))),[32,32]))+x2 #48*64*64
        
        x3=self.conv7_2(self.conv7(self.conv6(x2))) #48*32*32
        x3_t=self.convt3(self.unembed3(self.trans3_2(self.trans3_1(self.embed3(x3))),[16,16]))+x3 #48*32*32
        
        x4=self.conv9(self.conv8(x3_t)) #48*64*64
        x5=self.conv11(self.conv10_2(self.conv10(torch.cat([x4,x2_t],dim=1))))#48*128*128
        x6=self.conv13(self.conv12_2(self.conv12(torch.cat([x5,x1_t],dim=1))))#48*256*256
        
        y=self.conv16(self.conv15(self.conv14(torch.cat([x6,x0_t],dim=1))))+x
        
        return y