import torch.nn as nn
from timm.models.layers import DropPath
import torch
from functools import partial

class PatchEmbed_3d(nn.Module):
    """
    3D Volume to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=1, embed_dim=4096, norm_layer=None):
        # embed_dim = 16*16*16 = 4096 token在flatten之后的长度
        super().__init__()
        img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        # 一共有多少个token(patche)  (224/16)*(224/16)*(224/16) = 14*14*14 = 2744

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, P = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 输入图片的大小必须是固定的
        
        # x 大小224*224*224 经过k=16,s=16,c=4096的卷积核之后大小为 14*14*14*4096
        # flatten: [B, C, H, W, P] -> [B, C, HWP]   [B, 4096, 14, 14, 14] -> [B, 4096, 2744]
        # 对于Transfoemer模块，要求输入的是token序列，即 [num_token,token_dim] = [2744,4096]
        # transpose: [B, C, HWP] -> [B, HWP, C]   [B, 4096, 2744] -> [B, 2744, 4096]
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=256, 
                 patch_size=16, 
                 in_c=3, 
                 embed_dim=16*16*16, 
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4.0, 
                 qkv_bias=True,
                 qk_scale=None, 
                 drop_ratio=0.,
                 attn_drop_ratio=0., 
                 drop_path_ratio=0., 
                 embed_layer=PatchEmbed_3d, 
                 norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 # num_tokens = 1
        self.patch_size = patch_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) # 默认参数
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # token/patch的个数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # parameter构建可训练参数，第一个1是batch size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # 位置编码的大小和加入分类token之后的大小相同
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 构建等差序列，dropout率是递增的
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()
        

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit_weights)

    @staticmethod
    def _init_vit_weights(m):
        """
        ViT weight initialization
        :param m: module
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
   

    def forward(self, x):
        # [B, C, H, W, D] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # 把cls_token复制batch_size份
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]        

        x = self.pos_drop(x + self.pos_embed) 
        x = self.blocks(x)
        x = self.norm(x)
        
        x = self.pre_logits(x[:, 0])

        x = x.reshape(x.shape[0], self.embed_dim // self.patch_size ** 3, self.patch_size, self.patch_size, self.patch_size) # [B, 16, 16, 16]
        
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4., # 第一个全连接层节点个数是输入的四倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] 调整顺序
        print("C:",C)
        print("self.num_heads:",self.num_heads)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale # q dot-product k的转置，只对最后两个维度进行操作
        attn = attn.softmax(dim=-1) # 对每一行进行softmax
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim] 将多头的结果拼接在一起
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__=='__main__':
    model = VisionTransformer(img_size=256, patch_size=16, in_c=1, embed_dim=8*16*16*16, depth=12, num_heads=16, mlp_ratio=4.0)
    model.to("cuda: 0")
    input = torch.randn(1, 1, 256, 256, 256).to("cuda: 0")
    # input = torch.randn(1, 4096, 4096)
    # model = Attention(4096)
    output = model(input)
    print(output.shape)