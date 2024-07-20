import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from torchdiffeq import odeint, odeint_adjoint
from scene.cameras import Camera

import math


class CRiMKernel(nn.Module):
    def __init__(self, 
                 num_views: int = None,
                 view_dim: int = 32,
                 num_warp: int = 9,
                 method: str = 'euler',
                 adjoint: bool = False,
                 iteration: int = None,
                 ) -> None:
        super(CRiMKernel, self).__init__()

        self.num_warp = num_warp
        self.model = CRiM(num_views=num_views,
                          view_dim=view_dim,
                          num_warp=num_warp,
                          method=method,
                          adjoint=adjoint)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.lr_factor = 0.01 ** (1 / iteration)
    
    def get_warped_cams(self,
                        cam : Camera = None
                        ):
        
        Rt = self.get_Rt(cam)
        idx_view = cam.uid
        warped_Rt, ortho_loss = self.model(Rt, idx_view)
        warped_cams = [self.get_cam(cam, warped_Rt[i]) for i in range(self.num_warp)]

        return warped_cams, ortho_loss
    
    def get_cam(self,
                cam: Camera = None,
                Rt: torch.Tensor = None
                ) -> Camera:
        
        return Camera(colmap_id=cam.colmap_id, R=Rt[:3, :3], T=Rt[:3, 3], FoVx=cam.FoVx, FoVy=cam.FoVy,
                      image=cam.image, gt_alpha_mask=cam.gt_alpha_mask, image_name=cam.image_name,
                      uid=cam.uid, data_device=cam.data_device)

    def get_Rt(self, 
               cam: Camera = None
               ) -> torch.Tensor:
        R, T = cam.R, cam.T
        Rt = np.concatenate([R, T[:, None]], axis=-1)
        Rt_fill = np.array([0, 0, 0, 1])[None]
        Rt = np.concatenate([Rt, Rt_fill], axis=0)
        Rt = torch.tensor(Rt, dtype=torch.float32).cuda()
        return Rt
    
    def get_weight_and_mask(self,
                            img: torch.Tensor = None,
                            idx_view: int = None,
                            ):
        weight, mask = self.model.get_weight_and_mask(img, idx_view)
        return weight, mask

    def adjust_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_factor


class WV_Derivative(nn.Module):
    def __init__(self,
                 view_dim: int = 32,
                 num_views: int = 29,
                 num_warp: int = 5,
                 time_dim: int = 8,
                 ) -> None:
        super(WV_Derivative, self).__init__()

        self.view_dim = view_dim
        self.num_views = num_views
        self.num_warp = num_warp

        self.time_embedder = nn.Parameter(
            torch.zeros(num_warp, time_dim).type(torch.float32), 
            requires_grad=True
        )
        self.w_linear = nn.Linear(view_dim // 2 + time_dim, view_dim // 2)
        self.v_linear = nn.Linear(view_dim // 2 + time_dim, view_dim // 2)

        self.relu = nn.ReLU()

    def forward(self,
                t: int = 0,
                x: torch.Tensor = None,
                ) -> torch.Tensor:
        
        t_embed = self.time_embedder[int(t)]
        x = self.relu(x)

        w, v = torch.chunk(x, 2, dim=-1)

        w = torch.cat([w, t_embed], dim=-1)
        v = torch.cat([v, t_embed], dim=-1)

        w, v = self.w_linear(w), self.v_linear(v)

        return torch.cat([w, v], dim=-1)
    

class DiffEqSolver(nn.Module):
    def __init__(self, 
                 odefunc: nn.Module = None,
                 method: str = 'euler',
                 odeint_rtol: float = 1e-4,
                 odeint_atol: float = 1e-5,
                 num_warp: int = 5,
                 adjoint: bool = False,
                 ) -> None:
        super(DiffEqSolver, self).__init__()
        
        self.ode_func = odefunc
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.integration_time = torch.arange(0, num_warp, dtype=torch.long)
        self.solver = odeint_adjoint if adjoint else odeint
            
    def forward(self, 
                x: torch.Tensor = None,
                ) -> torch.Tensor:
        '''
        x                     : [ view_dim ]
        out                   : [ num_warp, view_dim ]
        '''
        self.integration_time = self.integration_time.type_as(x)
        out = self.solver(self.ode_func, x, self.integration_time.cuda(x.get_device()),
                     rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        return out


class CRiM(nn.Module):
    def __init__(self,
                 num_views: int = 29,
                 view_dim: int = 32,
                 num_warp: int = 9,
                 method: str = 'euler',
                 adjoint: bool = False,
                 ) -> None:
        super(CRiM, self).__init__()

        self.num_warp = num_warp

        self.view_embedder = nn.Parameter(
            torch.zeros(num_views, view_dim).type(torch.float32), 
            requires_grad=True
        )

        self.view_encoder = nn.ModuleList()
        self.Rt_encoder = nn.ModuleList()
        self.wv_derivative = nn.ModuleList()
        self.diffeq_solver = nn.ModuleList()
        self.rot_decoder = nn.ModuleList()
        self.trans_decoder = nn.ModuleList()
        self.theta_decoder = nn.ModuleList()

        self.mlpWeight = nn.ModuleList()
        self.mlpMask = nn.ModuleList()

        for i in range(num_views):
            
            self.Rt_encoder.append(nn.Linear(12, view_dim))
            self.view_encoder.append(nn.Linear(view_dim + view_dim, view_dim))
            self.wv_derivative.append(WV_Derivative(view_dim=view_dim, num_views=num_views, num_warp=num_warp))
            self.diffeq_solver.append(DiffEqSolver(odefunc=self.wv_derivative[i], method=method, num_warp=num_warp, adjoint=adjoint))
            self.rot_decoder.append(nn.Linear(view_dim // 2, 12))
            self.trans_decoder.append(nn.Linear(view_dim // 2, 6))
            self.theta_decoder.append(nn.Linear(view_dim // 2, 1))

            gain = 0.00001 / (math.sqrt((view_dim // 2 + 3) / 6))
            torch.nn.init.xavier_uniform_(self.rot_decoder[i].weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.trans_decoder[i].weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.theta_decoder[i].weight, gain=gain)
            self.rot_decoder[i].bias.data.fill_(0)
            self.trans_decoder[i].bias.data.fill_(0)
            self.theta_decoder[i].bias.data.fill_(0)

            self.mlpWeight.append(nn.Conv2d(64, 1, 1, bias=False))
            self.mlpMask.append(nn.Conv2d(64 * self.num_warp, 1, 1, bias=False))


        # conv, mlp_weight, mlp_mask from BAGS (https://github.com/snldmt/BAGS/)
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )

        self.mlp_weight = nn.Conv2d(64, 1, 1, bias=False)
        self.mlp_mask = nn.Conv2d(64 * self.num_warp, 1, 1, bias=False)


    def get_weight_and_mask(self, 
                            img: torch.Tensor = None,
                            idx_view: int = None,
                            ):
        '''
        Input:
            img                   : [ num_warp, 3, H, W ]
        Output:
            weight                : [ num_warp, 1, H, W ]
            mask                  : [ 1, H, W ]
        '''

        feat = self.conv(img)
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        weight = self.mlp_weight(feat)
        weight = F.softmax(weight, dim=0)
        feat = rearrange(feat, 't c h w -> 1 (t c) h w')
        mask = torch.sigmoid(self.mlp_mask(feat)[0])

        return weight, mask
        
        
    def forward(self,
                Rt: torch.Tensor = None,
                idx_view: int = None,
                ) -> torch.Tensor:
        
        '''
        new_Rt, w_loss, kernel_weights, kernel_mask = kernelnet(Rt, viewpoint_cam.uid, image.detach(), depth.detach())
        Input:
            Rt           : [4, 4]
            idx_view     : scalar
            image        : [3, 400, 600]
            depth        : [1, 400, 600]
        Output:
            new_Rt        : [num_warp, 4, 4]
            w_loss        : scalar
            kernel_weights: [num_warp, 1, 400, 600]
            kernel_mask   : [1, 400, 600]
        '''

        view_embed = self.view_embedder[idx_view]
        Rt_encoded = self.Rt_encoder[idx_view](Rt[:3, :].reshape(-1))
        view_embed = torch.cat([view_embed, Rt_encoded], dim=-1)

        view_encoded = self.view_encoder[idx_view](view_embed)
        latent_wv = self.diffeq_solver[idx_view](view_encoded)
        latent_w, latent_v = torch.chunk(latent_wv, 2, dim=-1)

        w = self.rot_decoder[idx_view](latent_w)
        w_rigid, w_distort = w[:, :3], w[:, 3:]
        theta = self.theta_decoder[idx_view](latent_w)[..., None]

        v = self.trans_decoder[idx_view](latent_v)
        v_rigid, v_distort = v[:, :3], v[:, 3:]

        w_norm = self.exp_map(w_rigid)
        w_skew = self.skew_symmetric(w_norm)
        R_exp = self.rodrigues_formula(w_skew, theta)
        G = self.G_formula(w_skew, theta)
        p = torch.matmul(G, v_rigid[..., None])
        Rt_rigid = self.transform_SE3(R_exp, p)

        eye = torch.eye(3)[None, ...].repeat(self.num_warp, 1, 1).to(w)
        R_distort = eye + w_distort.reshape(-1, 3, 3)
        Rt_distort = self.transform_SE3(R_distort, v_distort[..., None])
        
        Rt_transform = torch.matmul(Rt_rigid, Rt_distort)

        Rt_new = torch.einsum('ij, tjk -> tik', Rt, Rt_transform)

        w_loss = torch.matmul(R_distort, R_distort.transpose(1, 2))
        w_loss = (w_loss - eye).abs().mean()

        return Rt_new, w_loss
    
    def transform_SE3(self, 
                      exp_w_skew: torch.Tensor, 
                      p: torch.Tensor
                      ) -> torch.Tensor:
        
        delta_Rt = torch.cat([exp_w_skew, p], dim=-1)
        delta_Rt_fill = torch.tensor([0, 0, 0, 1])[None].repeat(delta_Rt.size(0), 1, 1).to(delta_Rt)
        delta_Rt = torch.cat([delta_Rt, delta_Rt_fill], dim=1)
        return delta_Rt
    
    def rodrigues_formula(self, 
                          w: torch.Tensor, 
                          theta: torch.Tensor,
                          ) -> torch.Tensor:
        
        term1 = torch.eye(3).to(w)
        term2 = torch.sin(theta) * w
        term3 = (1 - torch.cos(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3
    
    def G_formula(self,
                  w: torch.Tensor, 
                  theta: torch.Tensor,
                  ) -> torch.Tensor:
        term1 = torch.eye(3)[None].to(w) * theta
        term2 = (1 - torch.cos(theta)) * w
        term3 = (theta - torch.sin(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3

    def exp_map(self, 
                w: torch.Tensor,
                ) -> torch.Tensor:
        norm = torch.norm(w, dim=-1)[..., None] + 1e-10
        w = w / norm
        return w

    def skew_symmetric(self, 
                       w : torch.Tensor,
                       ) -> torch.Tensor:
        
        w1, w2, w3 = torch.chunk(w, 3, dim=-1)

        w_skew =  torch.cat([torch.zeros_like(w1), -w3, w2,
                             w3, torch.zeros_like(w1), -w1,
                             -w2, w1, torch.zeros_like(w1)], dim=-1)
        w_skew = w_skew.reshape(-1, 3, 3)
        return w_skew
    

    

if __name__ == '__main__':
    model = CRiM().cuda()
    Rt = torch.rand(12).reshape(3, 4).cuda()
    Rt_fill = torch.tensor([0, 0, 0, 1])[None].to(Rt)
    Rt = torch.cat([Rt, Rt_fill], dim=0)
    idx_view = 0

    new = model(Rt, idx_view)

    import pdb; pdb.set_trace()
