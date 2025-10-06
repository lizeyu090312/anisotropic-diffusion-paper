import math, torch, torch.nn as nn
#from torch.func import vmap
from torch.autograd.functional import jvp as jvp_autograd
from torch.func import functional_call
import torch.nn.functional as F

class GFn(nn.Module):
    """Scalar variance schedule g(t) with exponential (log‑linear) interpolation.

    times[0]  must be 0.0  (fixed g=1)
    times[-1] must be T    (fixed g=T)
    """

    def __init__(self, times: torch.Tensor, T: float, initg=None, g0 = 1e-9, device='cuda'):
    
        super().__init__()
        assert torch.isclose(times[0], torch.tensor(0.0, dtype=times.dtype)), "times[0] must be 0.0"
        assert torch.isclose(times[-1], torch.tensor(T, dtype=times.dtype)),  "times[-1] must be T"

        self.register_buffer("times", times.clone())
        self.T = torch.tensor([T],device=times.device)
        self.K = len(times)
        self.g0 = g0

        # default initial schedule: linear in arithmetic space between 2 and T
        if initg is None:
            g_init = torch.logspace(math.log10(self.g0), math.log10(T), self.K, device=device)
        else:
            g_init = initg
        assert torch.isclose(g_init[0], torch.tensor(self.g0, device=times.device))
        assert torch.isclose(g_init[-1], torch.tensor(T   , device=times.device))
        assert g_init.shape == times.shape
        log_init = torch.log(g_init)

        # learnable positive increments for knots 1 .. K‑2
        delta_init = log_init[1:] - log_init[:-1]
        def _inv_softplus(y):
            # numerically stable inverse of softplus
            return y + torch.log(-torch.expm1(-y))        # = log(eʸ – 1)
        delta_init = log_init[1:] - log_init[:-1]         # positive increments
        delta_param0 = _inv_softplus(delta_init)          # undo the softplus
        self.delta_param = nn.Parameter(delta_param0.clone().detach())

        self.register_buffer("eps", torch.tensor(1e-8, dtype=times.dtype))

    # ------------------------------------------------------------------
    # helper: build strictly increasing log_g vector of length K
    # ------------------------------------------------------------------
    def _log_knots(self):
        log_g0 = torch.tensor([math.log(self.g0)], device=self.times.device)  # g(0)=1

        # interior raw positive increments
        raw_inc = torch.nn.functional.softplus(self.delta_param) + self.eps  # [K-1]
        S = raw_inc.sum()

        target_gap = math.log(self.T) - math.log(self.g0)  # total amount to climb
        #alpha = torch.minimum(torch.tensor(1.0, device=self.times.device), target_gap / S + 1e-12)
        alpha = target_gap / S
        scaled = raw_inc * alpha  # scaled so that sum ≤ target

        tail_inc = self.eps  # guarantees exact landing

        log_g = torch.cumsum(torch.cat([log_g0, scaled, tail_inc.unsqueeze(0)]), dim=0)
        return log_g

    def _interp(self, t: torch.Tensor):
        logk = self._log_knots()  # (K,)
        flat = t.flatten()
        idx = torch.searchsorted(self.times, flat, right=False).clamp(1, self.K - 1)

        t0, t1 = self.times[idx - 1], self.times[idx]
        l0, l1 = logk[idx - 1], logk[idx]
        w = (flat - t0) / (t1 - t0)

        logg = l0 + w * (l1 - l0)
        g = torch.exp(logg)
        g_dot = g * (l1 - l0) / (t1 - t0)
        return g.view_as(t), g_dot.view_as(t)

    # public API
    def forward(self, t):
        return self._interp(t)[0]

    def g_and_grad(self, t):
        return self._interp(t)

def compute_DCT_basis(k: int, d: int, *, dtype=torch.float32, device="cuda") -> torch.Tensor:
    """
    Build the k lowest‐frequency 2D DCT basis functions on a d×d grid,
    grouping frequencies by f = max(u,v) so that k = (i+1)^2 for some i.
    """
    m = int(round(math.sqrt(k)))
    if m * m != k:
        raise ValueError(f"k={k} is not of the form (i+1)^2; sqrt(k)={math.sqrt(k):.3f}")
    i = m - 1

    basis_list = []
    for u in range(i + 1):
        alpha_u = math.sqrt(1.0 / d) if u == 0 else math.sqrt(2.0 / d)
        x = torch.arange(d, dtype=dtype, device=device).unsqueeze(1)  # (d,1)
        cos_u = alpha_u * torch.cos((2 * x + 1) * u * math.pi / (2 * d))

        for v in range(i + 1):
            alpha_v = math.sqrt(1.0 / d) if v == 0 else math.sqrt(2.0 / d)
            y = torch.arange(d, dtype=dtype, device=device).unsqueeze(0)  # (1,d)
            cos_v = alpha_v * torch.cos((2 * y + 1) * v * math.pi / (2 * d))

            basis_uv = cos_u @ cos_v  # (d,d)
            basis_list.append(basis_uv)

    # Stack into [k, d, d]
    return torch.stack(basis_list, dim=0)

# compute projection onto DCT basis represented by V
def proj_dct(V: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection onto span(V).
    X : (B, C, W, W)
    V : (K, W, W)
    """
    coeffs = torch.einsum('BCij,Kij->BCK',(X, V))
    VX = torch.einsum('BCK,Kij->BCij',(coeffs, V))
    return VX

# gh: tuple of g and h tensors, to form M = gV + h(I-V)
def mat_mul(gh, V, X, power: float = 1.0, identity_scaling: float = 0.0):
    """
    Compute  (M_t + s·I)^power  X      with  s = identity_scaling.
    * g  : 1-D tensor [B]
    * h  : 1-D tensor [B]
    * V  : [K,W,W]
    * X  : [B,C,W,W]
    * power, identity_scaling : floats
    """
    # unpack the time input
    (g_t,h_t) = gh

    # check shape
    B = X.shape[0]
    assert g_t.numel() == B and h_t.numel() == B

    # add the scaled identity before taking the power
    scale_U = (h_t + identity_scaling).pow(power).view(B, 1, 1, 1)
    scale_V = (g_t + identity_scaling).pow(power).view(B, 1, 1, 1)

    VX = proj_dct(V, X)          # V X
    UX = X - VX                  # U X
    return scale_U * UX + scale_V * VX


class ANI_absM_Precond_Wrapper(torch.nn.Module):
    def __init__(self,
                 edm_precond, V):
        super().__init__()
        # --------- reuse core parameters ---------------------------------
        self.img_resolution = edm_precond.img_resolution
        self.img_channels   = edm_precond.img_channels
        self.label_dim      = edm_precond.label_dim
        self.use_fp16       = edm_precond.use_fp16
        self.sigma_min      = edm_precond.sigma_min
        self.sigma_max      = edm_precond.sigma_max
        self.sigma_data     = edm_precond.sigma_data
        # --------- copy the trained UNet ---------------------------------
        self.model = edm_precond.model   # keep weights
        self.V = V # V : [K,W,W], dct basis
    # ------------------------------------------------------------------
    # forward :  returns  x + ∇ log p_t(x)
    # ------------------------------------------------------------------
    def forward(
        self,
        x,                              # [B,C,W,W]  noisy   x_t
        gh,                             # (g,h). g: [B]. h: [B]. g, h are SIGMA^2
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        g_t, h_t = (gh)
        sigma_g_t = (g_t).sqrt()
        sigma_h_t = (h_t).sqrt()
        c_noise = (sigma_g_t.log() + sigma_h_t.log()) /8
        dtype = (torch.float16
                 if self.use_fp16 and not force_fp32 and x.is_cuda
                 else torch.float32)
        if class_labels is not None:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)
        z_t = mat_mul((1/(g_t+self.sigma_data**2).sqrt(), 1/(h_t+self.sigma_data**2).sqrt()), self.V, x, power = 1.0, identity_scaling = 0.0)
        F_x = self.model(
            z_t.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        ).to(torch.float32)
        c_out_g = g_t.sqrt() * self.sigma_data / (g_t + self.sigma_data ** 2).sqrt()
        c_out_h = h_t.sqrt() * self.sigma_data / (h_t + self.sigma_data ** 2).sqrt()
        c_out_F_x = mat_mul((c_out_g,c_out_h), self.V, F_x, power = 1.0, identity_scaling = 0.0)
        c_skip_g = self.sigma_data ** 2 / (g_t + self.sigma_data ** 2)
        c_skip_h = self.sigma_data ** 2 / (h_t + self.sigma_data ** 2)
        c_skip_x = mat_mul((c_skip_g,c_skip_h), self.V, x, power = 1.0, identity_scaling = 0.0)
        D_x = c_skip_x + c_out_F_x
        return D_x
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma) 
    
#########
## net is ANI_absM_Precond, returns E[x_0]
#########
def M_euler_ode(edmnet, latents, t_vec, g_fn, h_fn, labels):
    g, _ = g_fn.g_and_grad(t_vec)
    h, _ = h_fn.g_and_grad(t_vec)
    g = g.detach()
    h = h.detach()

    xk = latents * g[-1].sqrt()
    B = xk.shape[0]
    for k in range(t_vec.shape[0]-1,0,-1):
        gk = g[k].unsqueeze(0).expand(B)
        hk = h[k].unsqueeze(0).expand(B)
        with torch.no_grad():
            M_times_score = edmnet(xk, (gk,hk), class_labels=labels) - xk
            flow_dir = mat_mul((gk, hk), edmnet.V, M_times_score, power = -0.5, identity_scaling = 0)
        gk_next = g[k-1].unsqueeze(0).expand(B)
        hk_next = h[k-1].unsqueeze(0).expand(B)

        xk = xk + mat_mul((gk_next.sqrt() - gk.sqrt(), hk_next.sqrt() - hk.sqrt()), 
                          edmnet.V, -flow_dir, power = 1.0, identity_scaling = 0)
    return xk.clamp(-1, 1)  

def M_EDM_ode(net, latents, t_vec, g_fn, h_fn, labels):
    g, _ = g_fn.g_and_grad(t_vec)
    h, _ = h_fn.g_and_grad(t_vec)
    
    #optional
    g = torch.cat([torch.zeros(1,device=g.device), g])
    h = torch.cat([torch.zeros(1,device=h.device), h])
    g = g.detach()
    h = h.detach()


    ### OPTION 1: average in t space
    t_vec = torch.cat([torch.zeros(1,device=g.device), t_vec])
    t_half = (t_vec[1:]+t_vec[:-1])/2
    g_half, _ = g_fn.g_and_grad(t_half)
    #g_half = torch.cat([g[1].unsqueeze(0)*0.5, g_half])
    h_half, _ = h_fn.g_and_grad(t_half)
    #h_half = torch.cat([h[1].unsqueeze(0)*0.5, h_half])
    
    ### OPTION 2: average in sqrt space
    #g_half = ((g[1:].sqrt()+g[:-1].sqrt())/2)**2
    #h_half = ((h[1:].sqrt()+h[:-1].sqrt())/2)**2

    ### OPTION 3: same as EDM
    #g_half = g[:-1]
    #h_half = h[:-1]

    # Main sampling loop.
    x_next = latents * g[-1].sqrt()
    B = x_next.shape[0]

    with torch.no_grad():
        for k in range(g.shape[0]-1,0,-1):
            gk = g[k].unsqueeze(0).expand(B)
            hk = h[k].unsqueeze(0).expand(B)
            gk_next = g[k-1].unsqueeze(0).expand(B)
            hk_next = h[k-1].unsqueeze(0).expand(B)
            gk_half = g_half[k-1].unsqueeze(0).expand(B)
            hk_half = h_half[k-1].unsqueeze(0).expand(B)
            
            x_cur = x_next
            x_hat = x_cur
        
            M_times_score = net(x_hat, (gk,hk), class_labels=labels) - x_hat
            flow_dir = mat_mul((gk, hk), net.V, M_times_score, power = -0.5, identity_scaling = 0)
            
            x_half = x_hat + mat_mul((gk_half.sqrt() - gk.sqrt(), hk_half.sqrt() - hk.sqrt()), 
                              net.V, -flow_dir, power = 1.0, identity_scaling = 0)
            
            if k-1 > 0:
                M_times_score_half = net(x_half, (gk_half,hk_half), class_labels=labels) - x_half
                flow_dir_half = mat_mul((gk_half, hk_half), net.V, M_times_score_half, power = -0.5, identity_scaling = 0)
                ratio_g = (gk_next.sqrt() - gk.sqrt())/(gk_half.sqrt() - gk.sqrt())
                ratio_h = (hk_next.sqrt() - hk.sqrt())/(hk_half.sqrt() - hk.sqrt())
                delta_flow_dir = mat_mul((ratio_g, ratio_h), net.V, (flow_dir_half - flow_dir))

                x_next = x_hat + mat_mul((gk_next.sqrt() - gk.sqrt(), hk_next.sqrt() - hk.sqrt()), 
                                  net.V, -(flow_dir + 0.5*delta_flow_dir))
            else:
                x_next = x_half

    return x_next

def ANILoss_gh_energy(edmnet, images, labels, g_fn, h_fn, T, tmin=1e-9):
    """
    images : [B,C,W,W]  in [-1,1]
    returns per-pixel squared error tensor (identical convention as EDMLoss)
    """
    B, _, _, _ = images.shape
    device = images.device

    t_vec = torch.rand(B,device=device)*(T-tmin)+tmin
    g, dot_g = g_fn.g_and_grad(t_vec)
    h, dot_h = h_fn.g_and_grad(t_vec)
    g = g
    h = h
    dot_g = dot_g
    dot_h = dot_h

    if labels is not None:
        labels = labels.detach()
    eps = torch.randn_like(images)        #eps ~ N(0,I)
    x_noisy = images + mat_mul((g,h), edmnet.V, eps, power=0.5, identity_scaling=0.0)
    den = edmnet(x_noisy, (g,h), class_labels=labels)
    diff = den - images     # model residual

    sigma_data = 1

    weighted_diff = mat_mul(((dot_g**2)/(g**2)/(g+sigma_data**2),(dot_h**2)/(h**2)/(h+sigma_data**2)), edmnet.V, diff, power=0.5, identity_scaling=0.0) 
    loss = weighted_diff.pow(2).sum(dim=[1,2,3])

    return loss.mean()

def ANILoss_gh_energy_plus_discretization(edmnet, images, labels, g_fn, h_fn, T, tmin=1e-9, disc_mult=1.0):
    """
    images : [B,C,W,W]  in [-1,1]
    returns per-pixel squared error tensor (identical convention as EDMLoss)
    """
    B, _, _, _ = images.shape
    device = images.device

    t_vec = torch.rand(B,device=device)*(T-tmin)+tmin
    g, dot_g = g_fn.g_and_grad(t_vec)
    h, dot_h = h_fn.g_and_grad(t_vec)

    if labels is not None:
        labels = labels.detach()
    eps = torch.randn_like(images)        #eps ~ N(0,I)
    x_noisy = images + mat_mul((g,h), edmnet.V, eps, power=0.5, identity_scaling=0.0)
    
    #####################
    ### compute detached score
    #####################
    detached = {k: v.detach() for k,v in edmnet.named_parameters()}
    M_times_score_detached = functional_call(edmnet, detached, (x_noisy, (g, h)),
                         {"class_labels": labels}) - x_noisy
    
    #####################
    ### dotgsqogsqogp1: loss wrt true score  (3.1 FID)
    #####################
    scaled_net_score = mat_mul((dot_g/(g)/((g+0.5)**0.5),dot_h/(h)/((h+0.5)**0.5)), edmnet.V, M_times_score_detached) 
    scaled_true_noisy_score = mat_mul((dot_g/(g)/((g+0.5)**0.5),dot_h/(h)/((h+0.5)**0.5)), edmnet.V, images-x_noisy) 

    loss = (scaled_true_noisy_score - scaled_net_score).pow(2).sum(dim=[1,2,3]).mean() 

    loss = loss - (scaled_true_noisy_score).pow(2).sum(dim=[1,2,3]).mean() + (scaled_net_score).pow(2).sum(dim=[1,2,3]).mean()

    loss = loss

    #####################
    ### loss wrt discretization
    #####################
    interval = 6400.0/8
    delta = (torch.rand(t_vec.shape,device=device)) * interval
    #delta = (torch.rand(t_vec.shape,device=device)*0.5+0.5) * interval
    t_vec_target = torch.clamp(t_vec - delta, min=1e-9, max=6400)

    t_vec_next = t_vec + 0.5 * (t_vec_target - t_vec)

    g_next, _ = g_fn.g_and_grad(t_vec_next)
    h_next, _ = h_fn.g_and_grad(t_vec_next)

    t_rand_mid = t_vec + torch.rand(t_vec.shape,device=device) * (t_vec_target - t_vec) 
    g_rand_mid, dotg_rand_mid = g_fn.g_and_grad(t_rand_mid)
    h_rand_mid, doth_rand_mid = h_fn.g_and_grad(t_rand_mid)
    
    #g_next = (g.sqrt()-(dot_g/g.sqrt() *delta))**2
    #h_next = (h.sqrt()-(dot_h/h.sqrt() *delta))**2
    
    # compute update direction
    flow_dir = mat_mul((g, h), edmnet.V, M_times_score_detached, power = -0.5)
    x_next = x_noisy + mat_mul((g_next.sqrt() - g.sqrt(), h_next.sqrt() - h.sqrt()), 
                      edmnet.V, -flow_dir, power = 1.0)
    M_times_score_next_detached = functional_call(edmnet, detached, (x_next, (g_next, h_next)),
                         {"class_labels": labels}) - x_next
    flow_dir_next = mat_mul((g_next, h_next), edmnet.V, M_times_score_next_detached, power = -0.5)

    # estimate d/ds flow
    dt_flow_dir = mat_mul((1/(g_next.sqrt()-g.sqrt()),1/(h_next.sqrt()-h.sqrt())), edmnet.V, 0.5 * (flow_dir_next - flow_dir)) 

    #disc_err = mat_mul(((dot_g)/(g**0.5)/((g+1)**0.5),(dot_h)/(h**0.5)/((h+1)**0.5)), edmnet.V, (flow_dir - flow_dir_next)) 

    displacement_at_t_mid = (mat_mul((g_rand_mid.sqrt()-g.sqrt(),h_rand_mid.sqrt()-h.sqrt()), edmnet.V, - flow_dir) 
                     + mat_mul(((g_rand_mid.sqrt()-g.sqrt())**2,(h_rand_mid.sqrt()-h.sqrt())**2), edmnet.V, - 0.5 * dt_flow_dir))
    x_rand_mid = x_noisy + displacement_at_t_mid
    
    velocity_at_t_mid = mat_mul((dotg_rand_mid/(g_rand_mid**0.5),doth_rand_mid/(h_rand_mid**0.5)), edmnet.V, 
                     - flow_dir + mat_mul(((g_rand_mid.sqrt()-g.sqrt()),(h_rand_mid.sqrt()-h.sqrt())), edmnet.V, - dt_flow_dir))

    M_times_score_rand_mid = functional_call(edmnet, detached, (x_rand_mid, (g_rand_mid, h_rand_mid)),
                         {"class_labels": labels}) - x_rand_mid
    
    true_velocity_at_t_mid = mat_mul((dotg_rand_mid/g_rand_mid,doth_rand_mid/h_rand_mid), edmnet.V, - M_times_score_rand_mid) 

    disc_err = mat_mul((1/(g_rand_mid+0.5)**0.5,1/(h_rand_mid+0.5)**0.5), edmnet.V, velocity_at_t_mid - true_velocity_at_t_mid)
    
    loss = loss + disc_err.norm(p=2,dim=[1,2,3]).pow(2).mean()
    return loss



#######################################################################
#######  Code for optimizing score_and_energy_loss over gh   ##########
#######################################################################
class Flow_Net(torch.nn.Module):
    def __init__(self,
                 edm_precond, V):
        super().__init__()
        # --------- reuse core parameters ---------------------------------
        self.img_resolution = edm_precond.img_resolution
        self.img_channels   = edm_precond.img_channels
        self.label_dim      = edm_precond.label_dim
        self.use_fp16       = edm_precond.use_fp16
        self.sigma_min      = edm_precond.sigma_min
        self.sigma_max      = edm_precond.sigma_max
        self.sigma_data     = 1
        # --------- copy the trained UNet ---------------------------------
        self.model = edm_precond.model   # keep weights
        self.V = V # V : [K,W,W], dct basis
    # ------------------------------------------------------------------
    # forward :  returns  x + ∇ log p_t(x)
    # ------------------------------------------------------------------
    def forward(
        self,
        x,                              # [B,C,W,W]  noisy   x_t
        gh,                             # (g,h). g: [B]. h: [B]. g, h are SIGMA^2 + 1
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        g_t, h_t = (gh)
        sigma_g_t = (g_t).sqrt()
        sigma_h_t = (h_t).sqrt()
        z_t = mat_mul((g_t, h_t), self.V, x, power = -0.5, identity_scaling = 1.0)
        # ---------- 2. noise-level embedding  -------------------------
        # use geometric mean of log(g)/4 and log(h)/4
        # if g(t)=h(t) for all t, this should be have IDENTICALLY to BASE_EDM, REGARDLESS of how g(t) varies.
        c_noise = (sigma_g_t.log() + sigma_h_t.log()) /8
        dtype = (torch.float16
                 if self.use_fp16 and not force_fp32 and x.is_cuda
                 else torch.float32)
        if self.label_dim:
            if class_labels is None:
                class_labels = torch.zeros(1, self.label_dim, device=x.device)
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)
        # ---------- 3. backbone predicts F_x = M_t^{-½}(x0 - x) -------
        F_x = self.model(
            z_t.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        ).to(torch.float32)
        return F_x
        
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
       
class ANI_absM_Precond_Flow_Net(torch.nn.Module):
    def __init__(self,
                 edm_precond, V):
        super().__init__()
        # --------- reuse core parameters ---------------------------------
        self.img_resolution = edm_precond.img_resolution
        self.img_channels   = edm_precond.img_channels
        self.label_dim      = edm_precond.label_dim
        self.use_fp16       = edm_precond.use_fp16
        self.sigma_min      = edm_precond.sigma_min
        self.sigma_max      = edm_precond.sigma_max
        self.sigma_data     = edm_precond.sigma_data
        # --------- copy the trained UNet ---------------------------------
        self.model = edm_precond.model   # keep weights
        self.V = V # V : [K,W,W], dct basis
    # ------------------------------------------------------------------
    # forward :  returns  x + ∇ log p_t(x)
    # ------------------------------------------------------------------
    def forward(
        self,
        x,                              # [B,C,W,W]  noisy   x_t
        gh,                             # (g,h). g: [B]. h: [B]. g, h are SIGMA^2
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        g_t, h_t = (gh)

        sigma_g_t = (g_t).sqrt()
        sigma_h_t = (h_t).sqrt()
        c_noise = (sigma_g_t.log() + sigma_h_t.log()) /8

        dtype = (torch.float16
                 if self.use_fp16 and not force_fp32 and x.is_cuda
                 else torch.float32)
        
        if class_labels is not None:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)

        z_t = mat_mul((1/(g_t+self.sigma_data**2).sqrt(), 1/(h_t+self.sigma_data**2).sqrt()), self.V, x, power = 1.0, identity_scaling = 0.0)

        F_x = self.model(
            z_t.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        ).to(torch.float32)

        c_out_g = g_t.sqrt() * self.sigma_data / (g_t + self.sigma_data ** 2).sqrt()
        c_out_h = h_t.sqrt() * self.sigma_data / (h_t + self.sigma_data ** 2).sqrt()
        c_out_F_x = mat_mul((c_out_g,c_out_h), self.V, F_x, power = 1.0, identity_scaling = 0.0)

        c_skip_g = self.sigma_data ** 2 / (g_t + self.sigma_data ** 2)
        c_skip_h = self.sigma_data ** 2 / (h_t + self.sigma_data ** 2)
        c_skip_x = mat_mul((c_skip_g,c_skip_h), self.V, x, power = 1.0, identity_scaling = 0.0)        

        D_x = c_skip_x + c_out_F_x
        flow_x = mat_mul((g_t, h_t), self.V, D_x-x, power = -0.5, identity_scaling = 0)
        return flow_x
        
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
def flow_matching_energy(flownet, images, labels, g_fn, h_fn, T, tmin=1e-9):
    """
    images : [B,C,W,W]  in [-1,1]
    returns scalar mean loss
    """
    B, _, _, _ = images.shape
    device = images.device
    t_vec = torch.rand(B,device=device)*(T-tmin)+tmin
    g_t, dotg_t = g_fn.g_and_grad(t_vec)
    h_t, doth_t = h_fn.g_and_grad(t_vec)

    if labels is not None:
        labels = labels.detach()
    eps = torch.randn_like(images)        
    x_noisy = images + mat_mul((g_t,h_t), flownet.V, eps, power=0.5, identity_scaling=0.0)
    flow_at_x = flownet(x_noisy, (g_t.detach(),h_t.detach()), class_labels=labels)

    target_flow = mat_mul((1/(g_t**0.5),1/(h_t**0.5)), flownet.V, images - x_noisy, power=1.0, identity_scaling=0.0) 

    # following loss does not back-prop through flownet to get time.
    flow_error_vec = mat_mul(((dotg_t)/(g_t**0.5)/(g_t+1)**0.5,(doth_t)/(h_t**0.5)/(h_t+1)**0.5), flownet.V, flow_at_x - target_flow, power=1.0, identity_scaling=0.0) 
    loss = flow_error_vec.pow(2).sum(dim=[1,2,3]).sum()
    
    # now, manually compute del_theta flow(x;theta):
    #del_theta_flow = estimate_flow_derivative_gh(flownet, x_noisy, (g_t, h_t), (dotg_t, doth_t), labels, flow_at_x.detach())
    del_flow_loss = 2 * mat_mul(((dotg_t**2)/(g_t)/(g_t+1),(doth_t**2)/(h_t)/(h_t+1)), flownet.V, flow_at_x - target_flow, power=1.0, identity_scaling=0.0).detach()
    
    additional_loss = create_flow_derivative_gh_term(flownet, x_noisy, (g_t, h_t), (dotg_t, doth_t), labels, flow_at_x, del_flow_loss)

    # additional loss term
    #additional_loss = torch.einsum('BCHW,BCHW->B',(del_flow_loss, del_theta_flow)).sum()
    assert additional_loss.item()==0

    loss = loss + additional_loss

    return loss


# weird implementation quirk, flow_at_x must be computed at x
def create_flow_derivative_gh_term(flownet, x, gh_t, dot_gh_t, labels, flow_at_x, del_flow_loss):
    dct_V = flownet.V
    B,d1,d2,d3 = x.shape
    d = d1*d2*d3
    g_t, h_t = gh_t
    dotg_t, doth_t = dot_gh_t
    assert g_t.shape[0]==B and h_t.shape[0]==B and dotg_t.shape[0]==B and doth_t.shape[0]==B

    # Sample random perturbation directions ei_s for each x:
    ei = torch.randn_like(x)
    ei = ei/ei.norm(p=2,dim=(1,2,3))[:,None,None,None]

    # compute the detached flow, if not provided
    flow_detached = flow_at_x.detach()
    del_flow_loss = del_flow_loss.detach()

    # compute del_x <del_flow_loss.detach(), flow(x, gh, label)
    F_b = (del_flow_loss * flow_at_x).sum()
    (del_x_F_b,) = torch.autograd.grad(F_b, x, create_graph=True, retain_graph=True)
    
    ################################################################
    # 1. form derivative term involving drds flow(x + rei + sdel_theta M ei)
    ################################################################
    dot_del_x_F_b_ei = (del_x_F_b * ei).sum()
    (del_x_dot_del_x_F_b_ei,) = torch.autograd.grad(dot_del_x_F_b_ei, x, create_graph=False, retain_graph=True)
    dot_del_x_dot_del_x_F_b_ei_Mei = (del_x_dot_del_x_F_b_ei.detach() * mat_mul((g_t, h_t), dct_V, ei, power=1.0, identity_scaling=0.0)).sum()

    ################################################################
    # 2. form derivative term involving ds flow(x + sdel_theta M score)
    ################################################################
    dot_del_x_F_b_Mscore = (del_x_F_b.detach() * mat_mul((g_t*g_t.detach()**0.5, h_t*h_t.detach()**0.5), dct_V, flow_detached, power=1.0, identity_scaling=0.0)).sum()
    
    ################################################################
    # 3. form derivative term involving dtheta M
    ################################################################
    F_c = (del_flow_loss * mat_mul((g_t/g_t.detach(), h_t/h_t.detach()), dct_V, flow_detached, power=1.0, identity_scaling=0.0)).sum()

    ########################################
    # 4. Combining everything
    ########################################
    additional_loss = 0.5 * d * dot_del_x_dot_del_x_F_b_ei_Mei + dot_del_x_F_b_Mscore + 0.5 * F_c

    return additional_loss-additional_loss.detach()

def ANILoss_gh_energy_all_version7wud(edmnet, images, labels, g_fn, h_fn, T, tmin=1e-9, disc_mult=1.0):
    """
    images : [B,C,W,W]  in [-1,1]
    returns per-pixel squared error tensor (identical convention as EDMLoss)
    """
    B, _, _, _ = images.shape
    device = images.device

    t_vec = torch.rand(B,device=device)*(T-tmin)+tmin
    g, dot_g = g_fn.g_and_grad(t_vec)
    h, dot_h = h_fn.g_and_grad(t_vec)

    if labels is not None:
        labels = labels.detach()
    eps = torch.randn_like(images)        #eps ~ N(0,I)
    x_noisy = images + mat_mul((g,h), edmnet.V, eps, power=0.5, identity_scaling=0.0)
    
    #####################
    ### compute detached score
    #####################
    detached = {k: v.detach() for k,v in edmnet.named_parameters()}
    M_times_score_detached = functional_call(edmnet, detached, (x_noisy, (g, h)),
                         {"class_labels": labels}) - x_noisy
    
    #####################
    ### dotgsqogsqogp1: loss wrt true score  (3.1 FID)
    #####################
    scaled_net_score = mat_mul((dot_g/(g)/((g+0.5)**0.5),dot_h/(h)/((h+0.5)**0.5)), edmnet.V, M_times_score_detached) 
    scaled_true_noisy_score = mat_mul((dot_g/(g)/((g+0.5)**0.5),dot_h/(h)/((h+0.5)**0.5)), edmnet.V, images-x_noisy) 

    loss = (scaled_true_noisy_score - scaled_net_score).pow(2).sum(dim=[1,2,3]).mean() 

    loss = loss - (scaled_true_noisy_score).pow(2).sum(dim=[1,2,3]).mean() + (scaled_net_score).pow(2).sum(dim=[1,2,3]).mean()

    #####################
    ### loss wrt discretization
    #####################
    interval = 6400.0/8
    delta = (torch.rand(t_vec.shape,device=device)) * interval
    #delta = (torch.rand(t_vec.shape,device=device)*0.5+0.5) * interval
    t_vec_target = torch.clamp(t_vec - delta, min=1e-9, max=6400)

    t_vec_next = t_vec + 0.5 * (t_vec_target - t_vec)

    g_next, _ = g_fn.g_and_grad(t_vec_next)
    h_next, _ = h_fn.g_and_grad(t_vec_next)

    t_rand_mid = t_vec + torch.rand(t_vec.shape,device=device) * (t_vec_target - t_vec) 
    g_rand_mid, dotg_rand_mid = g_fn.g_and_grad(t_rand_mid)
    h_rand_mid, doth_rand_mid = h_fn.g_and_grad(t_rand_mid)
    
    # compute update direction
    flow_dir = mat_mul((g, h), edmnet.V, M_times_score_detached, power = -0.5)
    x_next = x_noisy + mat_mul((g_next.sqrt() - g.sqrt(), h_next.sqrt() - h.sqrt()), 
                      edmnet.V, -flow_dir, power = 1.0)
    M_times_score_next_detached = functional_call(edmnet, detached, (x_next, (g_next, h_next)),
                         {"class_labels": labels}) - x_next
    flow_dir_next = mat_mul((g_next, h_next), edmnet.V, M_times_score_next_detached, power = -0.5)

    dt_flow_dir = mat_mul((1/(g_next.sqrt()-g.sqrt()),1/(h_next.sqrt()-h.sqrt())), edmnet.V, 0.5 * (flow_dir_next - flow_dir)) 
    displacement_at_t_mid = (mat_mul((g_rand_mid.sqrt()-g.sqrt(),h_rand_mid.sqrt()-h.sqrt()), edmnet.V, - flow_dir) 
                     + mat_mul(((g_rand_mid.sqrt()-g.sqrt())**2,(h_rand_mid.sqrt()-h.sqrt())**2), edmnet.V, - 0.5 * dt_flow_dir))
    x_rand_mid = x_noisy + displacement_at_t_mid
    
    velocity_at_t_mid = mat_mul((dotg_rand_mid/(g_rand_mid**0.5),doth_rand_mid/(h_rand_mid**0.5)), edmnet.V, 
                     - flow_dir + mat_mul(((g_rand_mid.sqrt()-g.sqrt()),(h_rand_mid.sqrt()-h.sqrt())), edmnet.V, - dt_flow_dir))

    M_times_score_rand_mid = functional_call(edmnet, detached, (x_rand_mid, (g_rand_mid, h_rand_mid)),
                         {"class_labels": labels}) - x_rand_mid
    
    true_velocity_at_t_mid = mat_mul((dotg_rand_mid/g_rand_mid,doth_rand_mid/h_rand_mid), edmnet.V, - M_times_score_rand_mid) 

    disc_err = mat_mul((1/(g_rand_mid+0.5)**0.5,1/(h_rand_mid+0.5)**0.5), edmnet.V, velocity_at_t_mid - true_velocity_at_t_mid)
    
    loss = loss + disc_err.norm(p=2,dim=[1,2,3]).pow(2).mean()
    return loss

# weird implementation quirk, flow_at_x must be computed at x
def create_flow_derivative_gh_term(flownet, x, gh_t, dot_gh_t, labels, flow_at_x, del_flow_loss):
    dct_V = flownet.V
    B,d1,d2,d3 = x.shape
    d = d1*d2*d3
    g_t, h_t = gh_t
    dotg_t, doth_t = dot_gh_t
    assert g_t.shape[0]==B and h_t.shape[0]==B and dotg_t.shape[0]==B and doth_t.shape[0]==B

    # Sample random perturbation directions ei_s for each x:
    ei = torch.randn_like(x)
    ei = ei/ei.norm(p=2,dim=(1,2,3))[:,None,None,None]

    # compute the detached flow, if not provided
    flow_detached = flow_at_x.detach()
    del_flow_loss = del_flow_loss.detach()

    # compute del_x <del_flow_loss.detach(), flow(x, gh, label)
    F_b = (del_flow_loss * flow_at_x).sum()
    (del_x_F_b,) = torch.autograd.grad(F_b, x, create_graph=True, retain_graph=True)
    
    ################################################################
    # 1. form derivative term involving drds flow(x + rei + sdel_theta M ei)
    ################################################################
    dot_del_x_F_b_ei = (del_x_F_b * ei).sum()
    (del_x_dot_del_x_F_b_ei,) = torch.autograd.grad(dot_del_x_F_b_ei, x, create_graph=False, retain_graph=True)
    dot_del_x_dot_del_x_F_b_ei_Mei = (del_x_dot_del_x_F_b_ei.detach() * mat_mul((g_t, h_t), dct_V, ei, power=1.0, identity_scaling=0.0)).sum()

    ################################################################
    # 2. form derivative term involving ds flow(x + sdel_theta M score)
    ################################################################
    dot_del_x_F_b_Mscore = (del_x_F_b.detach() * mat_mul((g_t*g_t.detach()**0.5, h_t*h_t.detach()**0.5), dct_V, flow_detached, power=1.0, identity_scaling=0.0)).sum()
    
    ################################################################
    # 3. form derivative term involving dtheta M
    ################################################################
    F_c = (del_flow_loss * mat_mul((g_t/g_t.detach(), h_t/h_t.detach()), dct_V, flow_detached, power=1.0, identity_scaling=0.0)).sum()

    ########################################
    # 4. Combining everything
    ########################################
    additional_loss = 0.5 * d * dot_del_x_dot_del_x_F_b_ei_Mei + dot_del_x_F_b_Mscore + 0.5 * F_c

    return additional_loss-additional_loss.detach()


def stiefel_project_polar(V_raw: torch.Tensor) -> torch.Tensor:
    """
    SVD-based polar orthogonalization (closest orthonormal in Frobenius norm).
    V_raw: [K, k, H, W]  (columns are basis vectors flattened to length d=H*W)
    Returns: [K, k, H, W] with orthonormal columns.
    """
    V_raw = V_raw[None,:,:,:]

    K, k, H, W = V_raw.shape
    d = H * W
    # Arrange as X \in R^{K x d x k}
    X = V_raw.reshape(K, k, d).transpose(1, 2).contiguous()  # [K, d, k]

    # (Optional but robust) compute SVD in float64, then cast back
    orig_dtype = X.dtype
    X64 = X.to(torch.float64)

    # Batched thin SVD: X = U Σ Vh, with U: [K,d,k], Vh: [K,k,k]
    U, S, Vh = torch.linalg.svd(X64, full_matrices=False)

    # Polar factor: Q = U V^T
    Q64 = U @ Vh
    Q = Q64.to(orig_dtype)

    # Back to [K, k, H, W]
    V_orth = Q.transpose(1, 2).reshape(K, k, H, W)
    V_orth = V_orth[0,:,:,:]
    return V_orth

class GFnLinear(nn.Module):
    """Monotone piecewise-linear time-warp r: [0,T] -> [0,T].
       Same call pattern: forward(t) -> (r(t), dr/dt)."""
    def __init__(self, times: torch.Tensor, T: float):
        super().__init__()
        assert torch.isclose(times[0],  torch.tensor(0., dtype=times.dtype, device=times.device))
        assert torch.isclose(times[-1], torch.tensor(T,  dtype=times.dtype, device=times.device))
        self.register_buffer("times", times.clone())
        self.T = torch.tensor(T, dtype=times.dtype, device=times.device)
        self.K = times.numel()
        # Softmax -> positive inc that sum exactly to T (stable & decoupled scale)
        self.theta = nn.Parameter(torch.zeros(self.K - 1, device=times.device))
        self.register_buffer("eps", torch.tensor(1e-8, dtype=times.dtype, device=times.device))

    def _tau_knots(self):
        w = F.softmax(self.theta, dim=0)        # sum(w)=1
        inc = w * self.T                        # sum(inc) = T
        tau = torch.cumsum(torch.cat([torch.zeros(1, device=self.times.device, dtype=self.times.dtype), inc]), dim=0)
        return tau                               # length K, tau[0]=0, tau[-1]=T

    def forward(self, t: torch.Tensor):
        tauk = self._tau_knots()
        flat = t.flatten()
        idx = torch.searchsorted(self.times, flat, right=False).clamp(1, self.K - 1)
        t0, t1 = self.times[idx - 1], self.times[idx]
        y0, y1 = tauk[idx - 1], tauk[idx]
        denom = (t1 - t0) + self.eps
        w = (flat - t0) / denom
        tau = y0 + w * (y1 - y0)
        dtau_dt = (y1 - y0) / denom
        return tau.view_as(t), dtau_dt.view_as(t)


class GFnComposite(nn.Module):
    def __init__(self, g_fn, r_fn):
        super().__init__()
        self.g_fn = g_fn
        self.r_fn = r_fn
    def forward(self, t: torch.Tensor):
        r, dot_r = self.r_fn(t)
        g, dot_g = self.g_fn(r)
        return g, dot_g * dot_r

    def g_and_grad(self, t):
        return self.forward(t)

class LogGSpaceWarp(nn.Module):
    """
    Warp a fixed schedule g_p(t) in log-g space:
        s = psi(t) = (log g_p(t) - log g0) / (log T - log g0) ∈ [0,1]
        q = w(s)   monotone PL map on [0,1]
        log g_w(t) = log g0 + (log T - log g0) * q
        g_w(t) = exp(log g_w(t))
    Returns (g_w(t), d g_w / dt).
    """
    def __init__(self, g_p_module, K: int = 64):
        super().__init__()
        self.gp = g_p_module                    # fixed purple schedule (can also be trainable)
        dtype, device = self.gp.times.dtype, self.gp.times.device
        self.register_buffer("s_knots", torch.linspace(0., 1., K, dtype=dtype, device=device))
        self.theta = nn.Parameter(torch.zeros(K - 1, dtype=dtype, device=device))  # softmax increments
        self.register_buffer("eps", torch.tensor(1e-18, dtype=dtype, device=device))

        # cache scalars
        g0_t  = torch.tensor(float(self.gp.g0), dtype=dtype, device=device)
        self.register_buffer("log_g0",  torch.log(g0_t.clamp_min(1e-38)))
        self.register_buffer("log_T",   torch.log(self.gp.T))
        self.register_buffer("log_gap", self.log_T - self.log_g0)

    def _w_knots(self):
        w = F.softmax(self.theta, dim=0)            # >=0, sum=1
        y = torch.cumsum(torch.cat([torch.zeros(1, device=w.device, dtype=w.dtype), w], dim=0), dim=0)
        return y                                     # length K, strictly increasing, y[0]=0, y[-1]=1

    @staticmethod
    def _interp_pl(x, xk, yk, eps):
        flat = x.flatten()
        idx  = torch.searchsorted(xk, flat, right=False).clamp(1, len(xk) - 1)
        x0, x1 = xk[idx - 1], xk[idx]
        y0, y1 = yk[idx - 1], yk[idx]
        denom  = (x1 - x0) + eps
        w      = (flat - x0) / denom
        y      = y0 + w * (y1 - y0)
        dydx   = (y1 - y0) / denom
        return y.view_as(x), dydx.view_as(x)

    def forward(self, t):
        g_p, gdot_p = self.gp(t)                                     # g_p(t), g'_p(t)
        log_g_p = torch.log(g_p.clamp_min(1e-38))
        s = (log_g_p - self.log_g0) / (self.log_gap + self.eps)      # ψ(t) ∈ [0,1]

        q_knots = self._w_knots()
        q, dqds = self._interp_pl(s, self.s_knots, q_knots, self.eps)

        log_g_w = self.log_g0 + self.log_gap * q
        g_w = torch.exp(log_g_w)

        # d/dt log g_w = (dq/ds) * d/dt log g_p = (dq/ds) * (g'_p / g_p)
        dloggdt = dqds * (gdot_p / g_p.clamp_min(1e-38))
        g_w_dot = g_w * dloggdt
        return g_w, g_w_dot
    
    def g_and_grad(self, t):
        return self.forward(t)

    # Optional: explicit r(t) and dr/dt
    def r_and_grad(self, t):
        g_p, gdot_p = self.gp(t)
        log_g_p = torch.log(g_p.clamp_min(1e-38))
        s = (log_g_p - self.log_g0) / (self.log_gap + self.eps)
        q_knots = self._w_knots()
        q, dqds = self._interp_pl(s, self.s_knots, q_knots, self.eps)

        # target log g after warp
        log_g_target = self.log_g0 + self.log_gap * q

        # invert g_p in log space (segmentwise)
        logk = self.gp._log_knots()                                   # (K,)
        tk   = self.gp.times
        flat = log_g_target.flatten()
        idx  = torch.searchsorted(logk, flat, right=False).clamp(1, len(logk) - 1)
        l0, l1 = logk[idx - 1], logk[idx]
        t0, t1 = tk[idx - 1], tk[idx]
        alpha  = (l1 - l0) / ((t1 - t0) + self.eps)                   # d log g / dt on the segment
        r = t0 + (flat - l0) / (alpha + self.eps)
        r = r.view_as(t)

        # dr/dt = q'(s) * (dψ/dt at t) / (dψ/dt at r)
        dpsidt_t = (gdot_p / g_p.clamp_min(1e-38)) / (self.log_gap + self.eps)
        _, gdot_pr = self.gp(r)
        g_pr = self.gp(r)[0]
        dpsidt_r = (gdot_pr / g_pr.clamp_min(1e-38)) / (self.log_gap + self.eps)
        drdt = dqds * dpsidt_t / (dpsidt_r + self.eps)
        return r, drdt
    

# ----- implicit inverse: s = c_noise^{-1}(c_target) with correct backward -----
class _CNoiseInverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c_target, g_fn, h_fn, bisect_iters: int, eps: float):
        # Vectorized bisection on [0, T]
        device, dtype = c_target.device, c_target.dtype
        T = float((g_fn.T if not isinstance(g_fn.T, torch.Tensor) else g_fn.T.detach()).item())
        zero = torch.zeros((), device=device, dtype=dtype)
        Tt   = torch.tensor(T, device=device, dtype=dtype)

        @torch.no_grad()
        def c_val(s):
            g = g_fn(s)[0]; h = h_fn(s)[0]
            return (torch.log(g.clamp_min(eps)) + torch.log(h.clamp_min(eps))) / 8.0

        with torch.no_grad():
            c0 = c_val(zero); cT = c_val(Tt)
            inc = (cT > c0).item()
            lo  = torch.zeros_like(c_target)           # 0
            hi  = torch.full_like(c_target, T)         # T
            ct  = c_target.clamp(min=float(min(c0, cT).item()), max=float(max(c0, cT).item()))
            for _ in range(int(bisect_iters)):
                mid = 0.5 * (lo + hi)
                cmid = c_val(mid)
                go_right = (cmid < ct) if inc else (cmid > ct)
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi,  mid)
            s = 0.5 * (lo + hi)

        # Save s for backward
        ctx.g_fn = g_fn; ctx.h_fn = h_fn
        ctx.eps  = eps
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, grad_s):
        (s,) = ctx.saved_tensors
        g, g1 = ctx.g_fn(s); h, h1 = ctx.h_fn(s)
        eps = ctx.eps
        c1 = (g1 / g.clamp_min(eps) + h1 / h.clamp_min(eps)) / 8.0
        grad_c = grad_s / c1.clamp_min(eps)   # ∂L/∂c_target = (∂L/∂s) / c'(s)
        # Only c_target is a Tensor input; g_fn/h_fn/bisect_iters/eps are Python objects
        return grad_c, None, None, None, None
    
class CNoiseProjectorBisect(nn.Module):
    def __init__(self, g_fn, h_fn, g_wrapper, bisect_iters: int = 32, eps: float = 1e-18):
        super().__init__()
        self.g_fn = g_fn
        self.h_fn = h_fn
        self.gw   = g_wrapper
        self.bisect_iters = int(bisect_iters)
        self.eps = float(eps)

        # Sanity check on horizons
        Tg = float(self._to_float_T(g_fn.T))
        Th = float(self._to_float_T(h_fn.T))
        assert abs(Tg - Th) < 1e-6, "g_fn.T and h_fn.T must be equal"
        self.T = Tg
    def wrapper_parameters(self):
        return self.gw.parameters()
    @staticmethod
    def _to_float_T(T):
        if isinstance(T, torch.Tensor):
            return T.reshape(-1)[0].item()
        return float(T)
    # Only the value (for bisection). Kept separate to save work.
    def _c_noise_value(self, s):
        g = self.g_fn(s)[0]
        h = self.h_fn(s)[0]
        return (torch.log(g.clamp_min(self.eps)) + torch.log(h.clamp_min(self.eps))) / 8.0
    def _invert_c_noise_bisect(self, c_target, dtype, device):
        T = torch.tensor(self.T, dtype=dtype, device=device)
        zero = torch.zeros((), dtype=dtype, device=device)

        with torch.no_grad():
            c0 = self._c_noise_value(zero)  # scalar
            cT = self._c_noise_value(T)     # scalar
            increasing = (cT > c0).item()

            # clamp targets into reachable range
            cmin, cmax = (c0, cT) if increasing else (cT, c0)
            ct = c_target.clamp(min=float(cmin.item()), max=float(cmax.item()))

            lo = torch.zeros_like(ct)           # 0
            hi = torch.full_like(ct, float(T))  # T

            for _ in range(self.bisect_iters):
                mid = (lo + hi) * 0.5
                cmid = self._c_noise_value(mid)
                if increasing:
                    go_right = cmid < ct
                else:
                    go_right = cmid > ct
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi,  mid)

            s = 0.5 * (lo + hi)
        return s  # detached; we’ll use closed-form ds/dt

    def _c_prime(self, s):
        g, g1 = self.g_fn(s); h, h1 = self.h_fn(s)
        eps = self.eps
        return (g1 / g.clamp_min(eps) + h1 / h.clamp_min(eps)) / 8.0, g, g1, h, h1

    def forward(self, t):
        gw, gw_dot = self.gw(t)                                    # g_wrap(t), d/dt g_wrap(t)
        c_target   = 0.25 * torch.log(gw.clamp_min(self.eps))      # u(t) = 1/4 log gw
        # s depends on c_target with correct backward (implicit gradient)
        s = _CNoiseInverseFn.apply(c_target, self.g_fn, self.h_fn, self.bisect_iters, self.eps)

        # values and derivatives at s(t)
        c1, g_val, g1_s, h_val, h1_s = self._c_prime(s)
        dsdt = 0.25 * (gw_dot / gw.clamp_min(self.eps)) / c1.clamp_min(self.eps)
        dgdt = g1_s * dsdt
        dhdt = h1_s * dsdt
        return g_val, dgdt, h_val, dhdt, s


class c_wrapper_g(nn.Module):
    def __init__(self, cproj):
        super().__init__()
        self.cproj = cproj

    def forward(self, t):
        g_val, dgdt, h_val, dhdt, s = self.cproj(t)
        return g_val, dgdt

    def g_and_grad(self, t):
        return self.forward(t)

class c_wrapper_h(nn.Module):
    def __init__(self, cproj):
        super().__init__()
        self.cproj = cproj

    def forward(self, t):
        g_val, dgdt, h_val, dhdt, s = self.cproj(t)
        return h_val, dhdt
    
    def g_and_grad(self, t):
        return self.forward(t)