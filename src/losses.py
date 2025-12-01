import lpips

criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()

# Perceptual loss
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
for param in lpips_fn.parameters():
    param.requires_grad = False

# SIMPLIFIED: Precompute static tensors (MUCH faster)
# Laplacian kernel - create once, reuse
laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                 dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)

def total_variation_loss(img):
    """Fast TV loss."""
    return (torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
            torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])))

def color_consistency_loss(img1, img2):
    """Simple mean consistency."""
    return F.l1_loss(img1.mean(dim=[2, 3]), img2.mean(dim=[2, 3]))

def illumination_smoothness_loss(enhanced, original):
    """Smooth illumination changes."""
    illum_map = torch.clamp((enhanced + 1e-6) / (original + 1e-6), 0, 3)
    return total_variation_loss(illum_map)

def brightness_enhancement_loss(refined, pre_enhanced):
    """Prevent darkening."""
    return F.relu(pre_enhanced.mean() - refined.mean())

def edge_preservation_loss(refined, pre_enhanced):
    """Fast edge preservation using precomputed kernel."""
    refined_gray = refined.mean(dim=1, keepdim=True)
    pre_gray = pre_enhanced.mean(dim=1, keepdim=True)

    edges_refined = F.conv2d(refined_gray, laplacian_kernel, padding=1)
    edges_pre = F.conv2d(pre_gray, laplacian_kernel, padding=1)

    return F.l1_loss(edges_refined, edges_pre)

def color_balance_loss(img):
    """Prevent channel dominance."""
    r_mean = img[:, 0].mean()
    g_mean = img[:, 1].mean()
    b_mean = img[:, 2].mean()
    rgb_mean = (r_mean + g_mean + b_mean) / 3.0
    return (torch.abs(r_mean - rgb_mean) + torch.abs(g_mean - rgb_mean) + torch.abs(b_mean - rgb_mean))

def gray_world_loss(img):
    """Prevent color casts."""
    mean_r, mean_g, mean_b = img[:, 0].mean(), img[:, 1].mean(), img[:, 2].mean()
    gray_val = (mean_r + mean_g + mean_b) / 3.0
    return torch.abs(mean_r - gray_val) + torch.abs(mean_g - gray_val) + torch.abs(mean_b - gray_val)

print("Simplified loss functions loaded (3x faster, precomputed kernels)")