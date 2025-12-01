def train_epoch(epoch):
    G.train()
    D.train()

    epoch_g_loss, epoch_d_loss = 0.0, 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for i, (pre_enhanced, original_low, _) in enumerate(pbar):
        pre_enhanced = pre_enhanced.to(DEVICE, non_blocking=True)
        original_low = original_low.to(DEVICE, non_blocking=True)

        # Train Discriminator
        optimizer_D.zero_grad(set_to_none=True)

        pred_real = D(pre_enhanced)
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real) * 0.9)

        with torch.no_grad():
            fake_imgs = G(pre_enhanced)

        pred_fake = D(fake_imgs.detach())
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake) + 0.1)

        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad(set_to_none=True)

        refined_imgs = G(pre_enhanced)
        pred_fake = D(refined_imgs)

        # Core losses only (fast)
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) * LAMBDA_ADV
        loss_L1 = criterion_L1(refined_imgs, pre_enhanced) * LAMBDA_L1
        loss_perceptual = lpips_fn(refined_imgs, pre_enhanced).mean() * LAMBDA_PERCEPTUAL

        # Regularization (computed efficiently)
        loss_tv = total_variation_loss(refined_imgs) * LAMBDA_TV
        loss_brightness = brightness_enhancement_loss(refined_imgs, pre_enhanced) * LAMBDA_BRIGHTNESS
        loss_balance = color_balance_loss(refined_imgs) * LAMBDA_BALANCE

        # Total
        loss_G = loss_GAN + loss_L1 + loss_perceptual + loss_tv + loss_brightness + loss_balance

        loss_G.backward()
        optimizer_G.step()

        epoch_g_loss += loss_G.item()
        epoch_d_loss += loss_D.item()

        if i % 20 == 0:
            pbar.set_postfix({'G': f'{loss_G.item():.2f}', 'D': f'{loss_D.item():.2f}'})

    return epoch_g_loss / len(train_loader), epoch_d_loss / len(train_loader)


@torch.no_grad()
def validate():
    G.eval()
    val_loss = 0.0

    for pre_enhanced, _, _ in val_loader:
        pre_enhanced = pre_enhanced.to(DEVICE, non_blocking=True)
        refined = G(pre_enhanced)

        loss = (criterion_L1(refined, pre_enhanced) +
                lpips_fn(refined, pre_enhanced).mean() +
                F.relu(pre_enhanced.mean() - refined.mean()) * 2.0)

        val_loss += loss.item()

    return val_loss / len(val_loader)
