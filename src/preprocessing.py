def apply_light_preenhancement(img_rgb):
    """
    LIGHTER pre-enhancement that leaves room for refinement network.
    Only does basic brightness boost, NOT full enhancement.
    """
    if isinstance(img_rgb, Image.Image):
        img_rgb = np.array(img_rgb)

    # Light denoising only
    img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, h=5, hColor=5,
                                                     templateWindowSize=5, searchWindowSize=11)

    # Convert to LAB
    lab = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # GENTLE CLAHE (not aggressive)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Small brightness boost only
    l_enhanced = np.clip(l_enhanced.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)

    # Merge
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Light gamma only
    rgb_enhanced = np.power(rgb_enhanced / 255.0, 0.9) * 255

    return np.clip(rgb_enhanced, 0, 255).astype(np.uint8)


# Main functions
def apply_improved_retinex(img_rgb):
    """Light pre-enhancement for training."""
    return apply_light_preenhancement(img_rgb)

def apply_simple_retinex(img_rgb):
    """Light pre-enhancement (alias)."""
    return apply_light_preenhancement(img_rgb)

def apply_msrcr(img_rgb, scales=[15, 80, 250]):
    """MSRCR - only for final testing."""
    if isinstance(img_rgb, Image.Image):
        img_rgb = np.array(img_rgb)

    img_rgb = img_rgb.astype(np.float32)
    img_log = np.log1p(img_rgb)

    msr = np.zeros_like(img_rgb)
    for scale in scales:
        blurred = cv2.GaussianBlur(img_rgb, (0, 0), scale)
        blurred_log = np.log1p(blurred)
        msr += (img_log - blurred_log)

    msr = msr / len(scales)
    alpha = 125.0
    img_sum = np.sum(img_rgb, axis=2, keepdims=True)
    color_restoration = np.log1p(alpha * (img_rgb / (img_sum + 1e-6)))
    enhanced = msr * color_restoration
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-6) * 255
    enhanced = np.power(enhanced / 255.0, 0.8) * 255

    return np.clip(enhanced, 0, 255).astype(np.uint8)