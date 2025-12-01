class LowLightEnhancer:
    """
    Easy-to-use inference class for the trained model.
    """
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedUNetGenerator().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((400, 600)),
            T.ToTensor()
        ])

    def enhance(self, image_path_or_pil):
        """
        Enhance a low-light image.

        Args:
            image_path_or_pil: Path to image file or PIL Image

        Returns:
            PIL Image of enhanced result
        """
        # Load image
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil

        # Apply pre-enhancement
        img_np = np.array(img)
        pre_enhanced_np = apply_simple_retinex(img_np)
        pre_enhanced_pil = Image.fromarray(pre_enhanced_np)

        # Transform
        img_tensor = self.transform(pre_enhanced_pil).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            enhanced_tensor = self.model(img_tensor)

        # Convert back to PIL
        enhanced_np = enhanced_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
        enhanced_pil = Image.fromarray(enhanced_np)

        return enhanced_pil

    def enhance_batch(self, image_paths, output_folder):
        """
        Enhance multiple images and save to folder.
        """
        os.makedirs(output_folder, exist_ok=True)

        for img_path in tqdm(image_paths, desc="Enhancing"):
            enhanced = self.enhance(img_path)
            fname = os.path.basename(img_path)
            output_path = os.path.join(output_folder, fname)
            enhanced.save(output_path)

        print(f"Saved {len(image_paths)} enhanced images to {output_folder}")


# Example usage of the enhancer
enhancer = LowLightEnhancer(
    model_path=os.path.join(SAVE_DIR, "generator_inference.pth"),
    device=DEVICE
)

print("\nLowLightEnhancer ready for inference!")
print("Usage: enhanced_img = enhancer.enhance('path/to/image.jpg')")