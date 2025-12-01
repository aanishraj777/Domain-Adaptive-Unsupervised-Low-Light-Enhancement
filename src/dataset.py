class PreEnhancedDataset(Dataset):
    """
    Dataset that applies pre-enhancement on-the-fly.
    Returns: (pre_enhanced_image, original_low_image)
    """
    def __init__(self, low_folder, transform=None, use_msrcr=False):
        self.low_folder = low_folder
        self.names = sorted([f for f in os.listdir(low_folder)
                           if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.use_msrcr = use_msrcr

        if transform is None:
            self.transform = T.Compose([
                T.Resize((400, 600)),  # H x W
                T.ToTensor(),          # [0, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        path = os.path.join(self.low_folder, self.names[idx])
        img = Image.open(path).convert("RGB")

        # Apply pre-enhancement (Stage 1)
        img_np = np.array(img)
        if self.use_msrcr:
            enhanced_np = apply_msrcr(img_np)
        else:
            enhanced_np = apply_simple_retinex(img_np)

        enhanced_pil = Image.fromarray(enhanced_np)

        # Transform both
        enhanced = self.transform(enhanced_pil)
        original = self.transform(img)

        return enhanced, original, self.names[idx]