#  DAULLIE – Domain Adaptive Unsupervised Low-Light Image Enhancement  

DAULLIE is a deep learning system that enhances **low-light images** using a **two-stage unsupervised pipeline**.  
It combines lightweight **Retinex preprocessing** with a **U-Net + PatchGAN refinement network** to produce brighter, cleaner, and more detailed images.

---

## 🌟 Key Features

- 💡 **Two-Stage Enhancement**
  - Stage 1: Retinex pre-enhancement (CLAHE + gamma + denoise)  
  - Stage 2: Deep U-Net refinement trained with adversarial learning  

- 🧠 **Fully Unsupervised**
  - No paired bright images required  
  - Works across indoor (LOL) and outdoor low-light domains  

- ⚡ **High Performance**
  - Improved illumination & clarity  
  - Strong texture preservation  
  - Effective for SLAM, tracking, and night-vision tasks  

- 🪶 **Lightweight Model**
  - ~2.3M parameters  
  - Fast inference and deployment  

---

## 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| Framework | PyTorch |
| Preprocessing | OpenCV (Retinex, CLAHE) |
| Losses | GAN, L1, LPIPS, TV, Color/Illumination |
| Training | Google Colab T4 GPU |
| Output Model | `.pth` |

---

## 📁 Project Structure

```
DAULLIE/
│
├── src/
│   ├── pre_enhancement.py
│   ├── dataset.py
│   ├── losses.py
│   ├── train.py
│   ├── inference.py
│   ├── export_model.py
│   └── models/
│       ├── generator.py
│       └── discriminator.py
│
├── models/
│   └── model_info.json
│   └── readme.txt
│
├── results/
│   ├── epoch_comparison/
│   ├── indoor_lol/
│   ├── outdoor_synthetic/
│   └── test_results/
│
└── notebooks/
```

---

## 🚀 Quick Start

### 1️⃣ Install Requirements
```
pip install -r requirements.txt
```

### 2️⃣ Run Inference
```python
from inference import LowLightEnhancer

enhancer = LowLightEnhancer("models/generator_inference.pth")
output = enhancer.enhance("low_light.jpg")
output.save("enhanced.jpg")
```

---

## 📝 Model Summary

- Two-stage unsupervised enhancement  
- Improved U-Net generator + PatchGAN discriminator  
- Balanced loss functions for color, texture, and illumination  
- Best results achieved at **Epoch 55**  

---

## 📷 Example Outputs  
See the `results/` folder for indoor, outdoor, and epoch-wise comparisons.

---

## 🎯 Use Cases
- Night-time photography  
- Robotics & SLAM  
- CCTV enhancement  
- Feature extraction in dark environments  
- General computer vision preprocessing  

---

## 🔧 Future Improvements

- Improve output stability by reducing artifacts such as blotchy textures, halo effects, and color inconsistencies  
- Enhance loss balancing to better preserve structure and illumination in challenging scenes  
- Add a dedicated evaluation script for automatic PSNR, SSIM, and NIQE computation  
- Integrate a lighter LPIPS variant for faster training cycles  
- Introduce temporal consistency for video-based low-light enhancement  
- Replace PatchGAN with a multi-scale or attention-based discriminator for finer detail reconstruction  
- Improve synthetic data generation for stronger outdoor domain generalization  
- Deploy the model using ONNX or TensorRT for real-time inference  
- Expand training to high resolutions (e.g., 4K) using memory-efficient tiling strategies  

---

## ⭐ Support  
If this project helps you, consider giving it a **star ⭐** on GitHub!
