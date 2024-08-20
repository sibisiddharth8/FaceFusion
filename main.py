import os
import matplotlib.pyplot as plt
import gdown
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from faceswap import swap_n_show, swap_n_show_same_img, swap_face_single, fine_face_swap

# Ensure torch and torchvision are installed
try:
    import torch
    import torchvision
except ImportError:
    raise ImportError("Please install torch and torchvision. You can install them using 'pip install torch torchvision'.")

# Print versions to verify compatibility
print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# Check the alternative import
try:
    from torchvision.transforms.functional import rgb_to_grayscale
except ImportError:
    raise ImportError("Unable to import 'rgb_to_grayscale' from 'torchvision.transforms.functional'. Ensure you have the correct versions of torch and torchvision.")

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Specify the correct path to the model file
model_output_path = 'model/FaceFusion-SoC.onnx'

# Check if the model file exists
if not os.path.exists(model_output_path):
    raise FileNotFoundError(f"Model file {model_output_path} does not exist. Please check the path or download the model.")

swapper = insightface.model_zoo.get_model(model_output_path, download=False, download_zip=False)

# Load images
img1_fn = 'images/aaaa.png'
img2_fn = 'images/bbbb.png'
# Swap faces between two images
# swap_n_show(img1_fn, img2_fn, app, swapper)

# Swap faces within the same image 
# swap_n_show_same_img(img1_fn, app, swapper)

# Add face to an image
swap_face_single(img1_fn, img2_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 2x', device="cpu")

# Fine face swapper
fine_face_swap(img1_fn, img2_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 2x', device="cpu")
