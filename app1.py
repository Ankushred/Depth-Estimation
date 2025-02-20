import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

filename = "test.jpg"

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)


with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()
plt.imshow(depth_map)  # Use grayscale colormap
plt.axis("off")  # Hide axes

# Save the figure
plt.savefig("depth img "+filename, dpi=300,transparent=True,bbox_inches="tight", pad_inches=0)
plt.close()
depth_min = depth_map.min()
depth_max = depth_map.max()
depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)  # Normalize to 0-1
depth_gray = (depth_normalized * 255).astype(np.uint8)  # Convert to uint8 for grayscale
cv2.imwrite("Grayscale "+filename, depth_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()