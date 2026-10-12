import os
import torch
from torchvision.transforms import v2
from data_loader import PolyGen
from torchvision.utils import save_image
from freq_space_interpolation import freq_space_interpolation, extract_amp_spectrum

transform =   v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.ToDtype(torch.float32, scale=True)]) 
dataset1 = PolyGen(root='PolypGen2021_MultiCenterData_v3/', center = 1, transform = transform)
dataset2 = PolyGen(root='PolypGen2021_MultiCenterData_v3/', center = 3, transform = transform)
img1 = dataset1.__getitem__(0)[0]
img2 = dataset2.__getitem__(2)[0]
# img1 = dataset1.__getitem__(0)[0]
# img2 = dataset2.__getitem__(4)[0]
img3 = freq_space_interpolation(img1, extract_amp_spectrum(img2), ratio=0.5)

# save_image(img1, 'source.jpg')
# save_image(img2, 'target.jpg')
# save_image(img3, 'result.jpg')

output_dir = "visualize_output"
os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

save_image(img1, os.path.join(output_dir, 'source_new.jpg'))
save_image(img2, os.path.join(output_dir, 'target_new.jpg'))
save_image(img3, os.path.join(output_dir, 'result_new.jpg'))
