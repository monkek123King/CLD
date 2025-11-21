import os
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import json

from tools.tools import get_input_box
from tools.dataset import LayoutTrainDataset, collate_fn

data_root = "Path_to_dataset"
save_root = "Path_to_save_ground_truth"

dataset = LayoutTrainDataset(data_root, split="test")
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

os.makedirs(save_root, exist_ok=True)

for i, batch in enumerate(loader):
    print(f"Processing case {i}...")

    case_dir = os.path.join(save_root, f"case_{i}")
    os.makedirs(case_dir, exist_ok=True)

    layer_boxes = get_input_box(batch["layout"][0]) 
    layer_boxes_path = os.path.join(case_dir, "layer_boxes.json")
    with open(layer_boxes_path, "w") as f:
        json.dump(layer_boxes, f, indent=4)


    pixel_RGBA = batch["pixel_RGBA"][0]  # [L, C, H, W]
    composite_img = pixel_RGBA[0]
    pixel_RGBA = pixel_RGBA[1:]
    L = pixel_RGBA.shape[0]

    composite_img_path = os.path.join(case_dir, "composite.png")
    TF.to_pil_image(composite_img).save(composite_img_path)

    for j in range(L):
        img = pixel_RGBA[j]  # [C, H, W]
        img_path = os.path.join(case_dir, f"{j:02d}.png")
        TF.to_pil_image(img).save(img_path)
