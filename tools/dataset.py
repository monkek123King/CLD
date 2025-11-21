import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as T
from collections import defaultdict

def collate_fn(batch):
    pixels_RGBA = [torch.stack(item["pixel_RGBA"]) for item in batch]  # [L, C, H, W]
    pixels_RGB  = [torch.stack(item["pixel_RGB"])  for item in batch]  # [L, C, H, W]
    pixels_RGBA = torch.stack(pixels_RGBA)  # [B, L, C, H, W]
    pixels_RGB  = torch.stack(pixels_RGB)   # [B, L, C, H, W]

    return {
        "pixel_RGBA": pixels_RGBA,
        "pixel_RGB": pixels_RGB,
        "whole_img": [item["whole_img"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "height": [item["height"] for item in batch],
        "width": [item["width"] for item in batch],
        "layout": [item["layout"] for item in batch],
    }

class LayoutTrainDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        full_dataset = load_dataset(
            "artplus/PrismLayersPro",
            cache_dir=data_dir,
        )
        full_dataset = concatenate_datasets(list(full_dataset.values()))

        if "style_category" not in full_dataset.column_names:
            raise ValueError("Dataset must contain a 'style_category' field to split by class.")

        categories = np.array(full_dataset["style_category"])
        category_to_indices = defaultdict(list)
        for i, cat in enumerate(categories):
            category_to_indices[cat].append(i)

        subsets = []
        for cat, indices in category_to_indices.items():
            total_len = len(indices)
            idx_90 = int(total_len * 0.9)
            idx_95 = int(total_len * 0.95)

            if split == "train":
                selected_idx = indices[:idx_90]
            elif split == "test":
                selected_idx = indices[idx_90:idx_95]
            elif split == "val":
                selected_idx = indices[idx_95:]
            else:
                raise ValueError("split must be 'train', 'val', or 'test'")

            subsets.append(full_dataset.select(selected_idx))

        self.dataset = concatenate_datasets(subsets)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        def rgba2rgb(img_RGBA):
            img_RGB = Image.new("RGB", img_RGBA.size, (128, 128, 128))
            img_RGB.paste(img_RGBA, mask=img_RGBA.split()[3])
            return img_RGB

        def get_img(x):
            if isinstance(x, str):
                img_RGBA = Image.open(x).convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
            else:
                img_RGBA = x.convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
            return img_RGBA, img_RGB

        whole_img_RGBA, whole_img_RGB = get_img(item["whole_image"])
        whole_cap = item["whole_caption"]
        W, H = whole_img_RGBA.size
        base_layout = [0, 0, W - 1, H - 1]

        layer_image_RGBA = [self.to_tensor(whole_img_RGBA)]
        layer_image_RGB  = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]

        base_img_RGBA, base_img_RGB = get_img(item["base_image"])
        layer_image_RGBA.append(self.to_tensor(base_img_RGBA))
        layer_image_RGB.append(self.to_tensor(base_img_RGB))
        layout.append(base_layout)

        layer_count = item["layer_count"]
        for i in range(layer_count):
            key = f"layer_{i:02d}"
            img_RGBA, img_RGB = get_img(item[key])
            
            w0, h0, w1, h1 = item[f"{key}_box"]

            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))

            W_img, H_img = w1 - w0, h1 - h0
            if img_RGBA.size != (W_img, H_img):
                img_RGBA = img_RGBA.resize((W_img, H_img), Image.BILINEAR)
                img_RGB  = img_RGB.resize((W_img, H_img), Image.BILINEAR)

            canvas_RGBA.paste(img_RGBA, (w0, h0), img_RGBA)
            canvas_RGB.paste(img_RGB, (w0, h0))

            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([w0, h0, w1, h1])

        return {
            "pixel_RGBA": layer_image_RGBA,
            "pixel_RGB": layer_image_RGB,
            "whole_img": whole_img_RGB,
            "caption": whole_cap,
            "height": H,
            "width": W,
            "layout": layout,
        }