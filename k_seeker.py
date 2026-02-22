import torch
import torch.nn as nn
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import os

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 500_000  # set to integer to limit search
SAE_CHECKPOINT = "results/checkpoints/vague-sweep-90_CLIP_512_4096_256/vague-sweep-90_CLIP_512_4096_256.pt"

SAVE_DIR = "./extreme_k_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Load your model
# -----------------------------
from soft_sae.soft_top_k import SoftTopKSAE

sae = SoftTopKSAE.from_pretrained(SAE_CHECKPOINT, device=DEVICE)
sae.eval()


import clip

clip_model, clip_preprocess = clip.load("ViT-B/16")

clip_model = clip_model.to(DEVICE)
clip_model.eval()


# -----------------------------
# Tracking min / max k
# -----------------------------
min_k = float("inf")
max_k = -float("inf")

min_image = None
max_image = None

min_url = None
max_url = None


def fetch_image(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception:
        return None


from torch.utils.data import IterableDataset, DataLoader


# -----------------------------
# Streaming Dataset Wrapper
# -----------------------------
class CC3MStreamDataset(IterableDataset):
    def __init__(self, hf_dataset, preprocess, max_samples=None):
        self.dataset = hf_dataset
        self.preprocess = preprocess
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for sample in self.dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break

            url = sample["image_url"]
            img = fetch_image(url)
            if img is None:
                continue

            try:
                tensor_img = self.preprocess(img)
            except Exception:
                continue

            yield tensor_img, img, url
            count += 1


# -----------------------------
# Load CC3M in streaming mode
# -----------------------------
hf_dataset = load_dataset(
    "conceptual_captions",
    split="train",
    streaming=True,
)

stream_dataset = CC3MStreamDataset(
    hf_dataset,
    preprocess=clip_preprocess,
    max_samples=NUM_SAMPLES,
)

BATCH_SIZE = 32


def collate_fn(batch):
    tensors, pil_imgs, urls = zip(*batch)

    tensors = torch.stack(tensors, dim=0)
    pil_imgs = list(pil_imgs)  # keep as list
    urls = list(urls)

    return tensors, pil_imgs, urls


dataloader = DataLoader(
    stream_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    collate_fn=collate_fn,
    prefetch_factor=8
)

count = 0


for batch in tqdm(dataloader):

    batch_tensor_imgs, batch_pil_imgs, batch_urls = batch

    # Move tensor batch to device
    batch_tensor = batch_tensor_imgs.to(DEVICE)

    with torch.no_grad():
        feats = clip_model.encode_image(batch_tensor)
        k_est = sae.estimate_k(sae.normalize(feats))

    for i in range(len(k_est)):
        k_val = k_est[i].item()

        if k_val < min_k:
            min_k = k_val
            min_image = batch_pil_imgs[i]
            min_url = batch_urls[i]

        if k_val > max_k:
            max_k = k_val
            max_image = batch_pil_imgs[i]
            max_url = batch_urls[i]


# -----------------------------
# Save results
# -----------------------------
if min_image is not None:
    min_path = os.path.join(SAVE_DIR, "min_k.jpg")
    min_image.save(min_path)
    print(f"Saved MIN k image → {min_path}")
    print(f"Min k value: {min_k}")
    print(f"URL: {min_url}")

if max_image is not None:
    max_path = os.path.join(SAVE_DIR, "max_k.jpg")
    max_image.save(max_path)
    print(f"Saved MAX k image → {max_path}")
    print(f"Max k value: {max_k}")
    print(f"URL: {max_url}")

print("\nDone.")
print(f"Final Min k: {min_k}")
print(f"Final Max k: {max_k}")
