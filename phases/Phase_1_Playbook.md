# 🌱 Phase 1 – Define Scope & Gather Data (Weeks 1–2)

This document provides a **step-by-step technical guide** to successfully complete Phase 1 of the Plant Health Evaluator project.  
It covers dataset definition, collection, annotation, cleaning, and preparation for model training.

---

## 0) One-Time Project Setup (Repository + Environment)

1. **Create folders**
   ```
   plant-health-evaluator/
   ├─ planning/
   ├─ data/
   │  ├─ raw/
   │  │  ├─ plantvillage/
   │  │  └─ scraped/
   │  ├─ interim/
   │  ├─ annotations/
   │  ├─ manifests/
   │  └─ splits/
   └─ scripts/
   ```

2. **Set up a Python virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install --upgrade pip
   pip install pillow opencv-python pandas tqdm imagehash scikit-image matplotlib jupyter
   ```

3. **Optional tools**
   ```bash
   pip install roboflow ultralytics
   pip install labelImg
   ```

---

## 1) Finalize Scope & Success Criteria

1. **Indicators for v1**
   - Color: `green_healthy`, `yellow_chlorosis`, `brown_necrosis`
   - Fungi Presence: `fungi_present` ∈ {0, 1}
   - Wilting Presence: `wilting_present` ∈ {0, 1}

2. **Skip Size Regression for v1**
   - No reliable scale in public datasets.
   - Add later when capturing your own data.

3. **Target Dataset Sizes**
   - Color: ≥300 green, ≥150 yellow, ≥100 brown
   - Fungi: ≥250 positives, ≥400 negatives
   - Wilting: ≥200 positives, ≥400 negatives

4. **Label Formats**
   - Color → image-level labels
   - Fungi/Wilting → bounding boxes (YOLO) → later converted to binary image-level labels

---

## 2) Acquire Base Dataset (PlantVillage)

1. **Install Kaggle CLI**
   ```bash
   pip install kaggle
   ```

2. **Set API key**
   - Download `kaggle.json` from your Kaggle account
   - Place it in: `C:\Users\<you>\.kaggle\kaggle.json`

3. **Download Tomato dataset**
   ```bash
   mkdir -p data/raw/plantvillage
   kaggle datasets download -d noulam/tomato -p data/raw/plantvillage
   cd data/raw/plantvillage
   tar -xf tomato.zip 2>nul || unzip -q tomato.zip
   cd ../../../
   ```

---

## 3) Supplement Dataset with Web-Scraped Images

### Option A — Fatkun Chrome Extension
1. Install **Fatkun Batch Image Downloader**
2. Search Google Images for:
   - “tomato plant yellow leaves”
   - “tomato plant wilting”
   - “tomato leaf blight”
3. Use Fatkun to bulk-download ≥400px images.
4. Save to `data/raw/scraped/<keyword>/`

### Option B — Python Scraping (optional)
- Record sources and queries in a CSV manifest:
  ```
  path,source,query,width,height,downloaded_at
  data/raw/scraped/wilting/img_0001.jpg,google_images,"tomato wilting",1024,768,2025-10-05
  ```

---

## 4) Curate, Clean & Deduplicate Images

1. **Rename images**
   - Use consistent naming like `pv_###.jpg` or `scr_keyword_###.jpg`.

2. **Filter small/corrupt images**
   - Remove images <256×256 or unreadable.

3. **Run duplicate detection script**
   ```python
   import os, pandas as pd, imagehash
   from PIL import Image
   from tqdm import tqdm
   from pathlib import Path

   ROOTS = ["data/raw/plantvillage", "data/raw/scraped"]
   rows = []
   for root in ROOTS:
       for p in Path(root).rglob("*.*"):
           try:
               with Image.open(p) as im:
                   im = im.convert("RGB")
                   h = imagehash.average_hash(im)
                   rows.append({"path": str(p), "phash": str(h)})
           except Exception:
               pass

   df = pd.DataFrame(rows)
   df.to_csv("data/manifests/_hashes.csv", index=False)
   print("Duplicate hash list saved!")
   ```

4. **Manually review duplicates**
   - Remove low-quality versions.

---

## 5) Create Color Labels

1. **Create `data/manifests/color_labels.csv`**
   ```
   path,color_label
   data/raw/.../img1.jpg,green_healthy
   data/raw/.../img2.jpg,yellow_chlorosis
   data/raw/.../img3.jpg,brown_necrosis
   ```

2. **Labeling Guide**
   - Green = mostly green foliage, no visible damage
   - Yellow = interveinal yellowing
   - Brown = widespread necrosis or dead tissue

---

## 6) Annotate Fungi & Wilting

### Option A — LabelImg (local)
```bash
labelImg
```
- Classes file:
  ```
  fungi
  wilting
  ```
- Save to `data/annotations/labelimg/` in YOLO format.

### Option B — Roboflow
- Create detection project with classes `fungi` and `wilting`.
- Label online, export YOLO format to `data/annotations/roboflow_yolo/`.

---

## 7) Convert Boxes → Image-Level Presence Labels

```python
import os, pandas as pd
from pathlib import Path

IMG_ROOT = Path("data/raw")
ANN_ROOTS = [Path("data/annotations/labelimg"), Path("data/annotations/roboflow_yolo")]
rows = []

for img in IMG_ROOT.rglob("*.jpg"):
    fungi = 0; wilting = 0
    for ann_root in ANN_ROOTS:
        txt = ann_root / img.relative_to(IMG_ROOT).with_suffix(".txt")
        if txt.exists():
            for line in open(txt):
                cls = line.strip().split()[0]
                if cls in ["0","fungi"]: fungi = 1
                if cls in ["1","wilting"]: wilting = 1
    rows.append({"path": str(img), "fungi_present": fungi, "wilting_present": wilting})

pd.DataFrame(rows).to_csv("data/manifests/presence_labels.csv", index=False)
print("Presence labels saved!")
```

---

## 8) Merge Manifests & Create Train/Val/Test Splits

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

imgs = pd.read_csv("data/manifests/images.csv")
color = pd.read_csv("data/manifests/color_labels.csv")
pres = pd.read_csv("data/manifests/presence_labels.csv")

df = imgs.merge(color, on="path", how="left").merge(pres, on="path", how="left")
df["fungi_present"] = df["fungi_present"].fillna(0).astype(int)
df["wilting_present"] = df["wilting_present"].fillna(0).astype(int)
df = df.dropna(subset=["color_label"])

train, temp = train_test_split(df, test_size=0.3, stratify=df["color_label"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["color_label"], random_state=42)

Path("data/splits").mkdir(parents=True, exist_ok=True)
train.to_csv("data/splits/train.csv", index=False)
val.to_csv("data/splits/val.csv", index=False)
test.to_csv("data/splits/test.csv", index=False)
```

---

## 9) Dataset Balance Check

```python
import pandas as pd
for split in ["train","val","test"]:
    df = pd.read_csv(f"data/splits/{split}.csv")
    print(f"\n== {split.upper()} ==")
    print("Color:", df["color_label"].value_counts().to_dict())
    print("Fungi present:", df["fungi_present"].sum(), "/", len(df))
    print("Wilting present:", df["wilting_present"].sum(), "/", len(df))
```

---

## 10) Document Dataset Card

Create `planning/dataset_card.md` and include:
- Sources (PlantVillage + scraped)
- Label schema
- Counts per split
- Limitations (lighting, no scale)
- Licensing (cite PlantVillage)

---

## 11) Phase 1 QA Checklist

- [ ] Folder structure and venv set up
- [ ] PlantVillage dataset downloaded
- [ ] Scraped images collected
- [ ] Duplicates removed
- [ ] `color_labels.csv` complete
- [ ] Fungi/Wilting annotations complete
- [ ] `presence_labels.csv` generated
- [ ] Train/Val/Test splits created
- [ ] Dataset card documented

---

## Tips
- Consistency > volume: label fewer but cleaner images.
- Keep a fixed “sanity” subset aside for visual verification.
- Aim to finish labeling before Week 2 ends.
