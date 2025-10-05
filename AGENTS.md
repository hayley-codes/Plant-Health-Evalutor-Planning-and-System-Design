# 🌿 Plant Health Evaluator Project Checklist

## Phase 1: Define Scope & Gather Data (Weeks 1–2)
- [ ] Define measurable health indicators (color, size, fungi, wilting, pests)
- [ ] Review and download PlantVillage tomato dataset
- [ ] Write or use an image scraper (e.g., Selenium, BeautifulSoup, Fatkun)
- [ ] Organize images into `train/val/test` folders
- [ ] Use LabelImg or Roboflow to annotate fungi, pest, and wilting regions
- [ ] Verify class balance and adjust dataset size if needed

## Phase 2: Build & Train Model (Weeks 3–5)
- [ ] Implement preprocessing pipeline (resize 224×224, normalize 0–1, augmentations)
- [ ] Load pre-trained CNN (MobileNetV2 or ResNet50)
- [ ] Add custom output heads:
  - [ ] Color classification (Softmax)
  - [ ] Size regression (MSE loss)
  - [ ] Fungi detection (Binary cross-entropy)
  - [ ] Wilting detection (Binary cross-entropy)
- [ ] Define multi-loss function and optimizer
- [ ] Train model; log accuracy and loss per output
- [ ] Validate on test set and save model weights (`.h5` or `.pt`)

## Phase 3: Health Scoring System (Week 6)
- [ ] Define numerical mapping for each indicator (e.g., green = 100, brown = 0)
- [ ] Implement scoring formula:
  \[
  Health = 0.4(Color) + 0.2(Size) + 0.3(Fungi) + 0.1(Wilting)
  \]
- [ ] Write a Python script to take an image → output a total health score
- [ ] Test on multiple example images and verify logic

## Phase 4: Hardware Integration (Weeks 7–8)
- [ ] Set up and test ESP32-CAM (image capture quality and Wi-Fi connection)
- [ ] Write Arduino code to capture and send images to the PC
- [ ] Develop Python receiver script to listen for and save incoming images
- [ ] Automatically trigger inference using the trained model
- [ ] Display the resulting health score (terminal, GUI, or web app)

## Phase 5: Testing, Optimization & Reporting (Weeks 9–10)
- [ ] Test system on multiple plants under varying conditions
- [ ] Adjust scoring weights and thresholds for accuracy
- [ ] Optimize inference speed and image transfer time
- [ ] Create visual graphs comparing predicted vs. actual health
- [ ] Write final report summarizing model performance and hardware setup
- [ ] Record a short demo video showcasing the full system