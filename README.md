# Predicting Urban Traffic Patterns — TrafficVission-AI 🚦

TrafficVission-AI is a Python-based project designed to analyze and predict **urban traffic behavior** using computer vision and deep learning techniques. By leveraging pretrained models like YOLO and classical methods like Haar Cascades, the project provides vehicle detection and congestion insights from both images and videos.

---

## 📌 Objectives

- Detect vehicles and traffic patterns using **computer vision**.
- Predict congestion and flow dynamics through **image & video analysis**.
- Provide a lightweight **UI (index.html)** for visualization of predictions.
- Support multiple models (Haar cascades, YOLOv5, YOLOv8).

---

## 📂 Project Structure

```

Predicting-Urban-Traffic-Patterns-TrafficVission-AI-/
│
├── Dataset/                # Contains traffic datasets
├── Models/                 # Model definitions / checkpoints
│
├── 1.py – 6.py             # Experimental or preprocessing scripts
├── TrafficImagePrediction.py   # Run traffic prediction on images
├── TrafficVideoPrediction.py   # Run traffic prediction on videos
├── main.py                 # Central entry point (image/video handling)
│
├── haarcascade\_car.xml      # Haar Cascade model for car detection
├── yolov5s.pt               # Pretrained YOLOv5 small model
├── yolov5su.pt              # Pretrained YOLOv5 variant
├── yolov8n.pt               # Pretrained YOLOv8 nano model
├── idd-lite.zip             # Dataset (IDD-Lite: Indian Driving Dataset)
│
├── index.html               # Simple visualization UI
├── red-traffic-light-flat-illustration.avif  # Asset for UI
│
└── .gitignore

````

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BALIYANSARTHAK/Predicting-Urban-Traffic-Patterns-TrafficVission-AI-.git
   cd Predicting-Urban-Traffic-Patterns-TrafficVission-AI-
````

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > If `requirements.txt` is missing, you can install commonly used packages:

   ```bash
   pip install opencv-python torch torchvision ultralytics numpy matplotlib
   ```

---

## 🚀 Usage

### 🔹 Image Prediction

Run traffic detection on a static image:

```bash
python TrafficImagePrediction.py --input path/to/image.jpg --output results/
```

### 🔹 Video Prediction

Run detection and prediction on a video:

```bash
python TrafficVideoPrediction.py --input path/to/video.mp4 --output output_video.mp4
```

### 🔹 Main Script

The main script can be used to control the mode:

```bash
python main.py --mode image --input path/to/image.jpg
python main.py --mode video --input path/to/video.mp4
```

---

## 📊 Results & Output

* **Image Mode** → Outputs annotated images with bounding boxes & traffic density info.
* **Video Mode** → Generates a processed video with overlays for detected vehicles.
* **UI (index.html)** → Can be extended to visualize real-time predictions.

Example outputs include:

* Vehicle counts
* Congestion estimation
* Highlighted bounding boxes on cars

---

## 🤝 Contribution

Contributions are welcome!

1. Fork the repo
2. Create your branch:

   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes
4. Push to the branch
5. Open a Pull Request 🚀

---

## 📜 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgments

* [YOLOv5](https://github.com/ultralytics/yolov5) & [YOLOv8](https://github.com/ultralytics/ultralytics) for pretrained models
* [OpenCV](https://opencv.org/) Haar Cascade for vehicle detection
* [IDD-Lite Dataset](https://idd.insaan.iiit.ac.in/dataset/download/) (Indian Driving Dataset)

---

## 📧 Contact

Maintained by **Sarthak Baliyan**
📍 Roorkee, Uttarakhand, India
💼 [GitHub](https://github.com/BALIYANSARTHAK)

---
