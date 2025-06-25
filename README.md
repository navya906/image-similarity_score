# 🖼️ Image Similarity Score

A simple Python project to compute similarity scores between images using deep learning models. This application leverages **ResNet50** and **cosine similarity** to compare image embeddings.

## 📌 Features

- Upload two images and calculate how visually similar they are.
- Uses **ResNet50** from Keras Applications for feature extraction.
- Computes **cosine similarity** between feature vectors.
- Simple and modular structure for easy extension.

## 📂 Project Structure
```
image-similarity_score/
│
├── images/ # Example image pairs
├── app.py # Main script
├── utils.py # Image processing and feature extraction
├── requirements.txt # Required Python libraries
└── README.md # Project documentation
```


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/navya906/image-similarity_score.git
cd image-similarity_score
```

### 2. Set Up the Environment
It's recommended to use a virtual environment:\

```
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install requirements
```
pip install -r requirements.txt
```

### 4. Run the App
```
python app.py
```

### 🛠️ Technologies Used
- Python 🐍
- TensorFlow / Keras
- NumPy
- OpenCV / PIL
- Scikit-learn (for cosine similarity)


### 🧑‍💻 Author
Navya Ghatta\
Computer Science Student | [LinkedIn](https://www.linkedin.com/in/navya-g-a97051314/)

