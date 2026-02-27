# Pneumonia Classification Web App

## 📖 About the Project
This project is an interactive web application built with **Streamlit** to classify chest X-ray images, identifying the presence of pneumonia. The Artificial Intelligence model was trained using Google Teachable Machine and adapted to run lightweight and efficiently using **TensorFlow Lite**.

The adaptation to use the `ai_edge_litert` (TFLite) library instead of traditional Keras was made to ensure optimized execution and avoid compatibility issues on macOS.

## 📊 Dataset and Training
The data used to train the model comes from a public dataset available on Mendeley Data:
🔗 **Dataset:** [Chest X-Ray Images (Pneumonia)](https://data.mendeley.com/datasets/rscbjbr9sj/2)

**Training Parameters (Teachable Machine):**
* **Epochs:** 15
* **Batch Size:** 128
* **Exported Model:** TensorFlow Lite (Unquantized)

## 🚀 Technologies Used
* **Python 3.x**
* **Streamlit:** For creating the user-friendly web interface.
* **AI Edge LiteRT (`ai_edge_litert.interpreter`):** For TFLite model inference, replacing the need for the full Keras/TensorFlow suite on Mac.
* **NumPy & Pillow (PIL):** For image manipulation, resizing, and normalization.

## 📁 Folder Structure
Ensure that the model files exported by Teachable Machine are in the correct directories before running the code:

```text
├── main.py                              # Main Streamlit script
├── data/                                # Dataset folder
│   ├── train/                           # Images used for training
│   └── test/                            # Images used for testing/validation
└── model/
    ├── model_unquant.tflite             # TFLite model weights
    └── labels.txt                       # Prediction classes (e.g., Normal, Pneumonia)

```

## ⚙️ How to Run Locally
1. Clone the repository
    ```bash
    git clone https://github.com/lorranybarross/pneumonia_classification.git
    cd pneumonia_classification
2. Install dependencies

    Create a virtual environment (optional but recommended) and install the required libraries using pip:
    ```bash
    pip install streamlit numpy Pillow ai-edge-litert
3. Start the Streamlit app
    ```bash
    streamlit run main.py
4. Test the application

    * Your browser will automatically open at `http://localhost:8501`.
    * Upload a chest X-ray image in `.jpg`, `.jpeg`, or `.png` format.
    * The model will run the inference and display the predicted diagnosis along with the confidence score.

## 🧠 How Pre-processing Works
When a user uploads an image through the interface, the script performs the following steps:

1. Resizes the image to `224x224` pixels using the LANCZOS filter, which is the format expected by the Teachable Machine model.
2. Converts the image into a NumPy array.
3. Normalizes the pixel values to a range of `-1` to `1` (formula: `image / 127.5 - 1`).
4. Passes the formatted tensor to the TFLite interpreter, which returns the class with the highest probability.
