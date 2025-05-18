# 🧠 Digit Detector using PyTorch

A deep learning project to recognize handwritten digits using a Convolutional Neural Network (CNN) built with PyTorch and trained on the MNIST dataset.

## 📌 Features

- Trains a CNN on the MNIST dataset
- Achieves high accuracy (~98%) on test data
- Saves the trained model
- Loads the model to predict digits from real-world custom images
- Supports CPU and GPU

---

## 🛠️ Tech Stack

- Python 🐍
- PyTorch 🔥
- Torchvision 🖼️
- MNIST Dataset
- PIL (for custom image prediction)
- Matplotlib (for optional visualization)

---

## 📂 Project Structure

```bash
digit-detector/
├── data/                     # MNIST data (auto downloaded)
├── digit_detector.pth        # Saved model weights
├── my_digit.png              # (Optional) Custom image for testing
├── main.py                   # Training and testing script
├── predict.py                # Prediction from custom image
└── README.md                 # Project documentation
```
## 🚀 Getting Started
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/digit-detector.git
cd digit-detector
```
### 2. Install dependencies
```bash
pip install torch torchvision matplotlib pillow
```
## 🧠 Train the Model
Run this script to train and save the CNN:

```bash
python main.py
```
## 🔍 Predict Custom Digits
Save a 28x28 black and white image (my_digit.png) in the repo folder and run:

```bash
python predict.py
```

It will output the predicted digit (0–9).

## 🖼️ Example Output
```yaml
Epoch 1, Loss: 0.2456
Epoch 2, Loss: 0.0782
Test Accuracy: 98.25%
Predicted Digit: 7
```
## 📌 Notes
Ensure custom images are 28x28 pixels, grayscale, and centered.

You can use tools like Paint or Pillow to create custom digits.

## 📄 License
This project is licensed under the MIT License.

## 🙌 Acknowledgements
- PyTorch

- MNIST Dataset
