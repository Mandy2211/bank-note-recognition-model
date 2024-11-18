# Bank Note Recognition Model 🏦💵

This project is a **machine learning model** designed to recognize and classify Indian banknotes of ₹10 and ₹100 denominations.
The model is built using a Convolutional Neural Network (CNN) architecture and trained on images of banknotes to distinguish between the two classes.

# Features
- Recognizes ₹10 and ₹100 banknotes.
- Trained using a dataset of images, resized and preprocessed to ensure accuracy.
- Implements data augmentation to improve the model's generalization.
- Achieves high accuracy on the test dataset.


# Model Architecture
The model uses a Convolutional Neural Network (CNN) inspired by the LeNet architecture:
- Input Shape: 28x28x3 (RGB images resized to 28x28 pixels)
- Activation Functions: ReLU and Softmax
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Data Augmentation: Applied using `ImageDataGenerator`.



# How to Use
1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   Open `Bank_Note_Recognition.ipynb` in Jupyter Notebook or Google Colab to train the model, evaluate it, or classify new banknote images.

4. Load the trained model:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('banknote_model.h5')
   ```

5. Predict on new images:
   ```python
   import cv2
   import numpy as np

   # Load and preprocess an image
   image = cv2.imread('path_to_image.jpg')
   image = cv2.resize(image, (28, 28))
   image = image.astype("float") / 255.0
   image = np.expand_dims(image, axis=0)

   # Predict the class
   prediction = model.predict(image)
   label = "₹10" if prediction[0][0] > 0.5 else "₹100"
   print(f"Predicted: {label}")
   ```


---

# Contributing
Feel free to contribute to this project by:
- Improving the dataset
- Optimizing the model architecture
- Extending it to recognize additional denominations

---

# Acknowledgments
- The dataset was created from images of Indian banknotes.
- Built using TensorFlow, Keras, and OpenCV.

---

Let me know if you'd like to modify or add anything specific! 😊
