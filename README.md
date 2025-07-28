# 🌾 Rice Grain Classification using CNN
This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify rice grains into five varieties:

Arborio

Basmati

Ipsala

Jasmine

Karacadag

The goal is to leverage deep learning for accurate identification of rice types, which can be beneficial in agriculture, food quality control, and automation systems.

### 📂 Dataset
Name: Rice Image Dataset

Total Images: 75,000 (15,000 per class)

Classes: 5 rice varieties

Source: Provided dataset (archive1.zip) mounted from Google Drive
https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

### 🧠 Tech Stack
Python

TensorFlow / Keras

OpenCV

NumPy & Pandas

Matplotlib & Seaborn

Google Colab (for training)

### 🚀 Project Workflow
Data Preparation

Mounted Google Drive and extracted the dataset

Resized images to 250×250 pixels

Applied grayscale conversion

Normalized pixel values [0,1]

Split dataset into training (80%) and validation (20%)

### Model Building

Built a CNN with multiple Conv2D, MaxPooling, Dropout, and BatchNormalization layers

Used Adam optimizer with learning rate = 0.0001

Loss Function: Categorical Crossentropy

Output Layer: Dense with Softmax (for 5 classes)

Training

Trained for 4 epochs with batch size = 32

Achieved ~97% training accuracy and ~96% validation accuracy

### Evaluation

Plotted training/validation accuracy and loss curves

Generated confusion matrix for class-wise performance

Evaluated accuracy on unseen validation data

Prediction

Successfully tested with single images

Achieved 100% confidence on test cases

### 📊 Results
Training Accuracy: ~97%

Validation Accuracy: ~96%

Test Prediction Example:

less
Copy
Edit
Predicted Class: Ipsala (Confidence: 1.00)
Confusion Matrix: Shows minimal misclassifications

### 📌 Conclusion
The CNN model effectively classifies rice grains into five distinct varieties with high accuracy.
This approach demonstrates the usefulness of deep learning in agricultural applications, especially in automating quality control processes.

### 🔮 Future Scope
Use Transfer Learning (e.g., MobileNet, VGG16) for faster training

Deploy the model as a Streamlit Web App for real-time predictions

Expand dataset with more rice varieties and different image conditions

Integrate with IoT for automated sorting in agriculture industries

### Author
Rushikesh

Passionate about AI & ML in real-world applications

📫 LinkedIn Profile (add your LinkedIn URL here)
