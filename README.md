# UM-Projects
Unified Mentor Internship Projects

Project 1: Animal Image Classification

Project 2: Forest Cover Type Prediction


---

# Project 1: Image Classification of Animals

## Project Overview
This project involves building a deep learning model to classify images of animals into one of 15 distinct categories. The dataset consists of images for each class, and the model is designed to identify the correct animal based on the input image. Techniques such as Convolutional Neural Networks (CNN) and Transfer Learning are utilized to build an efficient and accurate classifier.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation and Results](#evaluation-and-results)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

---

## Dataset
The dataset consists of 15 folders, each containing images of a particular animal species. All images are 224x224 pixels with 3 color channels (RGB), making them suitable for image classification tasks.

### Animal Classes:
- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

### Dataset Specifications:
- **Image Dimensions**: 224x224 pixels
- **Color Channels**: 3 (RGB)
- **Total Classes**: 15

---

## Project Structure
```
.
├── data/
│   ├── train/
│   │   ├── Bear/
│   │   ├── Bird/
│   │   ├── ... (13 other classes)
│   └── test/
│       ├── Bear/
│       ├── Bird/
│       ├── ... (13 other classes)
├── notebooks/
│   ├── Image_Classification.ipynb
├── models/
│   └── model.h5  # Trained model saved here
├── README.md
└── requirements.txt
```

---

## Model Architecture

The project uses **Convolutional Neural Networks (CNN)** to extract features from the images and classify them into the correct animal category. Additionally, **Transfer Learning** is employed using pre-trained models like VGG16 or ResNet50 to improve performance.

### CNN Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))  # 15 classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### Transfer Learning Model (Using VGG16)

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(15, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## Training Process

The model is trained on the dataset using data augmentation techniques to increase model robustness and prevent overfitting.

### Data Augmentation and Preprocessing

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height, img_width = 224, 224

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

### Model Training

```python
epochs = 10

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
```

---

## Evaluation and Results

After training the model, it was evaluated on the validation set:

```python
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy * 100:.2f}%')
```

The current model achieved an accuracy of approximately **54.31%** on the validation set. There is scope for improvement by using advanced techniques such as **Transfer Learning** and **Hyperparameter Tuning**.

---

## How to Run

### Prerequisites:
- Python 3.x
- TensorFlow 2.x
- Keras

### Steps to Run:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/animal-image-classification.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Organize the dataset into `data/train` and `data/test` folders.

4. Train the model by running the Jupyter notebook:
    ```bash
    cd notebooks
    jupyter notebook Image_Classification.ipynb
    ```

---

## Future Improvements

- **Transfer Learning**: Implementing pre-trained models such as **ResNet50**, **InceptionV3**, or **EfficientNet** to boost model performance.
- **Fine-Tuning**: Unfreezing some layers of the pre-trained models for fine-tuning on the specific dataset.
- **Hyperparameter Tuning**: Experimenting with batch size, learning rate, and number of layers for better accuracy.
- **Data Augmentation**: Further experimenting with data augmentation techniques like brightness and contrast shifts.

---

Feel free to fork this repository and contribute by creating pull requests!

---






---

# Project 2: Forest Cover Type Prediction

### Objective
The objective of this project is to build a machine learning model that predicts the type of forest cover in a given area using various environmental features. The dataset used for this project is derived from the Roosevelt National Forest in northern Colorado, provided by the forest department.

### Problem Statement
We aim to classify the type of forest cover in a 30m x 30m patch of land into one of the seven forest cover types:
1. **Spruce/Fir**
2. **Lodgepole Pine**
3. **Ponderosa Pine**
4. **Cottonwood/Willow**
5. **Aspen**
6. **Douglas-fir**
7. **Krummholz**

The model will use 12 primary features, including elevation, slope, soil type, and other environmental variables, to make accurate predictions.

### Dataset
The dataset consists of both categorical and numerical features. The key features include:
- **Elevation**: Elevation in meters.
- **Aspect**: Aspect in degrees azimuth.
- **Slope**: Slope in degrees.
- **Horizontal and Vertical Distances**: Distance to nearest hydrological, roadways, and fire points.
- **Hillshade**: Hillshade index at various times of the day.
- **Wilderness Area**: Binary columns indicating different wilderness areas.
- **Soil Type**: Binary columns for different soil types.
- **Cover_Type**: The target variable representing the forest cover type.

### Project Workflow
1. **Data Preprocessing**: The dataset is preprocessed, including handling missing values (if any), feature scaling, and encoding categorical variables.
2. **Exploratory Data Analysis (EDA)**: Initial exploration of the dataset to understand feature distributions and relationships.
3. **Model Building**: Various machine learning models are tested, including:
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - XGBoost
4. **Model Evaluation**: Models are evaluated using metrics like accuracy, precision, recall, and a confusion matrix.
5. **Model Saving**: The trained model is saved using `joblib` for future use or deployment.

---

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Contributors](#contributors)

---

## Installation
### Requirements:
To run this project, you'll need to install the following libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### Running the Project:
1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/forest-cover-type-prediction.git
    ```
2. Upload your dataset to the working directory (e.g., `forest_cover.csv`).
3. Open the Jupyter notebook or Google Colab file and follow the steps provided in the notebook.

---

## Project Structure

```
├── forest_cover_type_model.pkl     # Trained Random Forest model
├── forest_cover.csv                # Dataset used for training and testing
├── README.md                       # Project documentation
├── requirements.txt                # Required libraries
└── notebook.ipynb                  # Jupyter/Colab notebook for the project
```

---

## Usage

### 1. Data Loading
First, load the dataset into a pandas DataFrame:
```python
import pandas as pd
df = pd.read_csv('forest_cover.csv')
```

### 2. Data Preprocessing
Preprocess the dataset by scaling numerical features and splitting the data into training and test sets:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.iloc[:, :10])  # Scale only numerical columns
X_test_scaled = scaler.transform(X_test.iloc[:, :10])
```

### 3. Model Training
Train the Random Forest Classifier:
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
```

### 4. Model Evaluation
Evaluate the model on test data:
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predictions
y_pred = rf_model.predict(X_test_scaled)

# Accuracy and Classification Report
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Confusion Matrix
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
```

### 5. Save and Load the Model
Save the trained model for future use:
```python
import joblib
joblib.dump(rf_model, 'forest_cover_type_model.pkl')

# Load the model for future predictions
loaded_model = joblib.load('forest_cover_type_model.pkl')
```

---

## Model Evaluation

### Accuracy: `XX.XX%`
The Random Forest model achieved an accuracy of **XX.XX%** on the test set.

### Confusion Matrix:
```
[[TP, FP],
 [FN, TN]]
```

### Classification Report:
| Class            | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| Spruce/Fir       | x.xx      | x.xx   | x.xx     |
| Lodgepole Pine   | x.xx      | x.xx   | x.xx     |
| ...              | ...       | ...    | ...      |

---

## Contributors
- **Saurabh Yadav** - [GitHub Profile](https://github.com/SaurabhIndi)

---

