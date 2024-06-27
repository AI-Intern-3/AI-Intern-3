Writing data preprocessing and cleaning in Python for real-time applications involves using libraries like Pandas, NumPy, and scikit-learn for general data processing, and specialized libraries like OpenCV for image processing. Below are examples demonstrating preprocessing and cleaning for a face recognition system, sentiment analysis, and predictive maintenance.

### 1. Face Recognition System

#### Preprocessing and Cleaning Images

```python
import cv2
import numpy as np
import glob
import os

# Path to the images
image_path = "path_to_images/"

# Function to preprocess images
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image to 128x128
    resized_image = cv2.resize(gray_image, (128, 128))
    # Normalize the image
    normalized_image = resized_image / 255.0
    return normalized_image

# Function to clean images (remove blurry images)
def is_blurry(image, threshold=100):
    # Compute the Laplacian of the image and then the variance
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold

# Preprocess and clean images
for file in glob.glob(os.path.join(image_path, "*.jpg")):
    image = cv2.imread(file)
    if is_blurry(image):
        os.remove(file)
    else:
        processed_image = preprocess_image(image)
        # Save or use processed_image
```

### 2. Sentiment Analysis on Social Media

#### Preprocessing and Cleaning Text Data

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample data
data = {"text": ["I love this product! #awesome", "This is terrible... http://example.com"]}
df = pd.DataFrame(data)

# Function to clean and preprocess text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^A-Za-z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
```

### 3. Predictive Maintenance in Manufacturing

#### Preprocessing and Cleaning Time-Series Data

```python
import pandas as pd
import numpy as np

# Sample sensor data
data = {
    "timestamp": pd.date_range(start='1/1/2021', periods=100, freq='H'),
    "sensor1": np.random.randn(100).cumsum(),
    "sensor2": np.random.randn(100).cumsum()
}
df = pd.DataFrame(data)

# Function to preprocess sensor data
def preprocess_sensor_data(df):
    # Fill missing values with interpolation
    df = df.interpolate()
    # Normalize sensor data
    df['sensor1'] = (df['sensor1'] - df['sensor1'].mean()) / df['sensor1'].std()
    df['sensor2'] = (df['sensor2'] - df['sensor2'].mean()) / df['sensor2'].std()
    return df

# Cleaning function to filter noise
def filter_noise(df, threshold=1.5):
    df['sensor1'] = df['sensor1'].apply(lambda x: x if np.abs(x) < threshold else np.nan).interpolate()
    df['sensor2'] = df['sensor2'].apply(lambda x: x if np.abs(x) < threshold else np.nan).interpolate()
    return df

# Apply preprocessing and cleaning
df = preprocess_sensor_data(df)
df = filter_noise(df)
```

### Explanation

1. **Face Recognition System**:
   - **Preprocessing**: Converts images to grayscale, resizes them, and normalizes pixel values.
   - **Cleaning**: Detects and removes blurry images.

2. **Sentiment Analysis**:
   - **Preprocessing**: Removes URLs, mentions, hashtags, special characters, and numbers. Tokenizes, removes stop words, and lemmatizes the text.
   - **Vectorization**: Converts cleaned text into a matrix of token counts.

3. **Predictive Maintenance**:
   - **Preprocessing**: Fills missing sensor values using interpolation and normalizes the data.
   - **Cleaning**: Filters out noise by setting values beyond a threshold to NaN and interpolating.

These examples show how to implement data preprocessing and cleaning in Python for different real-time applications.
