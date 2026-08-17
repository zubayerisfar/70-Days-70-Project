# Crowd Density Estimation using CNN

A deep learning project that estimates the number of people in a crowd image using a Convolutional Neural Network (CNN).

The system generates a density map from an input image and estimates the crowd count based on the predicted density distribution.

The project includes a Streamlit web application where users can upload an image and view:

- Original image
- Predicted density heatmap
- Visual overlay
- Estimated crowd count


## Features

- Image-based crowd counting
- CNN based density map generation
- U-Net style architecture
- Streamlit interactive web interface
- Heatmap visualization
- Crowd count estimation


# Project Structure

```
Crowd Density Estimation using CNN/

│
├── Crowd_Density_Website/
│   │
│   ├── app.py
│   ├── crowd_counting_unet_clean.keras
│   ├── model_metadata.json
│   └── requirements.txt
│
│
└── Models and Codes/
    │
    ├── best_crowd_density_model.weights.h5
    ├── crowd_counting_unet.keras
    ├── crowd_counting_unet.weights.h5
    ├── convert_weights.py
    └── other conversion files
```


# Application

The main application is located inside:

```
Crowd_Density_Website/app.py
```

It provides a web interface using Streamlit.

The application:

1. Loads the trained CNN model
2. Accepts an uploaded crowd image
3. Preprocesses the image
4. Generates a density map
5. Calculates estimated crowd count
6. Displays visualization results


# Model Information

Deployment model:

```
crowd_counting_unet_clean.keras
```

Architecture:

```
U-Net based CNN
```

Input:

```
Image size:
240 x 320

Channels:
3 (RGB)

Normalization:
Pixel values divided by 255.0
```

Output:

```
Density map

Shape:
240 x 320 x 1
```


# Crowd Count Calculation

The model predicts a density map.

The estimated number of people is calculated using:

```
Estimated Count = sum(predicted_density_map) / 1000
```


# Installation

## Requirements

Recommended Python version:

```
Python 3.10 or Python 3.11
```

Install required packages:

```
pip install -r requirements.txt
```


# Running the Application

Navigate to the website folder:

```
cd Crowd_Density_Website
```

Run Streamlit:

```
streamlit run app.py
```

The application will open in the browser:

```
http://localhost:8501
```


# How To Use

1. Open the web application

2. Upload a crowd image.

Supported formats:

```
.jpg
.jpeg
.png
```

3. The application will generate:

- Estimated crowd count
- Density heatmap
- Density overlay visualization


# Model Conversion

The original Kaggle model was converted into a deployment-compatible Keras model.

The conversion files were only required during development:

```
crowd_counting_unet.keras
crowd_counting_unet.weights.h5
best_crowd_density_model.weights.h5
convert_weights.py
```

The final application only requires:

```
crowd_counting_unet_clean.keras
```


# Dataset

Training dataset:

```
Mall Crowd Density Dataset
```

Task:

```
Crowd Density Estimation
```


# Model Performance

Validation Results:

```
MAE:
2.146

RMSE:
2.721

MSE:
7.404

Pearson Correlation:
0.891
```


# Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow
- Matplotlib


# Deployment

To move this project to another computer:

1. Copy the entire project folder

2. Install Python

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Start the application:

```
streamlit run app.py
```


# Future Improvements

Possible improvements:

- Real-time CCTV crowd monitoring
- Video crowd counting
- Cloud deployment
- Mobile application integration
- Improved visualization


# Author

Crowd Density Estimation using CNN Project