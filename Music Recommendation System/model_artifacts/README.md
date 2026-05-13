# Music Recommendation Model

## Files Included:
- `music_recommendation_model.pkl` - Trained KNN model
- `feature_scaler.pkl` - StandardScaler for feature normalization
- `music_dataset.csv` - Original dataset with processed features
- `model_metadata.json` - Model configuration and feature information

## Usage in API:

```python
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model components
with open('music_recommendation_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for song lookup
df = pd.read_csv('music_dataset.csv')

# Function to get recommendations
def get_recommendations(song_index, num_recommendations=5):
    # Get features from dataset
    features = df[audio_features].iloc[song_index:song_index+1]
    # Scale features
    features_scaled = scaler.transform(features)
    # Get nearest neighbors
    distances, indices = knn_model.kneighbors(features_scaled, n_nei    ghbors=num_recommendations+1)
    # Return recommended songs
    return df.iloc[indices[0][1:]]
```

## Model Details:
- Algorithm: K-Nearest Neighbors (KNN)
- Metric: Cosine Similarity
- Features: 10 audio features + Genre one-hot encoding + Emotion encoding
- Training samples: 236,916 songs
