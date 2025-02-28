# Image Captioning

## Introduction
This project implements an Image Caption Generator using Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The model is inspired by the paper "Show and Tell: A Neural Image Caption Generator" and is implemented using TensorFlow and Keras. The dataset used is Flickr 8K, which consists of 8,000 images, each paired with five different captions.

### Model Architecture
The model consists of:
- **CNN (Convolutional Neural Network)**: Extracts image features and encodes the input image.
- **RNN (Recurrent Neural Network) with LSTM (Long Short-Term Memory)**: Generates captions based on the extracted image features. The image embedding is provided as the first input to the RNN network and only once.

![Model Architecture](https://github.com/raunak222/Image-Captioning/raw/master/Image/decoder.png)

## Dataset
The project utilizes the [Flickr 8K dataset](https://www.kaggle.com/adityajn105/flickr8k), which can be downloaded and processed using the Kaggle API.

## Dependencies
The project requires the following dependencies:
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- TQDM
- NLTK
- Kaggle API

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NgocHaiNguyen14/ImageCaptioning.git
   cd ImageCaptioning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("adityajn105/flickr8k")
   print("Path to dataset files:", path)
   ```

## Usage
Run the Jupyter Notebook `image_captioning.ipynb` to train and evaluate the model.

## References
1. [Image captioning with visual attention](https://www.tensorflow.org/tutorials/text/image_captioning)
2. [RNNs in Computer Vision — Image Captioning](https://thinkautonomous.medium.com/rnns-in-computer-vision-image-captioning-597d5e1321d1)
3. [Image Captioning Project from Udacity Computer Vision Nanodegree](https://github.com/raunak222/Image-Captioning)
4. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)
5. [Flickr 8k Dataset](https://www.kaggle.com/adityajn105/flickr8k)
