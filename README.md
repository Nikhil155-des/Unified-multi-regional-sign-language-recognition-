# Unified-multi-regional-sign-language-recognition-
# Unified Multi-Regional Sign Language Recognition using Transfer Learning

## Overview

This project focuses on developing a **multi-regional sign language recognition system** using **transfer learning**. The model is trained on **British Sign Language (BSL) and New Zealand Sign Language (NZSL)** and tested on **Indian Sign Language (ISL) and Australian Sign Language (Auslan)** to evaluate its generalization capability across different sign languages.

## Objectives

- Train a deep learning model on BSL and NZSL.
- Test the model's performance on ISL and Auslan.
- Utilize **MLSLT (Multilingual Sign Language Translation)** for feature extraction.
- Create a dataset of 100-200 commonly used words.

## Dataset

- The dataset consists of **100-200 commonly used words** in sign language.
- Training Set: **BSL & NZSL**
- Testing Set: **ISL & Auslan**
- Data is organized as follows:
  ```
  dataset/
  ├── train/
  │   ├── BSL_NZSL/
  │   │   ├── word1.mp4
  │   │   ├── word2.mp4
  ├── test/
  │   ├── ISL_Auslan/
  │   │   ├── word1.mp4
  │   │   ├── word2.mp4
  ```

## Methodology

1. **Data Preprocessing**

   - Convert sign language videos into keypoint pose representations.
   - Normalize and augment data for better generalization.

2. **Feature Extraction**

   - Use **MLSLT** for multilingual sign representation.
   - Extract pose vectors for each sign video.

3. **Model Training**

   - Implement **transfer learning** using:
     - **Inflated 3D Deep CNN** (for spatial-temporal features)
     - **MLSLT** (for sign translation)
     - **GLoT** (for efficient translation)
   - Train on BSL & NZSL dataset.

4. **Evaluation & Testing**

   - Test on ISL & Auslan.
   - Use accuracy, precision, recall, and F1-score for performance evaluation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Nikhil155-des/Unified-multi-regional-sign-language-recognition.git
   cd Unified-multi-regional-sign-language-recognition
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:

   - Download or collect sign language videos.
   - Organize them in the dataset structure mentioned above.

4. Run the training script:

   ```bash
   python train.py
   ```

5. Test the model:

   ```bash
   python test.py
   ```

## Results & Evaluation

- Model performance will be evaluated on ISL & Auslan.
- Metrics include:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Contributions

- Feel free to **contribute** by adding more sign languages or improving the model architecture.
- Submit a pull request with detailed documentation.

## License

This project is open-source and available under the **MIT License**.

## Contact

For queries or collaborations, contact [nikhildeshpande004@gmail.com] or open an issue in the repository.

