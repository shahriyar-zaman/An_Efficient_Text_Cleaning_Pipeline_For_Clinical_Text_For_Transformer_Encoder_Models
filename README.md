
# An Efficient Text Cleaning Pipeline for Clinical Text for Transformer Encoder Models

<p align="center"> <strong>Official Implementation for IEEE IS'24 Submission</strong> </p>

## Abstract

This research explores preprocessing clinical text data for enhancing the performance of transformer models, which are vital in the field of Natural Language Processing (NLP). Transformer models like BERT, BioBERT, BioClinicalBERT, and RoBERTa benefit significantly from effective preprocessing techniques. We introduce a new pipeline that removes repeated punctuation, normalizes text, and utilizes TF-IDF to filter less significant words. The proposed pipeline improves accuracy, particularly in clinical settings where precision is essential.

We evaluate our approach on the MIMIC-3 and PubMed datasets, demonstrating up to 90.16% accuracy on MIMIC-3 and 64.20% accuracy on PubMed datasets.

## Clinical Text Dataset

### Dataset Statistics

| Metric                         | MIMIC-III (Mortality Prediction) | PubMed (Medical Text) |
| ------------------------------ | -------------------------------- | --------------------- |
| Training Samples               | 33,954                           | 9,550                 |
| Test Samples                   | 9,822                            | 2,888                 |
| Class Labels                   | 2 (Mortality: Yes/No)            | 5 (Conditions)        |


## Methodology

1. **Data Preprocessing**: Removal of repeated punctuation, text normalization, and TF-IDF filtering to retain significant words while reducing noise.
2. **Model Training**: Fine-tuning transformer models (BERT, BioBERT, BioClinicalBERT, RoBERTa) using the proposed pipeline.
3. **Evaluation**: Models were evaluated on MIMIC-3 and PubMed datasets for improvements in accuracy.
4. **Result Analysis**: Models trained with the proposed pipeline consistently showed better accuracy compared to baseline preprocessing methods.

![image](https://github.com/shahriyar-zaman/An_Efficient_Text_Cleaning_Pipeline_For_Clinical_Text_For_Transformer_Encoder_Models/blob/cff99d775cbea4921f9e32ed074ac4040452d487/Figures/sys_acrh.png)

**Fig. 1: Workflow of the preprocessing pipeline and model evaluation.**

## Results

### Model Performance on Clinical Text Datasets

#### PubMed (Medical Text) Dataset

| Model             | Baseline Accuracy (%) | Pipeline Accuracy (%) |
| ----------------- | --------------------- | --------------------- |
| BERT-base         | 60.87                 | 62.43                 |
| BioBERT           | 61.36                 | 64.20                 |
| BioClinicalBERT   | 62.43                 | 63.05                 |
| RoBERTa           | 61.88                 | 63.37                 |

#### MIMIC-III (Mortality Prediction) Dataset

| Model             | Baseline Accuracy (%) | Pipeline Accuracy (%) |
| ----------------- | --------------------- | --------------------- |
| BERT-base         | 89.01                 | 90.16                 |
| BioBERT           | 88.92                 | 89.83                 |
| BioClinicalBERT   | 89.24                 | 89.97                 |
| RoBERTa           | 89.53                 | 89.36                 |

## 🚀 Quick Start

To get started with our models, follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/shahriyar-zaman/An_Efficient_Text_Cleaning_Pipeline_For_Clinical_Text_For_Transformer_Encoder_Models.git
cd Clinical_Text_Cleaning_Pipeline
```

### 2. Install Required Packages

Run the following commands to install the necessary libraries:

```bash
pip install transformers torch datasets scikit-learn
```

### 3. Run the Preprocessing Pipeline

Use the following command to run the preprocessing pipeline:

```bash
python preprocess_pipeline.py --input_file <input_data_file> --output_file <output_data_file>
```

## 🧪 Train-Test-Split

- **Training**: 33,954 (MIMIC-III), 9,550 (PubMed)
- **Testing**: 9,822 (MIMIC-III), 2,888 (PubMed)

Feel free to modify the splits or experiment with different datasets based on your use case.

## 📚 Model Training

For those interested in fine-tuning the models further, we recommend using the `train_model.py` script, which includes hyperparameters and configurations for:

- Epochs: 10 (Early Stopping)
- Batch Size: 32
- Learning Rate: 2e-5
- Models: BERT, BioBERT, BioClinicalBERT, RoBERTa

```bash
python train_model.py --model bert-base --epochs 10 --batch_size 32 --learning_rate 2e-5
```

## References

1. J. Johnson et al., "MIMIC-III: Medical Information Mart for Intensive Care III," 2016.
2. W. Lee et al., "BioBERT: A Pre-trained Biomedical Language Representation Model," 2020.
3. Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," 2019.
4. Z. Peng et al., "BioClinicalBERT: Biomedical and Clinical Language Models for Medical Text Mining," 2020.
