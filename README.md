# RetinaAI | AI-Assisted Retinal Screening & Clinical Decision Support

## Project Overview

RetinaAI is an AI-assisted retinal screening and clinical decision support research prototype developed to support ophthalmologists in diabetic retinopathy assessment, disease severity grading, clinical interpretation, and structured reporting.

The application integrates a custom-trained deep learning classifier with a generative AI model within an interactive clinical dashboard.

## Key Features

- Retinal fundus photograph uploading and visualization.
- AI-assisted diabetic retinopathy screening.
- Four-class diabetic retinopathy severity grading.
- Visualization of model classification scores.
- Generative AI-assisted clinical explanations.
- Ophthalmologist review and clinical documentation.
- Automated downloadable PDF assessment reports.
- Local CPU-based execution because of limited GPU capabilities.

## Model Architecture

### Deep Learning Model

A fine-tuned MobileNetV3-Large architecture was developed using transfer learning, data augmentation, weighted cross-entropy loss, and selective feature-layer fine-tuning.

The model classifies retinal photographs into four categories:

- Grade 0: No apparent diabetic retinopathy.
- Grade 1: Mild non-proliferative diabetic retinopathy.
- Grade 2: Moderate non-proliferative diabetic retinopathy.
- Grade 3: Severe non-proliferative or proliferative diabetic retinopathy.

### Generative AI Integration

Qwen2-VL-2B-Instruct generates structured clinical explanations based on the deep learning classification results, including:

- AI classification summary.
- Clinical significance.
- Recommended clinical assessment.
- Model limitations.

The generative AI component explains classifier outputs rather than independently diagnosing retinal photographs.

## Dataset

The combined dataset consists of 3,806 retinal photographs obtained from APTOS 2019 and selected JSIEC fundus image categories.

| Dataset partition | Images |
|---|---:|
| Training | 2,706 |
| Validation | 549 |
| Testing | 551 |
| Total | 3,806 |

## Model Performance

| Evaluation metric | Result |
|---|---:|
| Four-class test accuracy | 84.39% |
| DR detection sensitivity | 98.93% |
| DR detection specificity | 98.15% |
| Advanced DR recall | 68.92% |
| Macro F1-score | 0.77 |
| Weighted F1-score | 0.84 |

The reported results represent internal development evaluation. Binary screening metrics were derived from the four-class predictions. Performance against other retinal diseases and patient-level independence have not been established.

### Model Development and Training Methodology

The diabetic retinopathy classifier was developed using transfer learning and selective fine-tuning of an ImageNet-pretrained MobileNetV3-Large architecture.

The original classification layer was replaced with a four-class output layer for diabetic retinopathy severity classification.

The model development pipeline incorporated:

- Combined APTOS and JSIEC retinal fundus image datasets
- Image preprocessing and resizing to 224 × 224 pixels
- ImageNet-based normalization
- Data augmentation using horizontal flipping and random rotation
- Transfer learning and selective fine-tuning
- Class-weighted cross-entropy loss
- Adam optimization
- Comparative evaluation of four model versions
- Model selection based on validation accuracy and advanced DR recall
- Performance evaluation using Scikit-learn

Version 3 was selected for integration into the clinical decision support application based on its balanced classification performance.

The trained classifier was subsequently integrated with Qwen2-VL-2B-Instruct for structured generative AI-assisted clinical explanations.

### Computational Environment

The entire model development, training, evaluation, and application execution were performed locally without a dedicated GPU.

- Processor: AMD Ryzen 7 7730U
- CPU: 8 cores and 16 logical processors
- RAM: 16 GB
- Graphics: AMD Radeon integrated graphics
- Operating system: Windows 11
- Development environment: Visual Studio Code
- Deep learning framework: PyTorch

This implementation demonstrates the feasibility of developing a lightweight healthcare AI research prototype in a resource-constrained computational environment.

## Clinical Workflow

1. Upload a retinal fundus photograph.
2. Perform AI-assisted diabetic retinopathy classification.
3. Review predicted disease severity and model scores.
4. Generate a structured clinical explanation using Qwen.
5. Record ophthalmologist assessment.
6. Download the clinical assessment report.

## Technology Stack

- Python
- PyTorch
- Torchvision
- MobileNetV3-Large
- Qwen2-VL-2B-Instruct
- Hugging Face Transformers
- Transfer learning
- Fine-Tuning
- Scikit-learn
- Streamlit
- Pillow
- ReportLab
- VS Code

## Running the Application

Install the required dependencies:

`pip install -r requirements.txt`

Place the trained MobileNetV3 model weights in the project directory and ensure that the Qwen model is available locally.

Launch the application:

`python -m streamlit run app.py`

## Future Development

The current implementation focuses on diabetic retinopathy screening and severity assessment.

Future extensions may include differential classification of retinal vascular disorders, macular diseases, glaucoma-related retinal changes, automated lesion localization, uncertainty estimation, and external clinical validation.

Given the complexity and overlapping clinical manifestations of retinal diseases, further development will be guided by ophthalmologist collaboration, clinical priorities, and the availability of appropriately annotated retinal imaging datasets.

## Clinical Disclaimer

RetinaAI is a research and clinical decision support prototype. It is not a standalone diagnostic system or a replacement for professional ophthalmological assessment. Model classification scores are not clinically calibrated probabilities.

## Dataset References

- APTOS 2019 Blindness Detection: https://www.kaggle.com/competitions/aptos2019-blindness-detection
- JSIEC Fundus Image Dataset: https://doi.org/10.5281/zenodo.3477553
