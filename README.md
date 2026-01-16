Plant Leaf Disease Detection & Segmentation
------------------------------------------------------------------
Computer Vision | PyTorch | CNN | Semantic Segmentation

One-Line Description
------------------------------------------------------------------

Built an end-to-end computer vision system to classify plant leaf diseases and segment infected regions using deep learning and advanced image preprocessing.

Tech Stack
------------------------------------------------------------------

PyTorch, OpenCV, Albumentations, Scikit-learn, Matplotlib, Seaborn

CNN (Custom Architecture), DeepLabV3+ (ResNet-101 Backbone)

Pipeline Design
------------------------------------------------------------------

Implemented semantic segmentation using DeepLabV3+ to remove complex backgrounds before classification

Built a custom PyTorch Dataset & DataLoader pipeline for scalable training and testing

Designed modular preprocessing stages for reproducibility and easy deployment

Image Processing
------------------------------------------------------------------

Applied Gaussian filtering for noise reduction

Used CLAHE for contrast enhancement to highlight disease patterns

Standardized inputs via resize + normalization (256×256 RGB)

Data Augmentation
------------------------------------------------------------------

Integrated Albumentations for real-world robustness

Used rotation, flips, brightness & contrast transformations to reduce overfitting

Model Development
------------------------------------------------------------------

Designed a custom CNN classifier for 38 plant disease classes

Used Cross-Entropy Loss + Adam Optimizer for stable training

Supported GPU acceleration (CUDA) for faster convergence

Evaluation & Analysis
------------------------------------------------------------------

Generated classification reports (Precision, Recall, F1-Score)

Built raw and normalized confusion matrices for per-class performance analysis

Visualized error patterns and class imbalance

Visual Explainability
------------------------------------------------------------------

Produced segmented infection masks to highlight diseased leaf regions

Saved prediction outputs and visual results for interpretability and demos

Uniqueness
------------------------------------------------------------------

Combined classical image enhancement + deep learning segmentation + CNN classification in one unified pipeline

Focused on real-world image robustness, not just clean dataset performance

Designed for explainability, not just accuracy metrics

What I Learned
------------------------------------------------------------------

Built production-style ML pipelines using PyTorch

Integrated pretrained segmentation models into custom workflows

Performed multi-class performance analysis and visualization

Improved model generalization using augmentation and contrast enhancement

Future Scope
------------------------------------------------------------------

Deploy as a Streamlit Web Application-- will update it soon

Support mobile camera-based disease detection
