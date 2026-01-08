# Mushroom Classification with Convolutional Neural Networks

## Project Overview
This project implements a convolutional neural network (CNN) for mushroom image classification,
with the goal of distinguishing between edible and poisonous mushroom categories.
Given the safety-critical nature of the task, the model prioritises minimising false negatives
(i.e. misclassifying poisonous mushrooms as edible).

---

## Model & Approach
- The model is based on ResNet18 pretrained on ImageNet.
- Transfer learning was applied by freezing the backbone network and fine-tuning only the final
  fully connected layer.
- The classifier predicts 10 mushroom classes, consisting of 5 edible and 5 poisonous categories.

---

## Data Augmentation and Preprocessing
Data augmentation was applied during training to improve generalisation and reduce overfitting.
Techniques included random horizontal and vertical flips to account for orientation variability,
random resized cropping to introduce scale and positional variation, and random brightness and
contrast adjustments to simulate different lighting conditions.

For validation and testing, no data augmentation was applied. Images were only resized to 224×224,
converted to tensors, and normalised using ImageNet mean and standard deviation values to ensure
consistent and unbiased evaluation.

---

## Model Performance Summary
The model demonstrates strong performance in detecting poisonous mushrooms.
Validation accuracy stabilised between 0.60 and 0.63, while recall reached 0.99, indicating that
nearly all poisonous mushrooms were correctly identified. The model also achieved an AUC of 0.95,
showing strong discriminative capability.

This high recall comes at the cost of lower precision and specificity, as some edible mushrooms
were incorrectly classified as poisonous. Overall, the model prioritises safety while maintaining
a reasonable balance between recall and precision.

---

## Ethical Considerations and Evaluation Metrics
This task involves asymmetric risk: misclassifying a poisonous mushroom as edible can have severe
consequences, whereas misclassifying an edible mushroom as poisonous primarily affects usability.
Accordingly, recall (sensitivity) was prioritised as the primary evaluation metric, supported by
precision, specificity, confusion matrices, and ROC-AUC analysis to balance safety and reliability.

---

## Limitations
The reported results were obtained using training and validation splits derived from the same
dataset. As a result, the evaluation may overestimate the model’s true generalisation performance.
Before real-world deployment, evaluation on a fully independent held-out test dataset is strongly
recommended.

---

## Repository Structure
.
├── notebooks/      # Training and evaluation notebooks
├── src/            # Model and data loading code
├── .gitignore
└── README.md
