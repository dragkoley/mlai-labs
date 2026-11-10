   # Machine Learning and Artificial Intelligence — Labs

This repository contains university laboratory works for the **Machine Learning and Artificial Intelligence** course.

All data preprocessing, analysis, model architectures, and final validation outputs are fully preserved and saved inside the interactive Jupyter Notebook (.ipynb) file for quick and easy review.

## Projects Overview

### Lab 1: Allstate Claims Severity Prediction (Var. 5)
* **Goal:** Predict the severity of insurance claims using historical data from the [Kaggle Allstate Competition](https://kaggle.com).
* **Tech Stack:** python, pandas, numpy, keras, jax, seaborn, matplotlib, scikit-learn.

The initial assignment required building a simple baseline model using a basic `Sequential` neural network. However, driven by **pure enthusiasm** and hours of research into modern architectures for tabular data, I developed a custom, high-performance deep learning model to significantly push the AI's accuracy limits.

Here is the advanced architecture I implemented to handle the complex Allstate dataset:

1. **Entity Embeddings for Categorical Data:** Instead of simple one-hot encoding, each categorical column feeds into its own dedicated `Embedding` layer. This maps sparse text categories into dense continuous vectors, allowing the model to organically learn relationships and semantic similarities between different categories during training.
2. **Hybrid Normalization Strategy:**
    * Applied `BatchNormalization` to the numerical inputs to ensure stable distribution.
    * Swapped BatchNorm for `LayerNormalization` inside the hidden blocks. This proved crucial for heavy-tailed categorical columns like `cat116` (300+ unique values). While BatchNorm tended to smooth out and erase rare category signals across a batch size of 1024, `LayerNorm` normalizes each sample independently, preserving critical unique features.
3. **Pre-Activation ResNet Blocks:** Implemented a modern residual architecture optimized for deep networks. The layers follow the highly stable **Pre-Activation** order: `LayerNorm -> ReLU -> Dropout -> Dense`. Residual connections via addition (`add([x, res])`) create "gradient highways," allowing smooth backpropagation directly through the JAX/GPU backend without bottleneck issues.
4. **Log-Transform Target Engineering:** Optimized training directly using Mean Absolute Error (`loss="mae"`) combined with early stopping (`patience=20`) to prevent overfitting. In the final step, the predictions are inverse-transformed back to original dollar amounts using exponential shifting: `np.exp(pred) - 400`.

#### Why Pre-Activation?
In a standard sequence (`Dense -> LayerNorm -> ReLU`), inputs entering the residual addition are strictly positive and already filtered by the activation function. By flipping the order, the model accumulates outputs *before* activation and normalization, yielding a much cleaner, unconstrained signal. It mitigates unnormalized data propagation and ensures that downstream layers always receive well-behaved, normalized features right after the residual merge.

#### Performance Comparison & Results

By introducing advanced deep learning techniques, the custom architecture completely outperformed the baseline `Sequential` network, showing exceptional stability and accuracy across all 100 training epochs.

| Model Architecture                            | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | Key Takeaway                                                                                                                                            |
|:----------------------------------------------|:-------------------------:|:------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Baseline Network** (`Sequential`)           |          1164.84          |            1918.76             | Showed a stable base result but struggled with heavily skewed data and underpredicted rare, high-value claims to preserve gradient convergence.         |
| **Advanced Custom Net** (ResNet + Embeddings) |        **1149.90**        |           **1895.84**          | **Broke the 1150 MAE barrier!** RMSE dropped significantly by **22.92 points**, successfully capturing complex features and expensive insurance claims. |

#### Why it worked so well:
* **Target Shifting (+400) & MAE Loss:** Perfectly balanced the model's focus between average and large payouts, preventing the loss function from exploding on outliers.
* **Pre-Activation Safeguard:** Kept the weights stable and protected them from inflating over a long 100-epoch run.
* **LayerNorm over BatchNorm:** Instead of washing out rare signals across a large batch size of 1024, `LayerNorm` preserved unique characteristics of complex features (like `cat116`). This allowed the model to meticulously learn the profiles of expensive, non-standard insurance cases, which directly led to the massive drop in RMSE.



### Lab 2: Experiment Tracking & Infrastructure via MLflow (Docker-based)
* **Goal:** Scale the existing ML pipeline by deploying a fully containerized MLOps tracking infrastructure and evaluating multiple experiment runs.
* **Tech Stack:** Docker, Docker Compose, MLflow, PostgreSQL, Rustfs (S3-compatible storage), Keras 3 (JAX on GPU backend), Scikit-Learn.

While MLflow supports unified tracking out of the box, it mixes experiments across different working branches by default. To implement a production-grade, professional setup, I deployed an isolated multi-container environment via Docker Compose to abstract MLOps microservices away from the host system.

#### Enterprise-Grade Infrastructure Architecture

The decoupled network is defined across several highly reliable microservices:
1. **Metadata Store (`PostgreSQL`)**: A separate container instance hosting a relational database to systematically organize hyperparameter grids, metrics, run IDs, and training statuses.
2. **Artifact Store (`Rustfs`)**: An ultra-fast, S3-compatible object storage server container that acts as a secure data vault for preserving compiled training logs, evaluation charts, and trained JAX serialized model binaries (`.keras`).
3. **Tracking Server (`MLflow`)**: The centralized MLOps backend hub connected natively to the database and S3 buckets. It is accessible externally via a dedicated forwarded reverse-proxy port (`5801`), making it completely invisible to host system root execution conflicts.

#### ML Pipeline Integration & Autologging
The tracking system seamlessly intercepts metrics from multiple deep learning and preprocessing frameworks running on the native machine:
* **`mlflow.keras.autolog()`**: Intercepts the JAX high-performance JIT compiler execution loop, automatically tracking deep learning architecture layers, parameters, learning rate decay steps (`ReduceLROnPlateau`), and historical loss steps across every epoch.
* **`mlflow.sklearn.autolog()`**: Automatically captures preprocessing state transitions from `StandardScaler` and `OrdinalEncoder` feature-engineering matrices.

#### Experiment Evaluation Run Matrix
To thoroughly evaluate structural adjustments, a series of 3 distinct, fully isolated experiment iterations were pushed directly into the remote MLflow tracking core:

1. **Run 1 — `Sequential_Base_Model`**: Fixed the baseline network performance metrics and validated the automatic Docker network transmission loops.
2. **Run 2 — `Sequential_Base_Model_large`**: Evaluated expanding hidden capacities (`256 -> 128 -> 64`) and adjusting optimizer batch constraints.
3. **Run 3 — `ResNet_Log_Target`**: Pushed the custom advanced hybrid architecture (from Lab 1) to the server. Since training occurred in the log-transformed space, validation tracking automatically intercepted log-losses, while final production metrics were post-processed via exponential inverse-shifting (`np.exp(pred) - 400`) to guarantee an honest, real-dollar comparison matrix in the MLflow control panel.


### Lab 3: Image Classification with CNNs

This project focuses on image classification using convolutional neural networks (Variant 6: ResNet152).

#### Key Highlights

* **Pretrained Model Inference:** Tested a ready-to-use `ResNet152` pretrained on ImageNet on a sample image to predict the top-5 classes.
* **Custom CNN from Scratch:** Built and trained a lightweight 3-block ResNet architecture with `LayerNormalization` on two datasets:
   * `horses_or_humans`: achieved 99% accuracy.
   * `cats_vs_dogs`: achieved 86% accuracy.
* **Transfer Learning:** Fine-tuned `ResNet152` by freezing base layers and training a custom classification head on the same datasets:
   * `horses_or_humans`: achieved 100% accuracy.
   * `cats_vs_dogs`: achieved 99% accuracy.
* **Evaluation & Comparison:** Analyzed performance using Confusion Matrices, Classification Reports (Accuracy, Precision, Recall, F1-score), and per-class / macro-average Precision-Recall curves. Transfer learning with ResNet152 significantly outperformed the custom network on the more complex `cats_vs_dogs` dataset.
