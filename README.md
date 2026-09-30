# Machine Learning and Artificial Intelligence — Labs

This repository contains university laboratory works for the **Machine Learning and Artificial Intelligence** course.

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