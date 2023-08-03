
# BERT and LoRA Fine-Tuning from Scratch: Fast Inference with TensorRT

Explore the end-to-end process of building BERT and LoRA models from scratch, fine-tuning them on a specific task, and optimizing the resulting models for fast GPU inference using NVIDIA's TensorRT. 

## Motivation
This project, while serving as a comprehensive guide to understanding, implementing, and optimizing transformer-based models for improved performance and efficiency, is also a representation of my learning journey and serves as a portfolio piece showcasing some skills I have picked up along the way in machine learning and natural language processing.

## Technologies

This project makes use of the following technologies and tools:

* **Python**: The programming language used for implementing the project.
* **NumPy**: A library used for efficient multi-dimensional data operations where PyTorch tensors aren't suitable.
* **Pandas**: A library used for cleaning, transforming, and exploring the data prior to model fine-tuning.
* **PyTorch**: A library used to build the BERT and LoRA models from scratch and for fine-tuning.
* **Hugging Face Transformers**: A library used to access pretrained models and weights. It was predominantly employed to load models and conduct further training in the optimization section of this project.
* **NVIDIA TensorRT**: A high-performance deep learning inference optimizer and runtime library used to optimize models for fast inference.
* **Transformer-Deploy**: A library used to simplify the process of applying TensorRT inference optimizations to the models.
* **ONNX**: An open standard for representing machine learning models used for converting PyTorch models into a format that can be optimized by TensorRT.
* **Docker**: Deployed to ensure a consistent and replicable environment, streamlining the installation of NVIDIA dependencies such as CUDA and cuBLAS.

## Table of contents
* [Part 1: Building BERT from Scratch](#building-bert-from-scratch)
* [Part 2: Building LoRA From Scratch](#building-lora-from-scratch)
* [Part 3: Fine-Tuning BERT with LoRA for Stack Overflow Question Classification](#fine-tuning-bert-with-lora-for-stack-overflow-question-classification)
    * [Setup and Requirements](#setup-and-requirements)
    * [Dataset](#dataset)
    * [Models](#models)
    * [Process](#process)
    * [Results](#results)
    * [Conclusions](#conclusions)
* [Part 4: Optimizing BERT for Fast Inference with NVIDIA TensorRT](#optimizing-bert-for-fast-inference-with-nvidia-tensorrt)
    * [Setup and Requirements](#setup-and-requirements-1)
    * [Model Selection](#model-selection)
    * [INT8 Quantization Process](#int8-quantization-process)
    * [Comparison of Frameworks](#comparison-of-frameworks)
    * [Results](#results-1)
    * [Conclusions](#conclusions-1)
## Building BERT from Scratch
The `bert_from_scratch.py` script is used to build a BERT model for Sequence Classification from scratch. It implements the architecture and mechanics of the BERT model, creating a ready-to-train model that can be fine-tuned on specific tasks. Notably, the model architecture and module/parameter names mirror the Hugging Face implementation, ensuring that weights between the two are compatible. This script includes a `from_pretrained` class method, working in the same way as in Hugging Face's model.

For an in-depth explanation of the theory behind this model, as well as an understanding of the code, refer to the accompanying [Medium article](https://medium.com/@alexmriggio/bert-for-sequence-classification-from-scratch-code-and-theory-fb88053800fa).
## Building LoRA From Scratch
The `lora_from_scratch.py` script contains a custom implementation of the LoRA (Low-Rank Adaptation) method. LoRA is a technique that aims to maintain a high level of accuracy while reducing the computational cost and complexity associated with fine-tuning large models. It does this by appending low-rank adaptation matrices to specific parameters of the model, while the remaining parameters are "frozen" or kept constant. As a result, the total quantity of trainable parameters is greatly reduced. 

For a comprehensive understanding of the theory and implementation details behind LoRA, please refer to the accompanying [Medium article](https://medium.com/@alexmriggio/lora-low-rank-adaptation-from-scratch-code-and-theory-f31509106650).


## Fine-Tuning BERT with LoRA for Stack Overflow Question Classification

The `fine_tuning.ipynb` notebook involves fine-tuning the custom implementations of BERT and LoRA on the task of classifying whether questions submitted to Stack Overflow will be closed or not by the moderators.

### Setup and Requirements

#### Dependencies
To install the necessary dependencies for the `fine_tuning.ipynb` notebook, run the following command:

`pip install -r notebooks/requirements/requirements_fine_tuning.txt`



#### Data Directory

The `data/` directory is intially empty.

To replicate the results of the `fine-tuning.ipynb` notebook, you will need to [download the dataset](https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data) and place the data file in the `data/` directory. 

The expected data file is:
*  `train-sample.csv`.

#### Models Directory

The `models/` directory is initially empty. After running the fine-tuning process, this folder will contain the trained model files.


### Dataset

The dataset used for fine-tuning the models is a balanced collection of 140,272 questions submitted to Stack Overflow between 2008 and 2012. The train, validation, and test sets follow an 80/10/10 split.

#### Target Labels
* Open: These are questions that remained open on Stack Overflow.
* Closed: These are questions that were closed by moderators.

#### Text Features
* Question Text: This is a concatenation of the title and body of the question.

#### Tabular Features
* Account Age: The number of days between account creation and post creation.
* Number of Tags: The number of tags associated with each post.
* Reputation: The reputation of the account that posted the question.
* Title Length: The number of words in the title.
* Body Length: The number of words in the body of the question.

### Models

Six variants of BERT:

* **BERT-base**: BERT base uncased
* **BERT Overflow**: BERT-base pretrained on 152 million sentences from Stack Overflow
* **BERT Tabular**: BERT-base modified to accept tabular features
* **LoRA BERT - Rank 1**: BERT-base adapted with a low-rank approximation of rank 1
* **LoRA BERT - Rank 8**: BERT-base adapted with a low-rank approximation of rank 8
* **LoRA BERT - Rank 8 (last four layers)**: BERT-base adapted with a low-rank approximation of rank 8, applied only on the last four layers

### Process

Each model is initialized with pretrained weights from the Hugging Face Hub, then fine-tuned with the training and validation set using the training loop in the `BertTrainer` class. The models are evaluated on the test set using weights from the epoch with the lowest validation loss.

### Results

Performance of the models after fine-tuning:

| Model                | Loss   | Accuracy | Train Params | Relative Params | Train Time* | Epoch** |
| -------------------- | ------ | -------- | ------------ | --------------- | ---------- | ----- |
| BERT-base            | 0.4594 | 78.57%   | 109,483,778  | 100.00%         | 48         | 1     |
| BERT-overflow        | 0.5061 | 76.17%   | 149,018,882  | 136.11%         | 47         | 3     |
| BERT-base-tabular    | 0.4558 | 78.87%   | 109,485,326  | 100.00%         | 48         | 1     |
| BERT-base-lora-r8    | 0.4643 | 78.31%   | 296,450      | 0.27%           | 38         | 1     |
| BERT-base-lora-r1    | 0.4648 | 78.09%   | 38,402       | 0.04%           | 38         | 2     |
| BERT-base-lora-r8-last-4 | 0.4713 | 78.12%   | 99,842       | 0.09%           | 25         | 4     |

\* Train Time = Time in minutes for a single epoch to complete on an NVIDIA A6000 GPU.

** Epoch = The epoch in which the model had the lowest validation loss. Each model was fine-tuned for five epochs numbered 0 to 4.

### Conclusions

* The tabular data provided a slight increase in performance but may be due to the stochastic nature of the training process. The small performance boost likely does not justify the additional complexity.

* The relatively underwhelming performance of the BERT Overflow model was surprising, given its pretraining on text assumed to closely resemble the fine-tuning dataset.

* The LoRA models experienced hardly any drop in accuracy or increase in validation loss. Highlights the redundancy in the change-in-weight matrices and why LoRA has become so ubiquitous for training LLMs. 

* LoRA applied to the last four layers resulted in the shortest training time per epoch, potentially making it the most efficient model in terms of computational resources.

* The five epochs of training time was overkill for most models. However, the model with LoRA applied to the last four layers did not fully converge, probably can increase learning rate.
## Optimizing BERT for Fast Inference with NVIDIA TensorRT

The `tensorrt_optimization.ipynb` notebook covers the process of applying optimizations, including INT8 Quantization, to a fine-tuned BERT model for accelerating inference times with NVIDIA TensorRT, facilitated with the Transformer-deploy library.

Modified from the [transformer-deploy end-to-end walkthrough](https://github.com/ELS-RD/transformer-deploy/blob/main/demo/quantization/quantization_end_to_end.ipynb).

### Setup and Requirements

#### System Requirements

To run the `tensorrt_optimization` notebook, your system should have the following prerequisites:

* Nvidia CUDA 11.X
* TensorRT 8.2.1
* cuBLAS

#### Docker Image

It's recommended to use Docker to ensure that all dependencies are correctly installed, and the code runs as expected. You can use the following Docker image:

`docker pull ghcr.io/els-rd/transformer-deploy:0.6.0`

#### Transformer-Deploy Library

Additionally, the `tensorrt_optimization.ipynb` notebook depends on the `transformer-deploy` library. This library simplifies the process of applying TensorRT inference optimizations and is hosted on a separate GitHub repository. You can install it using the following command:

`pip install git+https://github.com/ELS-RD/transformer-deploy.git`

### Model Selection

I focus on optimizing the fine-tuned BERT-base model. This choice represents a balance between performance and complexity: the BERT-base model has the second-best performance of all the [models fine-tuned](#models), and it achieves this performance without the added complexity of incorporating tabular data.

### INT8 Quantization Process

The quantization process begins by automatically inserting Quantize (Q) and Dequantize (DQ) nodes into the computational graph of the model. The impact of quantizing individual components of the model on accuracy is then evaluated. Quantization is disabled for those components that substantially decrease accuracy.

Once this initial process is complete, the model is fine-tuned further using the training and validation set to recover some of the lost accuracy. After fine-tuning, the model is converted to an ONNX computational graph and subsequently into the TensorRT execution engine.

Finally, after being converted into the TensorRT execution engine, the model undergoes a timing and evaluation process on the test set. This allows for measurement of inference speed and assessment of accuracy of the optimized model.

### Comparison of Frameworks

I compare the model selected across three frameworks (PyTorch, ONNX and TensorRT) at varying precision levels. The goal is to benchmark each model's speed, which is measured in terms of the number of samples processed per second during inference, and the recovery of accuracy, which is measured by comparing the optimized model's accuracy to that of the original model.

### Results

Model accuracy and speed under inference of the test set with batch size of 32:

| Framework | Precision | Samples/s | Speedup | Accuracy | Recovery | Hardware |
| --------- | --------- | --------- | ------- | -------- | -------- | -------- |
| Pytorch   | FP32      | 125       | 1.00x   | 78.57%   | 100.00%  | GPU      |
| PyTorch   | FP16      | 370       | 2.96x   | 78.57%   | 100.00%  | GPU      |
| ONNX      | FP16      | 408       | 3.26x   | 78.57%   | 100.00%  | GPU      |
| TensorRT  | FP16      | 511       | 4.09x   | 78.57%   | 100.00%  | GPU      |
| TensorRT  | INT8      | 543       | 4.34x   | 78.16%   | 99.47%   | GPU      |


### Conclusions

* NVIDIA TensorRT offers substantial performance enhancements for rapid inference specifically tailored to their GPUs. The fact that the solution they've developed for their hardware achieves the fastest results is not unexpected.

* NVIDIA TensorRT stands out as the sole framework offering INT8 quantization for GPU inference. The recovery performance is notably impressive, with 99.5% of the original accuracy. The marginal loss in accuracy is likely to be inconsequential for most applications.

* The speed improvement of TensorRT INT8 over TensorRT FP16 is approximately 6%. This could potentially be due to many layers reverting to FP16 computation.

* ONNX Runtime emerges as a viable alternative for accelerated GPU inference. Although it may not surpass TensorRT in terms of speed, it does provide an option for CPU inference. For GPU inference, TensorRT would probably be the preferred choice for most applications given its superior speed and high recovery.
