# Food101

Food-101 Image Classification (EfficientNetV2B0, TensorFlow)

A clean, end-to-end transfer-learning pipeline for Food-101 image classification using TensorFlow 2.x and Keras. It loads the dataset from TensorFlow Datasets, builds an EfficientNetV2B0-based classifier, trains in three phases (feature extraction → partial fine-tune → full fine-tune), logs everything to TensorBoard, saves best weights, and supports simple single-image inference with human-readable class names.

### Highlights

Dataset: Food-101 via TensorFlow Datasets (supervised splits for train/validation; integer labels 0–100).

Backbone: EfficientNetV2B0 (ImageNet-pretrained, include_top=False), with global average pooling and a softmax classification head.

Precision & Speed: Mixed precision (mixed_float16) enabled for fast training on modern NVIDIA GPUs.

### Training Strategy:

  Feature Extraction (base frozen)
  Partial Fine-Tuning (continue training with lower LR)
  Full Fine-Tuning (all layers trainable, patience & LR reduction)
  Callbacks: Best-weights checkpointing (monitored on validation accuracy), early stopping, learning-rate reduction on plateau, and TensorBoard logging with timestamped run directories.
  Evaluation: Standard model.evaluate on the validation split.
  Inference: Single-image prediction helper that returns the class name (not just the index).

### Model & Training Configuration

  Input size: 224 × 224 × 3
  
  Preprocessing: Resize to 224; backbone includes internal rescaling appropriate for EfficientNetV2.
  
  Head: GlobalAveragePooling → Dense(num_classes) → Softmax (float32 output for numerical stability under mixed precision).
  
  Loss / Metrics: Sparse Categorical Crossentropy (integer labels 0–100), Accuracy.
  
  Optimizer: Adam (default for feature extraction), then lower learning rate (e.g., 1e-4) for fine-tuning phases.
  
  Batching & Pipeline: tf.data with map/resize, shuffle, batch (32), and prefetch for throughput.
  
  Mixed Precision: Global policy set to mixed_float16 for GPU tensor cores acceleration.
  
  Early Stopping: Patience on validation accuracy to prevent overfitting.
  
  ReduceLROnPlateau: Factor reduction when validation accuracy stalls, with a minimum LR safeguard.

### Logging & Checkpoints

**TensorBoard**: Each run is saved under a timestamped subfolder inside a top-level log directory (e.g., food101_experiments/feature_extraction/…, …/fine_tuning_10/…, …/fine_tuning_full/…).

**Checkpoints**: Best model weights are saved (monitored on val_accuracy) to a designated path (e.g., food101_checkpoints.weights.h5). This ensures you can reload the best-performing state for evaluation or inference.

### Results

Training curves (loss/accuracy) and run-to-run comparisons are available in TensorBoard.

Final performance depends on training duration, fine-tuning depth, and hardware; consult the latest TensorBoard logs in food101_experiments/ for your specific runs.

### Repository Contents (key items)

**Notebook(s)**: End-to-end pipeline: data load (TFDS), preprocessing, model build, staged training, logging, checkpointing, evaluation, and prediction.

**Logs**: Timestamped TensorBoard directories under food101_experiments/.

**Checkpoints**: Best-weights file (e.g., food101_checkpoints.weights.h5) saved during training
