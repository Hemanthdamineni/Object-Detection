"""
MNIST Object Detection - Complete Implementation with Comprehensive Reporting

This script implements a complete multi-task learning model for MNIST object detection 
with comprehensive reporting outputs suitable for academic/professional reports.

Features:
- Multi-task learning (classification + bounding box regression)
- MNIST dataset with random placement for object detection
- CNN-based feature extraction
- Comprehensive evaluation metrics and visualizations
- Report-ready outputs and saved plots

Report Sections Covered:
- Methodology: Model architecture summary and training process
- Results: Training curves, evaluation metrics, IoU analysis, predictions
- Visualizations: All plots saved as high-quality images for report inclusion
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import warnings
from datetime import datetime
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set environment variables - suppress only specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Create output directory with error handling
try:
    os.makedirs('report_outputs', exist_ok=True)
except PermissionError:
    print("Error: Cannot create report_outputs directory. Check permissions.")
    sys.exit(1)
except Exception as e:
    print(f"Error creating output directory: {e}")
    sys.exit(1)

print("Libraries imported and environment configured")
print(f"TensorFlow version: {tf.__version__}")
print(f"Report outputs will be saved to: ./report_outputs/")

# Global parameters - standardized
IMG_WIDTH = 75
IMG_HEIGHT = 75
BATCH_SIZE = 64
EPOCHS = 10  # Standardized with notebook
use_normalized_coordinates = True

# Set up TensorFlow strategy
strategy = tf.distribute.get_strategy()
print(f"Number of replicas: {strategy.num_replicas_in_sync}")
print(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training epochs: {EPOCHS}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def read_image_from_tensorflow_dataset(image, label):
    """Preprocess image and label from TensorFlow dataset."""
    # Reshape image to 28x28
    image = tf.reshape(image, [28, 28])
    
    # Random placement on 75x75 canvas
    y_minimum = tf.random.uniform([], 0, 48, dtype=tf.int32)
    x_minimum = tf.random.uniform([], 0, 48, dtype=tf.int32)
    
    # Pad image to 75x75
    image = tf.image.pad_to_bounding_box(
        tf.expand_dims(image, -1), y_minimum, x_minimum, IMG_HEIGHT, IMG_WIDTH
    )
    
    # Normalize image
    image = tf.cast(image, tf.float32) / 255.0
    
    # Calculate bounding box coordinates with validation
    y_min = tf.cast(y_minimum, tf.float32) / IMG_HEIGHT
    x_min = tf.cast(x_minimum, tf.float32) / IMG_WIDTH
    y_max = tf.cast(y_minimum + 28, tf.float32) / IMG_HEIGHT
    x_max = tf.cast(x_minimum + 28, tf.float32) / IMG_WIDTH
    
    # Ensure coordinates are within [0,1] range
    y_min = tf.clip_by_value(y_min, 0.0, 1.0)
    x_min = tf.clip_by_value(x_min, 0.0, 1.0)
    y_max = tf.clip_by_value(y_max, 0.0, 1.0)
    x_max = tf.clip_by_value(x_max, 0.0, 1.0)
    
    return image, {
        'classification': label,
        'bounding_box_regression_output': tf.stack([y_min, x_min, y_max, x_max])
    }


def get_training_dataset():
    """Get training dataset with error handling."""
    try:
        dataset = tfds.load('mnist', as_supervised=True, try_gcs=True)
        dataset = dataset['train']
        dataset = dataset.map(read_image_from_tensorflow_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(5000)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        raise


def get_validation_dataset():
    """Get validation dataset with error handling."""
    try:
        dataset = tfds.load('mnist', as_supervised=True, try_gcs=True)
        dataset = dataset['test']
        dataset = dataset.map(read_image_from_tensorflow_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        raise


def dataset_to_numpy_util(training_data_set, validation_data_set, num_values):
    """Convert TensorFlow dataset to NumPy arrays for visualization with memory management."""
    try:
        training_data_set = training_data_set.unbatch().batch(num_values)
        validation_data_set = validation_data_set.unbatch().batch(num_values)
        
        for training_data_batch, validation_data_batch in zip(training_data_set, validation_data_set):
            training_images, training_outputs = training_data_batch
            validation_images, validation_outputs = validation_data_batch
            
            training_images = training_images.numpy()
            training_labels = training_outputs['classification'].numpy()
            training_bounding_boxes = training_outputs['bounding_box_regression_output'].numpy()
            
            validation_images = validation_images.numpy()
            validation_labels = validation_outputs['classification'].numpy()
            validation_bounding_boxes = validation_outputs['bounding_box_regression_output'].numpy()
            
            return (training_images, training_labels, training_bounding_boxes,
                    validation_images, validation_labels, validation_bounding_boxes)
    except Exception as e:
        print(f"Error converting dataset to numpy: {e}")
        raise


def intersection_over_union(predicted_box, true_box):
    """
    Calculate Intersection over Union (IoU) with proper validation.
    
    Coordinate format: [y_min, x_min, y_max, x_max] (normalized to [0,1])
    - y_min, y_max: vertical coordinates (top to bottom)
    - x_min, x_max: horizontal coordinates (left to right)
    """
    try:
        # Unpack coordinates in the same order as tf.stack([y_min, x_min, y_max, x_max])
        y_min_pred, x_min_pred, y_max_pred, x_max_pred = predicted_box
        y_min_true, x_min_true, y_max_true, x_max_true = true_box
        
        # Validate bounding box coordinates (ensure min < max)
        if (y_max_pred <= y_min_pred or x_max_pred <= x_min_pred or 
            y_max_true <= y_min_true or x_max_true <= x_min_true):
            return 0.0
        
        # Ensure coordinates are within [0,1] range and maintain min < max relationship
        y_min_pred = max(0.0, min(1.0, y_min_pred))
        x_min_pred = max(0.0, min(1.0, x_min_pred))
        y_max_pred = max(y_min_pred, min(1.0, y_max_pred))  # Ensure y_max >= y_min
        x_max_pred = max(x_min_pred, min(1.0, x_max_pred))  # Ensure x_max >= x_min
        
        y_min_true = max(0.0, min(1.0, y_min_true))
        x_min_true = max(0.0, min(1.0, x_min_true))
        y_max_true = max(y_min_true, min(1.0, y_max_true))  # Ensure y_max >= y_min
        x_max_true = max(x_min_true, min(1.0, x_max_true))  # Ensure x_max >= x_min
        
        # Calculate intersection coordinates
        y_min_intersection = max(y_min_pred, y_min_true)
        x_min_intersection = max(x_min_pred, x_min_true)
        y_max_intersection = min(y_max_pred, y_max_true)
        x_max_intersection = min(x_max_pred, x_max_true)
        
        # Calculate intersection area (will be 0 if no overlap)
        intersection_height = max(0.0, y_max_intersection - y_min_intersection)
        intersection_width = max(0.0, x_max_intersection - x_min_intersection)
        intersection_area = intersection_height * intersection_width
        
        # Calculate individual box areas
        predicted_area = (y_max_pred - y_min_pred) * (x_max_pred - x_min_pred)
        true_area = (y_max_true - y_min_true) * (x_max_true - x_min_true)
        
        # Calculate union area
        union_area = predicted_area + true_area - intersection_area
        
        # Calculate IoU with epsilon to prevent division by zero
        if union_area <= 1e-6:
            return 0.0
        
        iou = intersection_area / union_area
        return max(0.0, min(1.0, iou))  # Ensure IoU is in [0,1] range
        
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0


def validate_bounding_box(box, box_name="box"):
    """Validate and debug bounding box coordinates."""
    y_min, x_min, y_max, x_max = box
    
    # Check coordinate ranges
    if not (0 <= y_min <= 1 and 0 <= x_min <= 1 and 0 <= y_max <= 1 and 0 <= x_max <= 1):
        print(f"Warning: {box_name} coordinates out of [0,1] range: {box}")
    
    # Check min < max relationship
    if y_max <= y_min or x_max <= x_min:
        print(f"Warning: {box_name} has invalid min/max relationship: {box}")
        return False
    
    return True


def calculate_iou_batch(predicted_boxes, true_boxes):
    """Calculate IoU for a batch of predictions with error handling and validation."""
    try:
        ious = []
        for i, (pred_box, true_box) in enumerate(zip(predicted_boxes, true_boxes)):
            validate_bounding_box(pred_box, f"predicted_box_{i}")
            validate_bounding_box(true_box, f"true_box_{i}")
            
            iou = intersection_over_union(pred_box, true_box)
            ious.append(iou)
        return np.array(ious)
    except Exception as e:
        print(f"Error calculating batch IoU: {e}")
        return np.array([0.0] * len(predicted_boxes))


print("Helper functions defined")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================


def create_model():
    """Create the complete model with both classification and regression outputs - standardized architecture."""
    try:
        inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='input_image')
        
        # Feature extractor - Convolutional layers (standardized with notebook)
        x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same', name='conv1')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool1')(x)
        
        x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', name='conv2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool2')(x)
        
        x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv3')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool3')(x)
        
        # Dense layers for feature processing
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_features')(x)
        x = tf.keras.layers.Dropout(0.3, name='dropout')(x)
        
        # Output heads
        classification_output = tf.keras.layers.Dense(10, activation='softmax', name='classification')(x)
        bounding_box_output = tf.keras.layers.Dense(4, name='bounding_box_regression_output')(x)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[classification_output, bounding_box_output],
            name='mnist_object_detection_model'
        )
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        raise


def compile_model(model):
    """Compile the model with appropriate loss functions and metrics - standardized."""
    try:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'classification': tf.keras.losses.SparseCategoricalCrossentropy(),
                'bounding_box_regression_output': tf.keras.losses.MeanSquaredError()
            },
            metrics={
                'classification': ['accuracy'],
                'bounding_box_regression_output': ['mse']
            },
            loss_weights={
                'classification': 1.0,
                'bounding_box_regression_output': 1.0
            }
        )
        return model
    except Exception as e:
        print(f"Error compiling model: {e}")
        raise


print("Model architecture defined")

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================


def train_model(model, training_dataset, validation_dataset):
    """Train the model and return training history with error handling."""
    try:
        # Training parameters
        steps_per_epoch = 60000 // BATCH_SIZE
        validation_steps = 10000 // BATCH_SIZE
        
        print(f"\nStarting training for {EPOCHS} epochs...")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train the model
        history = model.fit(
            training_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            verbose=1
        )
        
        return history
    except Exception as e:
        print(f"Error during training: {e}")
        raise


def evaluate_model(model, validation_dataset):
    """Evaluate the model and return metrics with error handling."""
    try:
        validation_steps = 10000 // BATCH_SIZE
        
        print("\nEvaluating model on validation set...")
        evaluation = model.evaluate(validation_dataset, steps=validation_steps, verbose=0)
        
        # Extract metrics
        test_loss = evaluation[0]
        classification_loss = evaluation[1]
        bbox_loss = evaluation[2]
        classification_accuracy = evaluation[3]
        bbox_mse = evaluation[4]
        
        return {
            'test_loss': test_loss,
            'classification_loss': classification_loss,
            'bbox_loss': bbox_loss,
            'classification_accuracy': classification_accuracy,
            'bbox_mse': bbox_mse
        }
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


print("Training and evaluation functions defined")

# ============================================================================
# REPORTING AND VISUALIZATION FUNCTIONS
# ============================================================================


def save_plot(filename, dpi=300, bbox_inches='tight'):
    """Save current plot with high quality for reports with error handling."""
    try:
        filepath = os.path.join('report_outputs', filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
        print(f"Plot saved: {filepath}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")


def plot_training_curves(history):
    """Plot and save training/validation curves with error handling."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training and Validation Curves', fontsize=16, fontweight='bold')
        
        # Total Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Classification Accuracy
        axes[0, 1].plot(history.history['classification_accuracy'], label='Training Accuracy', linewidth=2, color='green')
        axes[0, 1].plot(history.history['val_classification_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
        axes[0, 1].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Classification Loss
        axes[1, 0].plot(history.history['classification_loss'], label='Training Classification Loss', linewidth=2, color='purple')
        axes[1, 0].plot(history.history['val_classification_loss'], label='Validation Classification Loss', linewidth=2, color='brown')
        axes[1, 0].set_title('Classification Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bounding Box Loss
        axes[1, 1].plot(history.history['bounding_box_regression_output_loss'], label='Training BB Loss', linewidth=2, color='teal')
        axes[1, 1].plot(history.history['val_bounding_box_regression_output_loss'], label='Validation BB Loss', linewidth=2, color='pink')
        axes[1, 1].set_title('Bounding Box Regression Loss', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot('training_curves.png')
        plt.show()
    except Exception as e:
        print(f"Error plotting training curves: {e}")


def plot_iou_distribution(ious):
    """Plot IoU distribution and box plot with error handling."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Intersection over Union (IoU) Analysis', fontsize=16, fontweight='bold')
        
        # IoU Histogram
        axes[0].hist(ious, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(ious), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ious):.3f}')
        axes[0].axvline(np.median(ious), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(ious):.3f}')
        axes[0].set_title('IoU Score Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('IoU Score')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # IoU Box Plot
        axes[1].boxplot(ious, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_title('IoU Box Plot', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('IoU Score')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot('iou_analysis.png')
        plt.show()
    except Exception as e:
        print(f"Error plotting IoU distribution: {e}")


def plot_sample_predictions(images, true_labels, true_boxes, pred_labels, pred_boxes, ious, num_samples=20):
    """Plot sample predictions with bounding boxes with error handling."""
    try:
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle('Sample Predictions: True vs Predicted Bounding Boxes', fontsize=16, fontweight='bold')
        
        for i in range(min(num_samples, len(images))):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            # Display image
            img = images[i].squeeze()
            ax.imshow(img, cmap='gray')
            
            # Draw true bounding box (green)
            true_box = true_boxes[i]
            y_min, x_min, y_max, x_max = true_box
            true_rect = plt.Rectangle((x_min * IMG_WIDTH, y_min * IMG_HEIGHT),
                                     (x_max - x_min) * IMG_WIDTH, (y_max - y_min) * IMG_HEIGHT,
                                     linewidth=2, edgecolor='green', facecolor='none', label='True')
            ax.add_patch(true_rect)
            
            # Draw predicted bounding box (red)
            pred_box = pred_boxes[i]
            y_min, x_min, y_max, x_max = pred_box
            pred_rect = plt.Rectangle((x_min * IMG_WIDTH, y_min * IMG_HEIGHT),
                                     (x_max - x_min) * IMG_WIDTH, (y_max - y_min) * IMG_HEIGHT,
                                     linewidth=2, edgecolor='red', facecolor='none', label='Predicted')
            ax.add_patch(pred_rect)
            
            title = f'True: {true_labels[i]}, Pred: {pred_labels[i]}\nIoU: {ious[i]:.3f}'
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')

            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        save_plot('sample_predictions.png')
        plt.show()
    except Exception as e:
        print(f"Error plotting sample predictions: {e}")


def save_metrics_to_json(classification_acc, bbox_mse, total_loss, ious, history):
    """Save all metrics to JSON for easy access with proper type conversion."""
    try:
        # Convert numpy types to Python types for JSON serialization
        metrics = {
            'model_config': {
                'image_width': int(IMG_WIDTH),
                'image_height': int(IMG_HEIGHT),
                'batch_size': int(BATCH_SIZE),
                'epochs': int(EPOCHS)
            },
            'final_metrics': {
                'classification_accuracy': float(classification_acc),
                'bbox_mse': float(bbox_mse),
                'bbox_rmse': float(np.sqrt(bbox_mse)),
                'total_loss': float(total_loss)
            },
            'iou_stats': {
                'mean': float(np.mean(ious)),
                'median': float(np.median(ious)),
                'std': float(np.std(ious)),
                'min': float(np.min(ious)),
                'max': float(np.max(ious)),
                'percentile_25': float(np.percentile(ious, 25)),
                'percentile_75': float(np.percentile(ious, 75))
            },
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'classification_accuracy': [float(x) for x in history.history['classification_accuracy']],
                'val_classification_accuracy': [float(x) for x in history.history['val_classification_accuracy']]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = os.path.join('report_outputs', 'evaluation_metrics.json')
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")
    except Exception as e:
        print(f"Error saving metrics: {e}")


def print_evaluation_report(classification_acc, bbox_mse, total_loss, ious):
    """Print comprehensive evaluation report."""
    print("FINAL EVALUATION RESULTS - READY FOR REPORT")
    
    print("\nOVERALL MODEL PERFORMANCE:")
    print(f"   Total Loss: {total_loss:.6f}")
    
    print("\nCLASSIFICATION METRICS:")
    print(f"   Classification Accuracy: {classification_acc:.4f} ({classification_acc*100:.2f}%)")
    
    print("\nBOUNDING BOX REGRESSION METRICS:")
    print(f"   Mean Squared Error (MSE): {bbox_mse:.6f}")
    print(f"   Root Mean Squared Error (RMSE): {np.sqrt(bbox_mse):.6f}")
    
    print("\nINTERSECTION OVER UNION (IoU) ANALYSIS:")
    print(f"   Mean IoU: {np.mean(ious):.4f}")
    print(f"   Median IoU: {np.median(ious):.4f}")
    print(f"   Standard Deviation: {np.std(ious):.4f}")
    print(f"   Minimum IoU: {np.min(ious):.4f}")
    print(f"   Maximum IoU: {np.max(ious):.4f}")
    print(f"   25th Percentile: {np.percentile(ious, 25):.4f}")
    print(f"   75th Percentile: {np.percentile(ious, 75):.4f}")
    
    print("\nIoU QUALITY DISTRIBUTION:")
    # IoU quality categories
    poor_iou = np.sum(ious < 0.4) / len(ious) * 100
    fair_iou = np.sum((ious >= 0.4) & (ious < 0.6)) / len(ious) * 100
    good_iou = np.sum((ious >= 0.6) & (ious < 0.8)) / len(ious) * 100
    excellent_iou = np.sum(ious >= 0.8) / len(ious) * 100
    
    print(f"   Poor (IoU < 0.4): {poor_iou:.1f}%")
    print(f"   Fair (0.4 ≤ IoU < 0.6): {fair_iou:.1f}%")
    print(f"   Good (0.6 ≤ IoU < 0.8): {good_iou:.1f}%")
    print(f"   Excellent (IoU ≥ 0.8): {excellent_iou:.1f}%")


def print_model_summary_report(model):
    """Print formatted model summary for reports."""
    print("MODEL ARCHITECTURE SUMMARY FOR REPORT")

    model.summary()
    
    # Parameter count
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print("PARAMETER SUMMARY")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")


print("Reporting and visualization functions defined")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================


def main():
    """Main function to run the complete MNIST object detection pipeline with comprehensive error handling."""
    try:
        print("MNIST OBJECT DETECTION - FIXED IMPLEMENTATION")
        
        # 1. Load datasets
        print("\nLoading MNIST datasets...")
        training_dataset = get_training_dataset()
        validation_dataset = get_validation_dataset()
        print("Datasets loaded successfully")
        
        # 2. Create and compile model
        print("\nCreating and compiling model...")
        model = create_model()
        model = compile_model(model)
        print("Model created and compiled")
        
        # 3. Print model summary
        print_model_summary_report(model)
        
        # 4. Train model
        history = train_model(model, training_dataset, validation_dataset)
        print("Training completed")
        
        # 5. Evaluate model
        metrics = evaluate_model(model, validation_dataset)
        print("Evaluation completed")
        
        # 6. Generate predictions for analysis
        print("\nGenerating predictions for analysis...")
        sample_data = dataset_to_numpy_util(training_dataset, validation_dataset, 100)
        val_images, val_labels, val_boxes = sample_data[3], sample_data[4], sample_data[5]
        
        predictions = model.predict(val_images)
        pred_labels = np.argmax(predictions[0], axis=1)
        pred_boxes = predictions[1]
        
        # Calculate IoU scores
        ious = calculate_iou_batch(pred_boxes, val_boxes)
        print("Predictions and IoU analysis completed")
        
        # 7. Generate all visualizations
        print("\nGenerating comprehensive visualizations...")
        plot_training_curves(history)
        plot_iou_distribution(ious)
        plot_sample_predictions(val_images, val_labels, val_boxes, pred_labels, pred_boxes, ious)
        print("All visualizations generated and saved")
        
        # 8. Save metrics and generate report
        print("\nSaving metrics and generating final report...")
        save_metrics_to_json(metrics['classification_accuracy'], metrics['bbox_mse'],
                             metrics['test_loss'], ious, history)
        
        print_evaluation_report(metrics['classification_accuracy'], metrics['bbox_mse'],
                               metrics['test_loss'], ious)
        
        print("\nMNIST Object Detection pipeline completed successfully!")
        print("Check the './report_outputs/' directory for all generated files")
        print("All outputs are ready for inclusion in your academic report")
        
        # Cleanup GPU memory
        tf.keras.backend.clear_session()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"\nError in main pipeline: {e}")
        tf.keras.backend.clear_session()
        raise


if __name__ == "__main__":
    main()