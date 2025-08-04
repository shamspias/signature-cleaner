import numpy as np
import cv2
from typing import Tuple, Optional

# Only import if you have TensorFlow/PyTorch installed
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


class UNetModel:
    """Simple U-Net model for signature segmentation"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.framework = None

        if model_path and HAS_TENSORFLOW:
            self.load_tensorflow_model(model_path)
        elif model_path and HAS_PYTORCH:
            self.load_pytorch_model(model_path)

    def create_tensorflow_unet(self, input_size=(256, 256, 1)):
        """Create a simple U-Net architecture with TensorFlow"""
        if not HAS_TENSORFLOW:
            return None

        inputs = tf.keras.Input(input_size)

        # Encoder
        c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        # Bottleneck
        c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)

        # Decoder
        u4 = layers.UpSampling2D((2, 2))(c3)
        u4 = layers.concatenate([u4, c2])
        c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
        c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c4)

        u5 = layers.UpSampling2D((2, 2))(c4)
        u5 = layers.concatenate([u5, c1])
        c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u5)
        c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c5)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

        model = models.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def load_tensorflow_model(self, model_path: str):
        """Load a pre-trained TensorFlow model"""
        if HAS_TENSORFLOW:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.framework = 'tensorflow'
            except:
                # If loading fails, create a new model
                self.model = self.create_tensorflow_unet()
                self.framework = 'tensorflow'

    def load_pytorch_model(self, model_path: str):
        """Load a pre-trained PyTorch model"""
        if HAS_PYTORCH:
            # Implementation for PyTorch model loading
            pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict signature mask from image"""
        if self.model is None:
            return None

        # Preprocess image
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # Resize to model input size
        original_shape = image.shape[:2]
        resized = cv2.resize(image, (256, 256))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)

        # Predict
        if self.framework == 'tensorflow':
            prediction = self.model.predict(batch)[0]
        else:
            # PyTorch prediction
            prediction = None

        if prediction is not None:
            # Resize back to original size
            mask = cv2.resize(prediction, (original_shape[1], original_shape[0]))

            # Convert to binary
            binary_mask = (mask > 0.5).astype(np.uint8) * 255

            return binary_mask

        return None


class SignatureExtractorML:
    """ML-based signature extractor using various models"""

    def __init__(self):
        self.models = {}
        self.ensemble_weights = {'unet': 0.7, 'edge': 0.3}

    def load_models(self, model_paths: dict):
        """Load multiple models for ensemble prediction"""
        for name, path in model_paths.items():
            if name == 'unet':
                self.models[name] = UNetModel(path)

    def extract_with_edges(self, image: np.ndarray) -> np.ndarray:
        """Extract signature using edge detection and morphology"""
        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(image, 9, 75, 75)

        # Multi-scale edge detection
        edges1 = cv2.Canny(denoised, 30, 100)
        edges2 = cv2.Canny(denoised, 50, 150)
        edges3 = cv2.Canny(denoised, 100, 200)

        # Combine edges
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))

        # Morphological operations to connect components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Fill contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(closed)
        cv2.drawContours(mask, contours, -1, 255, -1)

        return mask

    def extract_with_ml(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Extract signature using ML models with confidence score"""
        results = []

        # Get predictions from each model
        if 'unet' in self.models and self.models['unet'].model is not None:
            unet_pred = self.models['unet'].predict(image)
            if unet_pred is not None:
                results.append(('unet', unet_pred))

        # Add edge-based extraction
        edge_pred = self.extract_with_edges(image)
        results.append(('edge', edge_pred))

        if not results:
            return None, 0.0

        # Ensemble predictions
        if len(results) > 1:
            # Weighted average
            final_mask = np.zeros_like(image, dtype=np.float32)
            total_weight = 0

            for name, pred in results:
                weight = self.ensemble_weights.get(name, 0.5)
                final_mask += pred.astype(np.float32) * weight
                total_weight += weight

            final_mask = final_mask / total_weight
            binary_mask = (final_mask > 127).astype(np.uint8) * 255

            # Calculate confidence based on agreement between models
            agreement = np.mean([np.mean(pred == binary_mask) for _, pred in results])
            confidence = agreement
        else:
            binary_mask = results[0][1]
            confidence = 0.7  # Default confidence for single model

        return binary_mask, confidence

    def post_process(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """Post-process the mask to clean up small artifacts"""
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 255

        # Smooth the result
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        return smoothed


# Integration with EnhancedSignatureProcessor
def integrate_ml_extraction(processor_instance, ml_extractor: SignatureExtractorML):
    """Monkey-patch the processor to use ML extraction"""

    def _ml_based_extraction(self, gray: np.ndarray, invert: bool) -> np.ndarray:
        """ML-based extraction using the ML extractor"""
        # Use the ML extractor
        mask, confidence = ml_extractor.extract_with_ml(gray)

        if mask is None:
            # Fallback to edge-based method
            mask = ml_extractor.extract_with_edges(gray)

        # Post-process
        cleaned = ml_extractor.post_process(mask)

        if invert:
            cleaned = cv2.bitwise_not(cleaned)

        return cleaned

    # Replace the method
    processor_instance._ml_based_extraction = lambda gray, invert: _ml_based_extraction(processor_instance, gray,
                                                                                        invert)


# Example usage:
if __name__ == "__main__":
    # Create ML extractor
    ml_extractor = SignatureExtractorML()

    # Load models (if you have pre-trained models)
    # ml_extractor.load_models({'unet': 'models/signature_unet.h5'})

    # Test with an image
    test_image = cv2.imread('test_signature.jpg', cv2.IMREAD_GRAYSCALE)
    if test_image is not None:
        mask, confidence = ml_extractor.extract_with_ml(test_image)
        print(f"Extraction confidence: {confidence:.2%}")

        # Display results
        cv2.imshow('Original', test_image)
        cv2.imshow('Extracted', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
