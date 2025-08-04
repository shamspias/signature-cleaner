import cv2
import numpy as np
from typing import Tuple, Dict


class EnhancedSignatureProcessor:
    """Enhanced signature processing with multiple algorithms and ML-ready structure"""

    def __init__(self):
        # Initialize any ML models here if needed
        self.ml_model = None

    def process_signature(
            self,
            image_data: np.ndarray,
            method: str = "adaptive",  # "adaptive", "otsu", "manual", "ml"
            threshold: int = 180,
            smoothing: float = 1.0,
            padding: int = 20,
            invert: bool = False,
            noise_reduction: bool = True,
            enhance_contrast: bool = True,
            min_signature_area: int = 100
    ) -> Tuple[np.ndarray, Dict]:
        """
        Enhanced signature processing with multiple methods
        """
        # Convert to grayscale if needed
        if len(image_data.shape) == 3:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_data.copy()

        # Store original dimensions
        original_height, original_width = gray.shape

        # Enhance contrast if enabled
        if enhance_contrast:
            gray = self._enhance_contrast(gray)

        # Apply smoothing
        if smoothing > 0:
            kernel_size = int(smoothing * 2 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Apply selected binarization method
        if method == "adaptive":
            binary = self._adaptive_threshold(gray, invert)
        elif method == "otsu":
            binary = self._otsu_threshold(gray, invert)
        elif method == "ml":
            binary = self._ml_based_extraction(gray, invert)
        else:  # manual
            binary = self._manual_threshold(gray, threshold, invert)

        # Enhanced noise reduction
        if noise_reduction:
            binary = self._enhanced_noise_reduction(binary, min_signature_area)

        # Find and validate signature
        result, metadata = self._extract_signature(binary, padding, min_signature_area)

        # Update metadata
        metadata.update({
            "original_size": (original_width, original_height),
            "method_used": method,
            "parameters": {
                "threshold": threshold,
                "smoothing": smoothing,
                "padding": padding,
                "invert": invert,
                "noise_reduction": noise_reduction,
                "enhance_contrast": enhance_contrast
            }
        })

        return result, metadata

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _adaptive_threshold(self, gray: np.ndarray, invert: bool) -> np.ndarray:
        """Adaptive thresholding that works better for varying lighting conditions"""
        # Calculate adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # Invert if needed
        if invert:
            adaptive = cv2.bitwise_not(adaptive)

        return adaptive

    def _otsu_threshold(self, gray: np.ndarray, invert: bool) -> np.ndarray:
        """Otsu's method for automatic threshold selection"""
        # Apply Otsu's thresholding
        threshold_type = cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, threshold_type + cv2.THRESH_OTSU)
        return binary

    def _manual_threshold(self, gray: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
        """Original manual threshold method"""
        if invert:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        return binary

    def _ml_based_extraction(self, gray: np.ndarray, invert: bool) -> np.ndarray:
        """
        Placeholder for ML-based extraction.
        In a real implementation, this would use a trained model.
        """
        # For now, fall back to Otsu with edge detection
        edges = cv2.Canny(gray, 50, 150)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine edges with Otsu result
        combined = cv2.bitwise_or(edges, otsu)

        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        if invert:
            combined = cv2.bitwise_not(combined)

        return combined

    def _enhanced_noise_reduction(self, binary: np.ndarray, min_area: int) -> np.ndarray:
        """Enhanced noise reduction with connected component analysis"""
        # Remove small components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Create mask for significant components
        mask = np.zeros_like(binary)

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                mask[labels == i] = 255

        # Additional morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def _extract_signature(self, binary: np.ndarray, padding: int, min_area: int) -> Tuple[np.ndarray, Dict]:
        """Extract signature with better validation"""
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # No signature found
            result = np.ones_like(binary) * 255
            metadata = {
                "signature_found": False,
                "processed_size": (binary.shape[1], binary.shape[0]),
                "confidence": 0.0
            }
            return result, metadata

        # Find the bounding box of all significant contours
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not valid_contours:
            # No significant signature found
            result = np.ones_like(binary) * 255
            metadata = {
                "signature_found": False,
                "processed_size": (binary.shape[1], binary.shape[0]),
                "confidence": 0.0
            }
            return result, metadata

        # Calculate overall bounding box
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        total_area = 0

        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            total_area += cv2.contourArea(contour)

        # Apply padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(binary.shape[1], x_max + padding)
        y_max = min(binary.shape[0], y_max + padding)

        # Crop
        cropped = binary[y_min:y_max, x_min:x_max]

        # Invert to get black on white
        result = cv2.bitwise_not(cropped)

        # Calculate confidence based on signature characteristics
        confidence = self._calculate_confidence(valid_contours, total_area, binary.shape)

        metadata = {
            "signature_found": True,
            "processed_size": (result.shape[1], result.shape[0]),
            "bounding_box": {
                "x": int(x_min),
                "y": int(y_min),
                "width": int(x_max - x_min),
                "height": int(y_max - y_min)
            },
            "num_components": len(valid_contours),
            "total_signature_area": int(total_area),
            "confidence": float(confidence)
        }

        return result, metadata

    def _calculate_confidence(self, contours: list, total_area: int, image_shape: tuple) -> float:
        """Calculate confidence score for signature detection"""
        if not contours:
            return 0.0

        # Factors for confidence calculation
        image_area = image_shape[0] * image_shape[1]
        area_ratio = total_area / image_area

        # Signature typically occupies 5-40% of image
        if 0.05 <= area_ratio <= 0.4:
            area_score = 1.0
        elif area_ratio < 0.05:
            area_score = area_ratio / 0.05
        else:
            area_score = max(0, 1 - (area_ratio - 0.4) / 0.6)

        # Number of components (signatures usually have 1-10 major components)
        num_components = len(contours)
        if 1 <= num_components <= 10:
            component_score = 1.0
        else:
            component_score = max(0, 1 - (num_components - 10) / 20)

        # Combined confidence
        confidence = (area_score * 0.6 + component_score * 0.4)

        return min(1.0, max(0.0, confidence))

    def analyze_image(self, image_data: np.ndarray) -> Dict:
        """Analyze image to suggest best processing parameters"""
        if len(image_data.shape) == 3:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_data.copy()

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Find peaks in histogram
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((i, hist[i]))

        # Sort peaks by intensity
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Analyze image characteristics
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # Suggest parameters based on analysis
        suggestions = {
            "mean_brightness": float(mean_val),
            "std_deviation": float(std_val),
            "suggested_method": "adaptive" if std_val > 30 else "otsu",
            "suggested_threshold": int(mean_val + std_val / 2),
            "needs_contrast_enhancement": std_val < 20,
            "image_quality": "low" if std_val < 15 else "medium" if std_val < 30 else "high"
        }

        return suggestions


# Utility function for ML model integration (example)
def create_unet_model():
    """
    Example function to create a U-Net model for signature segmentation.
    This would require TensorFlow/PyTorch in a real implementation.
    """
    # Placeholder for actual model creation
    # In production, you would load a pre-trained model here
    pass


# Example of how to integrate with existing SignatureProcessor
class SignatureProcessor(EnhancedSignatureProcessor):
    """Backward compatible wrapper"""

    def process_signature(
            self,
            image_data: np.ndarray,
            threshold: int = 180,
            smoothing: float = 1.0,
            padding: int = 20,
            invert: bool = False,
            noise_reduction: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        # Use enhanced processor with manual method for backward compatibility
        return super().process_signature(
            image_data,
            method="manual",
            threshold=threshold,
            smoothing=smoothing,
            padding=padding,
            invert=invert,
            noise_reduction=noise_reduction,
            enhance_contrast=False,
            min_signature_area=100
        )
