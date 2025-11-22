import cv2
import numpy as np
import os
from pathlib import Path
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')

class MissingPersonDetector:
    """
    Face recognition system to detect a missing person in public images.
    Uses DeepFace with VGG-Face model for high accuracy.
    """
    
    def __init__(self, reference_folder, public_folder, output_folder, threshold=65):
        """
        Initialize the detector.
        
        Args:
            reference_folder: Path to folder containing reference images of person X
            public_folder: Path to folder containing public images to search
            output_folder: Path to save annotated images
            threshold: Similarity threshold (0-100), default 65
        """
        self.reference_folder = Path(reference_folder)
        self.public_folder = Path(public_folder)
        self.output_folder = Path(output_folder)
        self.threshold = threshold
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Model configuration - VGG-Face is one of the best performing models
        self.model_name = "VGG-Face"
        self.detector_backend = "retinaface"  # Best face detector
        
        # Store reference image paths
        self.reference_images = []
        self.load_reference_images()
    
    def load_reference_images(self):
        """Load all reference images of person X."""
        print("Loading reference images...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        reference_files = [f for f in self.reference_folder.iterdir() 
                          if f.suffix.lower() in image_extensions]
        
        if not reference_files:
            raise ValueError(f"No images found in {self.reference_folder}")
        
        for img_path in reference_files:
            print(f"  ✓ Loaded: {img_path.name}")
            self.reference_images.append(str(img_path))
        
        print(f"\nLoaded {len(self.reference_images)} reference images\n")
    
    def detect_faces(self, img_path):
        """
        Detect all faces in an image using RetinaFace.
        
        Args:
            img_path: Path to image file
            
        Returns:
            List of face dictionaries with coordinates
        """
        try:
            # Use DeepFace's extract_faces which uses RetinaFace
            faces = DeepFace.extract_faces(
                img_path=str(img_path),
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            return faces
        except Exception as e:
            print(f"  Error detecting faces: {e}")
            return []
    
    def calculate_similarity(self, face1_path, face2_path):
        """
        Calculate similarity between two faces.
        
        Args:
            face1_path: Path to first face image
            face2_path: Path to second face image
            
        Returns:
            Similarity percentage (0-100)
        """
        try:
            result = DeepFace.verify(
                img1_path=face1_path,
                img2_path=face2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            # DeepFace returns distance (lower is more similar)
            # Convert to similarity percentage
            distance = result['distance']
            
            # VGG-Face threshold is around 0.4, normalize to 0-100 scale
            # Lower distance = higher similarity
            if self.model_name == "VGG-Face":
                # Distance typically ranges from 0 to 1+
                # 0.0-0.4 is considered a match
                similarity = max(0, min(100, (1 - distance / 0.8) * 100))
            else:
                similarity = max(0, min(100, (1 - distance) * 100))
            
            return similarity
            
        except Exception as e:
            return 0
    
    def compare_with_references(self, face_image_path):
        """
        Compare a detected face with all reference images.
        
        Args:
            face_image_path: Path to the detected face image
            
        Returns:
            Maximum similarity percentage
        """
        max_similarity = 0
        
        for ref_img in self.reference_images:
            similarity = self.calculate_similarity(ref_img, face_image_path)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def process_public_images(self):
        """Process all public images and detect faces."""
        print("Processing public images...")
        print(f"Using model: {self.model_name}")
        print(f"Face detector: {self.detector_backend}\n")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        public_files = [f for f in self.public_folder.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        if not public_files:
            print(f"No images found in {self.public_folder}")
            return
        
        processed_count = 0
        matched_count = 0
        
        for img_path in public_files:
            print(f"\n[{processed_count + 1}/{len(public_files)}] {img_path.name}")
            
            result = self.process_single_image(img_path)
            
            if result:
                processed_count += 1
                if result['has_match']:
                    matched_count += 1
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total images processed: {processed_count}")
        print(f"Images with matches (>={self.threshold}%): {matched_count}")
        print(f"Results saved to: {self.output_folder}")
        print(f"{'='*60}")
    
    def process_single_image(self, img_path):
        """
        Process a single public image.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Dictionary with processing results
        """
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print("  Error loading image")
            return None
        
        height, width = image.shape[:2]
        
        # Detect faces
        faces = self.detect_faces(img_path)
        
        if not faces:
            print("  No faces detected")
            return None
        
        print(f"  Detected {len(faces)} face(s)")
        
        has_match = False
        temp_face_dir = Path("temp_faces")
        temp_face_dir.mkdir(exist_ok=True)
        
        # Process each detected face
        for idx, face_obj in enumerate(faces):
            # Get facial area coordinates
            facial_area = face_obj['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # Calculate actual coordinates
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            
            # Extract face region for comparison
            face_roi = image[y1:y2, x1:x2]
            temp_face_path = temp_face_dir / f"temp_face_{idx}.jpg"
            cv2.imwrite(str(temp_face_path), face_roi)
            
            # Compare with reference images
            similarity = self.compare_with_references(str(temp_face_path))
            
            # Determine color based on threshold
            if similarity >= self.threshold:
                color = (0, 255, 0)  # Green for match
                has_match = True
                status = "MATCH"
            else:
                color = (0, 0, 255)  # Red for non-match
                status = "NO MATCH"
            
            # Draw rectangle with thicker border
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Prepare text
            text = f"{similarity:.1f}%"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            # Draw filled rectangle for text background
            text_y = y2 + 5
            if text_y + text_height + baseline + 10 > height:
                text_y = y1 - text_height - baseline - 5
            
            cv2.rectangle(
                image,
                (x1, text_y),
                (x1 + text_width + 10, text_y + text_height + baseline + 10),
                color,
                cv2.FILLED
            )
            
            # Draw text
            cv2.putText(
                image,
                text,
                (x1 + 5, text_y + text_height + 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
            print(f"    Face {idx+1}: {similarity:.1f}% similarity - {status}")
            
            # Clean up temp face
            temp_face_path.unlink()
        
        # Clean up temp directory
        try:
            temp_face_dir.rmdir()
        except:
            pass
        
        # Save annotated image
        output_path = self.output_folder / img_path.name
        cv2.imwrite(str(output_path), image)
        print(f"  ✓ Saved to: {output_path.name}")
        
        return {
            'has_match': has_match,
            'face_count': len(faces)
        }


def main():
    """Main execution function."""
    
    # Configuration
    REFERENCE_FOLDER = "reference_images"  # Folder with images of person X only
    PUBLIC_FOLDER = "public_images"        # Folder with images to search
    OUTPUT_FOLDER = "output"               # Folder to save results
    THRESHOLD = 65                         # Similarity threshold percentage
    
    print("="*60)
    print("Missing Person Face Recognition System")
    print("Using DeepFace with VGG-Face Model")
    print("="*60)
    print(f"Reference folder: {REFERENCE_FOLDER}")
    print(f"Public folder: {PUBLIC_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Threshold: {THRESHOLD}%")
    print("="*60 + "\n")
    
    # Initialize detector
    try:
        detector = MissingPersonDetector(
            reference_folder=REFERENCE_FOLDER,
            public_folder=PUBLIC_FOLDER,
            output_folder=OUTPUT_FOLDER,
            threshold=THRESHOLD
        )
        
        # Process all public images
        detector.process_public_images()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print(f"  1. '{REFERENCE_FOLDER}' folder exists with reference images")
        print(f"  2. '{PUBLIC_FOLDER}' folder exists with public images to search")
        print("  3. Required libraries are installed:")
        print("     pip install deepface opencv-python tf-keras")


if __name__ == "__main__":
    main()