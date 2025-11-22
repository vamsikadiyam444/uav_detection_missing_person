import cv2
import numpy as np
import os
from pathlib import Path
from deepface import DeepFace
import warnings
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import shutil
import boto3  # AWS SNS

warnings.filterwarnings('ignore')


# ------------------ GPS Helper Functions ------------------

def get_exif_data(image_path):
    """Extract EXIF data from an image."""
    image = Image.open(image_path)
    exif_data = image._getexif()
    if not exif_data:
        return {}
    data = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        data[tag] = value
    return data

def get_decimal_from_dms(dms, ref):
    """Convert GPS coordinates in DMS format to decimal."""
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_gps_coordinates(exif_data):
    """Extract latitude, longitude, and altitude if available."""
    if "GPSInfo" not in exif_data:
        return None, None, None

    gps_info = exif_data["GPSInfo"]

    # Extract latitude
    lat = None
    if 1 in gps_info and 2 in gps_info:
        lat_ref = gps_info[1]
        lat = get_decimal_from_dms(gps_info[2], lat_ref)

    # Extract longitude
    lon = None
    if 3 in gps_info and 4 in gps_info:
        lon_ref = gps_info[3]
        lon = get_decimal_from_dms(gps_info[4], lon_ref)

    # Extract altitude
    alt = None
    if 6 in gps_info:
        alt = float(gps_info[6])

    return lat, lon, alt

def generate_google_maps_link(lat, lon):
    """Return a Google Maps URL."""
    if lat is None or lon is None:
        return None
    return f"https://www.google.com/maps?q={lat},{lon}"


# ------------------ AWS SNS Helper ------------------

def send_sns_notification(topic_arn, subject, message):
    """Send a message to an AWS SNS topic."""
    try:
        sns = boto3.client('sns')
        response = sns.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message
        )
        print(f"  ✅ SNS Notification sent! MessageId: {response['MessageId']}")
    except Exception as e:
        print(f"  ❌ SNS Notification failed: {e}")


# ------------------ MissingPersonDetector Class ------------------

class MissingPersonDetector:
    """
    Face recognition system to detect a missing person in public images.
    Uses DeepFace with VGG-Face model for high accuracy.
    """
    
    def __init__(self, reference_folder, public_folder, output_folder, threshold=65):
        self.reference_folder = Path(reference_folder)
        self.public_folder = Path(public_folder)
        self.output_folder = Path(output_folder)
        self.threshold = threshold
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        self.model_name = "VGG-Face"
        self.detector_backend = "retinaface"
        
        self.reference_images = []
        self.load_reference_images()
    
    def load_reference_images(self):
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
        try:
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
        try:
            result = DeepFace.verify(
                img1_path=face1_path,
                img2_path=face2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            distance = result['distance']
            if self.model_name == "VGG-Face":
                similarity = max(0, min(100, (1 - distance / 0.8) * 100))
            else:
                similarity = max(0, min(100, (1 - distance) * 100))
            return similarity
        except Exception as e:
            return 0
    
    def compare_with_references(self, face_image_path):
        max_similarity = 0
        for ref_img in self.reference_images:
            similarity = self.calculate_similarity(ref_img, face_image_path)
            max_similarity = max(max_similarity, similarity)
        return max_similarity
    
    def process_public_images(self):
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
        image = cv2.imread(str(img_path))
        if image is None:
            print("  Error loading image")
            return None
        
        height, width = image.shape[:2]
        faces = self.detect_faces(img_path)
        if not faces:
            print("  No faces detected")
            return None
        
        print(f"  Detected {len(faces)} face(s)")
        
        has_match = False
        temp_face_dir = Path("temp_faces")
        temp_face_dir.mkdir(exist_ok=True)
        
        for idx, face_obj in enumerate(faces):
            facial_area = face_obj['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            face_roi = image[y1:y2, x1:x2]
            temp_face_path = temp_face_dir / f"temp_face_{idx}.jpg"
            cv2.imwrite(str(temp_face_path), face_roi)
            
            similarity = self.compare_with_references(str(temp_face_path))
            
            if similarity >= self.threshold:
                color = (0, 255, 0)
                has_match = True
                status = "MATCH"
            else:
                color = (0, 0, 255)
                status = "NO MATCH"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            text = f"{similarity:.1f}%"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_y = y2 + 5
            if text_y + text_height + baseline + 10 > height:
                text_y = y1 - text_height - baseline - 5
            cv2.rectangle(image, (x1, text_y), (x1 + text_width + 10, text_y + text_height + baseline + 10), color, cv2.FILLED)
            cv2.putText(image, text, (x1 + 5, text_y + text_height + 5), font, font_scale, (255, 255, 255), thickness)
            
            print(f"    Face {idx+1}: {similarity:.1f}% similarity - {status}")
            temp_face_path.unlink()
        
        shutil.rmtree(temp_face_dir, ignore_errors=True)
        
        output_path = self.output_folder / img_path.name
        cv2.imwrite(str(output_path), image)
        print(f"  ✓ Saved to: {output_path.name}")
        
        # ------------------ GPS Extraction ------------------
        exif = get_exif_data(img_path)
        lat, lon, alt = get_gps_coordinates(exif)
        map_link = generate_google_maps_link(lat, lon)
        if map_link:
            print(f"  🌐 Google Maps link: {map_link}")
        else:
            print("  ❌ No GPS data found for this image")
        
        # ------------------ SNS Notification ------------------
        if has_match:
            topic_arn = "arn:aws:sns:REGION:ACCOUNT_ID:YourTopicName"  # Replace with your SNS topic ARN
            subject = "Missing Person Detected!"
            message = f"Missing person detected in image: {img_path.name}\n"
            if map_link:
                message += f"Location (Google Maps): {map_link}\n"
            else:
                message += "Location: Not available\n"
            send_sns_notification(topic_arn, subject, message)
        
        return {'has_match': has_match, 'face_count': len(faces)}


# ------------------ Main Function ------------------

def main():
    REFERENCE_FOLDER = "reference_images"
    PUBLIC_FOLDER = "public_images"
    OUTPUT_FOLDER = "output"
    THRESHOLD = 65
    
    print("="*60)
    print("Missing Person Face Recognition System")
    print("Using DeepFace with VGG-Face Model")
    print("="*60)
    print(f"Reference folder: {REFERENCE_FOLDER}")
    print(f"Public folder: {PUBLIC_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Threshold: {THRESHOLD}%")
    print("="*60 + "\n")
    
    try:
        detector = MissingPersonDetector(
            reference_folder=REFERENCE_FOLDER,
            public_folder=PUBLIC_FOLDER,
            output_folder=OUTPUT_FOLDER,
            threshold=THRESHOLD
        )
        detector.process_public_images()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print(f"  1. '{REFERENCE_FOLDER}' folder exists with reference images")
        print(f"  2. '{PUBLIC_FOLDER}' folder exists with public images to search")
        print("  3. Required libraries are installed: pip install deepface opencv-python pillow tensorflow keras boto3")


if __name__ == "__main__":
    main()
