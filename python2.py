import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace
import warnings
from PIL import Image
from PIL.ExifTags import TAGS
import shutil
import boto3
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import argparse

warnings.filterwarnings('ignore')

# ------------------ GPS Helper Functions ------------------

def get_exif_data(image_path):
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
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_gps_coordinates(exif_data):
    if "GPSInfo" not in exif_data:
        return None, None, None
    gps_info = exif_data["GPSInfo"]
    lat = lon = alt = None
    if 1 in gps_info and 2 in gps_info:
        lat_ref = gps_info[1]
        lat = get_decimal_from_dms(gps_info[2], lat_ref)
    if 3 in gps_info and 4 in gps_info:
        lon_ref = gps_info[3]
        lon = get_decimal_from_dms(gps_info[4], lon_ref)
    if 6 in gps_info:
        alt = float(gps_info[6])
    return lat, lon, alt

def generate_google_maps_link(lat, lon):
    if lat is None or lon is None:
        return None
    return f"https://www.google.com/maps?q={lat},{lon}"

# ------------------ SES Helper Function ------------------

def send_email_with_attachment(sender, recipient, subject, body_text, attachment_path):
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(body_text, 'plain'))

    with open(attachment_path, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={attachment_path.name}')
        msg.attach(part)

    client = boto3.client('ses', region_name="us-east-1")

    try:
        response = client.send_raw_email(
            Source=sender,
            Destinations=[recipient],
            RawMessage={'Data': msg.as_string()}
        )
        print(f"✅ Email sent! Message ID: {response['MessageId']}")
    except ClientError as e:
        print(f"❌ Email failed: {e.response['Error']['Message']}")

# ------------------ MissingPersonDetector Class ------------------

class MissingPersonDetector:
    def __init__(self, reference_folder, public_folder, output_folder, threshold=55):
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
        reference_files = [
            f for f in self.reference_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        if not reference_files:
            raise ValueError(f"No images found in {self.reference_folder}")

        for img_path in reference_files:
            print(f"  ✓ Loaded: {img_path.name}")
            self.reference_images.append(str(img_path))

        print(f"\nLoaded {len(self.reference_images)} reference images\n")

    def detect_faces(self, img_path):
        try:
            return DeepFace.extract_faces(
                img_path=str(img_path),
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
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
            similarity = max(0, min(100, (1 - distance / 0.8) * 100))
            return similarity
        except Exception:
            return 0

    def compare_with_references(self, face_image_path):
        max_similarity = 0
        for ref_img in self.reference_images:
            similarity = self.calculate_similarity(ref_img, face_image_path)
            max_similarity = max(max_similarity, similarity)
        return max_similarity

    def process_public_images(self):
        print("Processing public images...\n")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        public_files = [
            f for f in self.public_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not public_files:
            print("No images found.")
            return

        for index, img_path in enumerate(public_files, start=1):
            print(f"\n[{index}/{len(public_files)}] {img_path.name}")
            self.process_single_image(img_path)

        print("\nProcessing completed.")

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

            temp_face_path = temp_face_dir / f"face_{idx}.jpg"
            cv2.imwrite(str(temp_face_path), face_roi)

            similarity = self.compare_with_references(str(temp_face_path))
            status = "MATCH" if similarity >= self.threshold else "NO MATCH"
            color = (0, 255, 0) if status == "MATCH" else (0, 0, 255)

            if status == "MATCH":
                has_match = True

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            label = f"{similarity:.1f}%"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            print(f"    Face {idx+1}: {similarity:.1f}% similarity - {status}")

        shutil.rmtree(temp_face_dir, ignore_errors=True)

        output_path = self.output_folder / img_path.name
        cv2.imwrite(str(output_path), image)
        print(f"  ✓ Saved: {output_path.name}")

        exif = get_exif_data(img_path)
        lat, lon, _ = get_gps_coordinates(exif)
        map_link = generate_google_maps_link(lat, lon)

        if map_link:
            print(f"  🌐 Google Maps link: {map_link}")
        else:
            print("  ❌ No GPS data")

        # ---------- EMAIL ALWAYS SENT ----------
        sender = "vamsikadiyam444@gmail.com"
        recipient = "m.komal12345@gmail.com"

        if has_match:
            subject = "MATCH FOUND!"
            body_text = f"A MATCH was found in image: {img_path.name}\n"
        else:
            subject = "NO MATCH FOUND"
            body_text = f"No match detected in image: {img_path.name}\n"

        body_text += f"Threshold value: {self.threshold}%\n"

        if map_link:
            body_text += f"Location: {map_link}\n"
        else:
            body_text += "Location: Not available\n"

        body_text += "See attached image."

        send_email_with_attachment(sender, recipient, subject, body_text, output_path)

        return {'has_match': has_match, 'faces': len(faces)}

# ------------------ Main Function ------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single public image to process"
    )
    args = parser.parse_args()

    REFERENCE_FOLDER = "reference_images"
    PUBLIC_FOLDER = "public_images"
    OUTPUT_FOLDER = "output"
    THRESHOLD = 55

    print("="*60)
    print(" Missing Person Detection System")
    print("="*60)

    detector = MissingPersonDetector(
        reference_folder=REFERENCE_FOLDER,
        public_folder=PUBLIC_FOLDER,
        output_folder=OUTPUT_FOLDER,
        threshold=THRESHOLD
    )

    # -------- SINGLE IMAGE MODE --------
    if args.image:
        print(f"\nRunning SINGLE IMAGE mode for: {args.image}\n")
        detector.process_single_image(Path(args.image))
        return

    # -------- FULL FOLDER MODE --------
    detector.process_public_images()


if __name__ == "__main__":
    main() 