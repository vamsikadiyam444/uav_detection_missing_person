from PIL import Image
from PIL.ExifTags import TAGS

# List of images to check
images = [
    "public_images/find.jpg",
    "public_images/find2.jpg",
    "public_images/rainy.jpg",
    "public_images/windy.jpg"
]

def has_gps(exif_data):
    """Check if image EXIF contains GPSInfo."""
    return 34853 in exif_data if exif_data else False

for img_path in images:
    try:
        img = Image.open(img_path)
        exif = img._getexif()
        if has_gps(exif):
            print(f"✅ {img_path} contains GPS coordinates")
        else:
            print(f"❌ {img_path} does NOT contain GPS coordinates")
    except Exception as e:
        print(f"❌ Failed to open {img_path}: {e}")
