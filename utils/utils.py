import base64
import os

def image_to_base64(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        with open(image_path, "rb") as image_file:
            # Read the entire image file in binary mode
            binary_data = image_file.read()
            # Encode the binary data to Base64
            base64_encoded_data = base64.b64encode(binary_data)
            # Convert bytes to string (important for many web contexts)
            base64_string = base64_encoded_data.decode('utf-8')
            return base64_string
    except Exception as e:
        print(f"Error converting image to Base64: {e}")
        return None