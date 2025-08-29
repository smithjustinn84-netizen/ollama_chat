import base64
import io
from PIL import Image


def image_to_base64(image_input):
    if image_input is None:
        return None
    # If image_input is a NumPy array (from gr.Image), convert it to PIL Image
    if isinstance(image_input, Image.Image):
        pil_image = image_input
    else:  # Assuming it's a file path or bytes, handle accordingly
        try:
            pil_image = Image.open(image_input)
        except TypeError:  # If it's already a PIL Image object
            pil_image = image_input
        except Exception as e:
            print(f"Error opening image: {e}")
            return None

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")  # Or JPEG, depending on your needs
    return base64.b64encode(buffered.getvalue()).decode('utf-8')