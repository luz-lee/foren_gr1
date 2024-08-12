import cv2
import numpy as np

def detect_watermark_image(image):
    # Step 1: Convert image to YCrCb color space
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, _, _ = cv2.split(ycrcb_image)

    # Step 2: Perform DCT on the Y channel
    dct_y = cv2.dct(np.float32(y_channel))
    
    # Step 3: Attempt to detect the watermark by analyzing DCT coefficients
    detected_text = detect_watermark_from_dct(dct_y)
    
    # Step 4: Visualize the detection result
    detected_image = cv2.idct(dct_y)
    detected_image = np.uint8(np.clip(detected_image, 0, 255))

    return detected_image, detected_text

def detect_watermark_from_dct(dct_y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    possible_texts = ["WATERMARK", "TEST", "SAMPLE"]  # List of potential watermarks
    detected_text = ""
    max_score = -np.inf

    for text in possible_texts:
        text_image = np.zeros_like(dct_y)
        text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        rows, cols = dct_y.shape

        for text_x in range(0, cols - text_size[0], 10):  # Adjust step as necessary
            for text_y in range(text_size[1], rows, 10):
                watermark = np.zeros_like(dct_y)
                watermark = cv2.putText(watermark, text, (text_x, text_y), font, 0.5, (1, 1, 1), 1, cv2.LINE_AA)
                score = np.sum(dct_y * watermark)
                
                if score > max_score:
                    max_score = score
                    detected_text = text

    return detected_text
