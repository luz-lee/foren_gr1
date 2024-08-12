import cv2
import numpy as np

def detect_watermark_image(image, watermark_positions, watermark_text):
    # Step 1: Convert image to YCrCb color space
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, _, _ = cv2.split(ycrcb_image)

    # Step 2: Perform DCT on the Y channel
    dct_y = cv2.dct(np.float32(y_channel))
    
    # Step 3: Highlight the watermark areas using known positions
    watermark_highlight = highlight_watermark_areas(dct_y, image, watermark_positions, watermark_text)
    
    # Convert the highlighted image back to BGR color space
    highlighted_image = cv2.cvtColor(watermark_highlight, cv2.COLOR_YCrCb2BGR)

    return highlighted_image, watermark_text

def highlight_watermark_areas(dct_y, original_image, watermark_positions, watermark_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    watermark_highlight = original_image.copy()
    watermark_highlight = cv2.cvtColor(watermark_highlight, cv2.COLOR_BGR2YCrCb)

    # Highlight the known watermark positions
    for (text_x, text_y, text_w, text_h) in watermark_positions:
        cv2.putText(watermark_highlight, watermark_text, (text_x, text_y), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(watermark_highlight, (text_x, text_y - text_h), (text_x + text_w, text_y), (0, 0, 255), 2)

    return watermark_highlight
