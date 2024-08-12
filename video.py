import cv2
import numpy as np
import random
import tempfile
from moviepy.editor import VideoFileClip
from utils import resize_image, text_to_image

def add_and_detect_watermark_video(video_path, watermark_text, num_watermarks=5):
    def add_watermark_to_frame(frame):
        watermark_positions = []

        h, w, _ = frame.shape
        h_new = (h // 8) * 8
        w_new = (w // 8) * 8
        frame_resized = cv2.resize(frame, (w_new, h_new))
        
        ycrcb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
        
        dct_y = cv2.dct(np.float32(y_channel))
        
        rows, cols = dct_y.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        for _ in range(num_watermarks):
            text_size = cv2.getTextSize(watermark_text, font, 0.5, 1)[0]
            text_x = random.randint(0, cols - text_size[0])
            text_y = random.randint(text_size[1], rows)
            watermark = np.zeros_like(dct_y)
            watermark = cv2.putText(watermark, watermark_text, (text_x, text_y), font, 0.5, (1, 1, 1), 1, cv2.LINE_AA)
            dct_y += watermark * 0.01
            watermark_positions.append((text_x, text_y, text_size[0], text_size[1]))
        
        idct_y = cv2.idct(dct_y)
        
        ycrcb_image[:, :, 0] = idct_y
        watermarked_frame = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
        
        watermark_highlight = watermarked_frame.copy()
        for (text_x, text_y, text_w, text_h) in watermark_positions:
            cv2.putText(watermark_highlight, watermark_text, (text_x, text_y), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(watermark_highlight, (text_x, text_y - text_h), (text_x + text_w, text_y), (0, 0, 255), 2)
        
        return watermarked_frame, watermark_highlight

    video = VideoFileClip(video_path)
    video_with_watermark = video.fl_image(lambda frame: add_watermark_to_frame(frame)[0])
    video_with_highlight = video.fl_image(lambda frame: add_watermark_to_frame(frame)[1])

    temp_fd, watermarked_video_path = tempfile.mkstemp(suffix=".mp4")
    temp_fd_highlight, highlight_video_path = tempfile.mkstemp(suffix=".mp4")
    
    video_with_watermark.write_videofile(watermarked_video_path, codec='libx264')
    video_with_highlight.write_videofile(highlight_video_path, codec='libx264')

    return watermarked_video_path, highlight_video_path, watermarked_video_path, highlight_video_path
