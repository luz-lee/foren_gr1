import gradio as gr
from image import add_and_detect_watermark_image
from video import add_and_detect_watermark_video
from detect import detect_watermark_image

# Image Interface
def process_image(image, text):
    watermarked_image_path, highlight_path, positions, watermark_text = add_and_detect_watermark_image(image, text)
    return watermarked_image_path, highlight_path, positions, watermark_text

image_interface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="numpy", label="Upload Image"), gr.Textbox(label="Watermark Text")],
    outputs=[gr.Image(label="Watermarked Image"), gr.Image(label="Watermark Highlight"),
             gr.JSON(label="Watermark Positions"), gr.Textbox(label="Watermark Text")],
    title="Image Watermark Application"
)

def process_detection(image, positions, text):
    return detect_watermark_image(image, positions, text)

detect_interface = gr.Interface(
    fn=process_detection,
    inputs=[gr.Image(type="numpy", label="Upload Image"), gr.JSON(label="Watermark Positions"), gr.Textbox(label="Watermark Text")],
    outputs=[gr.Image(label="Detected Watermark"), gr.Textbox(label="Detected Text")],
    title="Forensic Watermark Detection"
)

app = gr.TabbedInterface(
    interface_list=[image_interface, detect_interface],
    tab_names=["Image", "Detect"]
)    
# Video Interface
video_inputs = [
    gr.Video(label="Upload Video"),
    gr.Textbox(label="Watermark Text")
]

video_outputs = [
    gr.Video(label="Watermarked Video"),
    gr.Video(label="Watermark Highlight"),
    gr.File(label="Download Watermarked Video"),
    gr.File(label="Download Watermark Highlight")
]

def process_video(video, text):
    watermarked_video_path, highlight_video_path, _, _ = add_and_detect_watermark_video(video, text)
    return watermarked_video_path, highlight_video_path, watermarked_video_path, highlight_video_path

video_interface = gr.Interface(
    fn=process_video,
    inputs=video_inputs,
    outputs=video_outputs,
    title="Video Watermark Application",
    description="Upload a video and add a watermark text. Detect watermark and highlight its position."
)

# Forensic Watermark Detection Interface
detect_inputs = [
    gr.Image(type="numpy", label="Upload Image")
]

detect_outputs = [
    gr.Image(type="numpy", label="Watermark Detection Result"),
    gr.Textbox(label="Detected Watermark Text")
]

detect_interface = gr.Interface(
    fn=detect_watermark_image,
    inputs=detect_inputs,
    outputs=detect_outputs,
    title="Forensic Watermark Detection",
    description="Upload an image to detect forensic watermarks."
)

# Combine interfaces in tabs
app = gr.TabbedInterface(
    interface_list=[image_interface, video_interface, detect_interface],
    tab_names=["Image", "Video", "Detect"]
)

if __name__ == "__main__":
    app.launch()
