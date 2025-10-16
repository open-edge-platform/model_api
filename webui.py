#!/usr/bin/env python3

from email.mime import image
import os
import glob
import uuid
import cv2
import numpy as np
import gradio as gr
from fastrtc import WebRTC

from PIL import Image
from pathlib import Path

from model_api.models import Model
from model_api.metrics import PerformanceMetrics
from model_api.visualizer import Visualizer

SUBSAMPLE = 2

class ModelAPIGradioApp:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = "CPU"
        self.available_devices = ["CPU", "GPU", "NPU"]
        self.stats = PerformanceMetrics()
        self.available_models = self._discover_models()
        self.visualizer = Visualizer()
        self.load_model(list(self.available_models.keys())[0])

    def _discover_models(self):
        """Discover all available models in the data folder"""
        models = {}
        data_path = Path("data")
        
        # Find all model.xml files
        model_files = list(data_path.glob("**/*.xml"))
        
        for model_file in model_files:
            # Get the relative path from data folder
            rel_path = model_file.relative_to(data_path)
            parts = rel_path.parts
            
            # Skip if not enough folder depth (need at least 3 folders)
            if len(parts) < 3:
                continue
            
            # Create model name from the 3 upper folders (excluding model.xml)
            # Example: AnomalyDetection/STFPM/openvino_fp32
            model_name = "/".join(parts[-4:-1])
            model_path = str(model_file)
            
            models[model_name] = model_path
        
        # Sort models by name for better UI experience
        return dict(sorted(models.items()))



    def load_model(self, model_name):
        """Load a model"""
        try:
            self.model_name = model_name
            model_path = self.available_models[model_name]
            print(f"Loading model: {model_path} to device: {self.device}")
            self.model = Model.create_model(model_path, device=self.device)
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")


    def set_device(self, device_name):
        """Set the device for inference"""
        print(f"Setting device to: {device_name}")
        self.device = device_name
        if self.model:
            self.load_model(self.model_name)


    def run_inference(self, image):
        """Run inference on the input image"""
        print("Running inference...")
        try:
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            predictions = self.model(image_array)
            stats = self.model.get_performance_metrics()
            result_image = self.visualizer.render(image=image_array, result=predictions)
            
            # Format statistics for display
            stats_dict = self.format_statistics(stats)
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None, self.format_statistics(PerformanceMetrics())
        return result_image, stats_dict
    
    def format_statistics(self, stats):
        """Format statistics into a dictionary for display"""
        return {
            'fps': f"{stats.get_fps():.2f}",
            'total_frames': f"{stats.get_total_frames():,}",
            'preprocess_mean': f"{stats.get_preprocess_time().mean() * 1000:.2f}",
            'inference_mean': f"{stats.get_inference_time().mean() * 1000:.2f}",
            'postprocess_mean': f"{stats.get_postprocess_time().mean() * 1000:.2f}",
            'total_mean': f"{stats.total_time.mean() * 1000:.2f}",
            'total_min': f"{stats.get_total_time_min() * 1000:.2f}",
            'total_max': f"{stats.get_total_time_max() * 1000:.2f}",
        }

    def load_images_from_folder(self, folder_path):
        """Load images from a folder and return list of image paths"""
        try:
            if not folder_path or not os.path.exists(folder_path):
                return []
            
            # Common image extensions
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
            image_files = []
            
            for extension in image_extensions:
                image_files.extend(glob.glob(os.path.join(folder_path, extension)))
                image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
            
            # Sort the files for consistent ordering
            image_files.sort()
            print(f"Found {len(image_files)} images in {folder_path}")
            return image_files
            
        except Exception as e:
            print(f"Error loading images from folder {folder_path}: {str(e)}")
            return []
    
    def stream_vid(self, video):
        cap = cv2.VideoCapture(video)

        # This means we will output mp4 videos
        video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        desired_fps = fps
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        iterating, frame = cap.read()

        n_frames = 0

        # Use UUID to create a unique video file
        output_video_name = f"/tmp/output_{uuid.uuid4()}.mp4"

        # Output Video
        output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (width, height)) # type: ignore

        while iterating:
#            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           # frame = cv2.resize( frame, (0,0), fx=0.5, fy=0.5)
            try:
                predictions = self.model(frame)
                result_image = self.visualizer.render(image=frame, result=predictions)
                output_video.write(result_image)
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                return None

            if n_frames % (desired_fps*2) == 0:
                output_video.release()
                # Get current performance metrics
                stats = self.model.get_performance_metrics()
                stats_dict = self.format_statistics(stats)
                
                # Yield video path and stats as tuple
                yield (
                    output_video_name,
                    stats_dict['fps'],
                    stats_dict['total_frames'],
                    stats_dict['preprocess_mean'],
                    stats_dict['inference_mean'],
                    stats_dict['postprocess_mean'],
                    stats_dict['total_mean'],
                    stats_dict['total_min'],
                    stats_dict['total_max']
                )
                output_video_name = f"/tmp/output_{uuid.uuid4()}.mp4"
                output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (width, height)) # type: ignore

            iterating, frame = cap.read()
            n_frames += 1

    def start_camera_stream(self, frame):
        """Process camera stream - runs inference on captured camera frame"""
        try:
            if frame is None:
                return None, "0.00", "0", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00"
            
            #if isinstance(image, Image.Image):
            #    image_array = np.array(image)
            #else:
            #    image_array = image
            
            # Run inference
            predictions = self.model(frame)
            result_image = self.visualizer.render(image=frame, result=predictions)

            # Get current performance metrics
            stats = self.model.get_performance_metrics()
            stats_dict = self.format_statistics(stats)
            
            return (
                result_image,
                stats_dict['fps'],
                stats_dict['total_frames'],
                stats_dict['preprocess_mean'],
                stats_dict['inference_mean'],
                stats_dict['postprocess_mean'],
                stats_dict['total_mean'],
                stats_dict['total_min'],
                stats_dict['total_max']
            )
        except Exception as e:
            print(f"Error during camera inference: {str(e)}")
            return None, "0.00", "0", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00"

    def load_and_process_image(self, image_path):
        if image_path:
            image = Image.open(image_path)
            result_img, stats = self.run_inference(image)
            return (
                image,
                result_img,
                stats['fps'],
                stats['total_frames'],
                stats['preprocess_mean'],
                stats['inference_mean'],
                stats['postprocess_mean'],
                stats['total_mean'],
                stats['total_min'],
                stats['total_max']
            )
        return None, None, "0.00", "0", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00"

    def create_interface(self):
        with gr.Blocks(title="Geti Predict", theme=gr.themes.Base()) as demo:
            gr.Markdown("# Geti Predict")
            with gr.Row():
                with gr.Column(scale=5):
                    tabs = gr.Tabs()
                    with tabs:
                        with gr.TabItem("Image"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    gr.Markdown("### Input Source")
                                    sample_image_files = glob.glob("data/images/*")
                                    sample_image_dropdown = gr.Dropdown(
                                        choices=sample_image_files,
                                        label="Sample Images",
                                        value=sample_image_files[0] if sample_image_files else None
                                    )
                                    sample_image_display = gr.Image(sources=["upload", "clipboard"], label="Image", type="pil", show_label=False)

                                with gr.Column(scale=3):
                                    gr.Markdown("### Inferences")
                                    result = gr.Image(label="Result", type="pil", show_label=False)


                        with gr.TabItem("Dataset"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    gr.Markdown("## Input Dataset")
                                    image_gallery = gr.Gallery(
                                        label="Gallery",
                                        elem_id="gallery",
                                        object_fit="contain",
                                        height="auto",
                                        allow_preview=True,
                                        show_label=False,
                                        show_fullscreen_button=True,
                                        columns=4
                                    )

                                with gr.Column(scale=3):
                                    gr.Markdown("## Inferences")
                                    ds_result = gr.Image(label="DS Result", type="pil", show_label=False)
                            
                                    
                        with gr.TabItem("Video"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    gr.Markdown("## Input Source")
                                    video_input = gr.Video(label="Video", show_label=False)
                                with gr.Column(scale=3):
                                    gr.Markdown("## Inference Stream")
                                    result_stream = gr.Video(
                                        label="Processed Video",
                                        streaming=True,
                                        autoplay=True,
                                        show_label=False,
                                        show_download_button=True)

                        with gr.TabItem("Camera"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    gr.Markdown("## Input Source")
                                    cam_input = gr.Image(
                                        sources=["webcam"],
                                        label="Webcam",
                                        type="numpy",
                                        show_label=False
                                        )
                                with gr.Column(scale=3):
                                    gr.Markdown("## Camera Inference Stream")
                                    cam_result = gr.Image(label="Camera Result", type="pil", show_label=False)
                        with gr.TabItem("Model"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("## Load/Reload Model")
                                    model_dropdown = gr.Dropdown(
                                        choices=self.available_models.keys(),
                                        label="Select Model",
                                        value=self.available_models.keys().__iter__().__next__(),
                                        interactive=True
                                    )
                                    device_dropdown = gr.Dropdown(
                                        choices=self.available_devices,
                                        label="Select Device Source",
                                        value=self.available_devices[0],
                                    )
                                    device_info = gr.Markdown(f"**Device:** {self.device}")

                                with gr.Column(scale=1):
                                    gr.Markdown("### Model Load Status")
                                    model_status = gr.Markdown("")

                            model_dropdown.input(
                                fn=self.load_model,
                                inputs=model_dropdown,
                                outputs=[]
                            )

                            device_dropdown.input(
                                fn=self.set_device,
                                inputs=device_dropdown,
                                outputs=[]
                            )
                with gr.Column(scale=1):
                    # Performance Metrics Dashboard
                    gr.Markdown("### Performance Metrics")
                    with gr.Row():
                        fps_display = gr.Textbox(label="🚀 FPS", value="0.00", interactive=False)
                        total_frames_display = gr.Textbox(label="🎞️ Total Frames", value="0", interactive=False)
                    
                    with gr.Accordion("📊 Detailed Timing (ms)", open=True):
                        with gr.Row():
                            preprocess_display = gr.Textbox(label="⚙️ Preprocess", value="0.00", interactive=False)
                            inference_display = gr.Textbox(label="🧠 Inference", value="0.00", interactive=False)
                            postprocess_display = gr.Textbox(label="🎨 Postprocess", value="0.00", interactive=False)
                        
                        with gr.Row():
                            total_mean_display = gr.Textbox(label="📈 Total (Mean)", value="0.00", interactive=False)
                            total_min_display = gr.Textbox(label="⚡ Total (Min)", value="0.00", interactive=False)
                            total_max_display = gr.Textbox(label="🐌 Total (Max)", value="0.00", interactive=False)

                                   
                    def load_and_process_image_gallery(evt: gr.SelectData):
                        results = self.load_and_process_image(evt.value['image']['path'] if isinstance(evt.value, dict) else evt.value)
                        return results[1:]
                    
                             
                    sample_image_dropdown.input(
                        fn=self.load_and_process_image,
                        inputs=sample_image_dropdown,
                        outputs=[
                            sample_image_display,
                            result,
                            fps_display,
                            total_frames_display,
                            preprocess_display,
                            inference_display,
                            postprocess_display,
                            total_mean_display,
                            total_min_display,
                            total_max_display
                        ]
                    )
                    
                    image_gallery.select(
                        fn=load_and_process_image_gallery,
                        inputs=None,
                        outputs=[
                            ds_result,
                            fps_display,
                            total_frames_display,
                            preprocess_display,
                            inference_display,
                            postprocess_display,
                            total_mean_display,
                            total_min_display,
                            total_max_display
                        ]
                    )

                    video_input.play(
                        fn=self.stream_vid,
                        inputs=[video_input],
                        outputs=[
                            result_stream,
                            fps_display,
                            total_frames_display,
                            preprocess_display,
                            inference_display,
                            postprocess_display,
                            total_mean_display,
                            total_min_display,
                            total_max_display
                        ]
                    )

                    live_stream = cam_input.stream(
                        fn=self.start_camera_stream,
                        inputs=[cam_input],
                        outputs=[
                            cam_result,
                            fps_display,
                            total_frames_display,
                            preprocess_display,
                            inference_display,
                            postprocess_display,
                            total_mean_display,
                            total_min_display,
                            total_max_display
                        ],
                        time_limit=60,
                        stream_every=0.1,
                        concurrency_limit=2)

            return demo


# Create the demo instance at module level for gradio CLI hotreload
app = ModelAPIGradioApp()
demo = app.create_interface()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )