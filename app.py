from flask import Flask, request, render_template, send_from_directory
import cv2
from tools.DSText.ExtractFrame_FromVideo import extract_frame_from_video 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torchvision.transforms.functional as F
import torch
from models import build_model
from util.tool import load_model
from main import get_args_parser
from eval import Detector
from tools.DSText.ExtractFrame_FromVideo import extract_frame_from_video
from inference import annotate_frames,generate_output_video

app = Flask(__name__)
UPLOAD_FOLDER = '/data/cmpe258-sp24/jingshu/Data/app_uploaded_video'
EXTRACTED_FRAME_FOLDER = 'app_extracted_frames'
PROCESSED_FOLDER = '/data/cmpe258-sp24/jingshu/Data/app_processed_video/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    print("receiving request ${request.method} from the client...")
    if request.method == 'POST':
        # save the upload video in a folder
        video_file = request.files['video']
        print("uploaded video file: ", video_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)
        video_name = video_file.filename.split('.')[0]
        print("uploaded video name: ", video_name)
        
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        args = parser.parse_args()
        frames_dir = os.path.join(args.mot_path,EXTRACTED_FRAME_FOLDER)
        frames_dir = os.path.join(frames_dir,video_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Process video
        # step 1: extract frames from the video
        # extract_frame_from_video(video_path,frames_dir)
        # step 2: annotated frames through inference
        # annotate_frames(args, EXTRACTED_FRAME_FOLDER)
        # step 3: generated video from annotated frames
        generate_output_video(PROCESSED_FOLDER)

        # Display processed video
        video_path = os.path.join(PROCESSED_FOLDER,video_file.filename)
        print("processed video path: ", video_path)
        return render_template('display_video.html', video_file=video_file.filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)