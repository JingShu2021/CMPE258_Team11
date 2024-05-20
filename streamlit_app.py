import streamlit as st
import os
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

UPLOAD_FOLDER = '/data/cmpe258-sp24/jingshu/Data/app_uploaded_video'
EXTRACTED_FRAME_FOLDER = 'app_extracted_frames'
PROCESSED_FOLDER = '/data/cmpe258-sp24/jingshu/Data/app_processed_video/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join(UPLOAD_FOLDER, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        return False

st.title("Video Upload for Processing")

uploaded_file = st.file_uploader("Choose a file", type=['mp4', 'avi', 'mov'])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success("File saved successfully.")
        video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        video_name = uploaded_file.name.split('.')[0]
        print("uploaded video name: ", video_name)
        
        # Get the frames folder
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        args = parser.parse_args()
        frames_dir = os.path.join(args.mot_path,EXTRACTED_FRAME_FOLDER)
        frames_dir = os.path.join(frames_dir,video_name)
        # os.makedirs(frames_dir, exist_ok=True)
        
        # Process video
        # step 1: extrac frames from the video
        extract_frame_from_video(video_path,frames_dir)
        # step 2: annotated frames through inference
        annotate_frames(args, EXTRACTED_FRAME_FOLDER)
        # step 3: generated video from annotated frames
        generate_output_video(PROCESSED_FOLDER)

        # Display processed video
        processed_video_path = os.path.join(PROCESSED_FOLDER,video_name)
        print("processed video path: ", processed_video_path)
        st.video(processed_video_path)
    else:
        st.error("Failed to save file.")

st.button("Back to Upload", on_click=lambda: st.experimental_rerun())