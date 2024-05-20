import cv2
from tools.DSText.ExtractFrame_FromVideo import extract_frame_from_video 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torchvision.transforms.functional as F
import torch
from tqdm import tqdm
from models import build_model
from util.tool import load_model
from main import get_args_parser
from eval import Detector,parse_xml_rec,sort_key,write_lines
import moviepy
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import *


EXTRACTED_FRAME_FOLDER = 'app_extracted_frames'
PROCESSED_FOLDER = '/data/cmpe258-sp24/jingshu/Data/app_processed_video'


os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def annotate_frames(args, frames_dir):
    print('args:', args)
    # load model and weights
    detr, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr = detr.cuda()
    detr.eval()

    dict_cost = {
    "backbone_time" : 0,
    "nect_time" : 0,
    "upsample_time" : 0,
    "transformer_time" : 0,
    "det_head_time" : 0,
    "rec_head_time" : 0,
    "memory_embed_time" : 0,
     "postprocess_time": 0 
    }
    
    args.mot_path = os.path.join(args.mot_path,frames_dir)
    seq_nums = os.listdir(args.mot_path)
    print("Seq_nums: ",seq_nums)
    number_frame = 0
    for seq_num in seq_nums:
        print("solve {}".format(seq_num))
        number_frame += len(os.listdir(os.path.join(args.mot_path, seq_num)))
        
        det = Detector(args, model=detr, seq_num=seq_num)
        time_cost = det.detect(dict_cost,vis=args.show)
        
    print("frame number:",number_frame)
    getid_text(os.path.join(args.output_dir, 'preds'))
    print(time_cost)

def getid_text(new_xml_dir_):
    for xml in tqdm(os.listdir(new_xml_dir_)):
        id_trans = {}
        id_cond = {}
        if ".txt" in xml or "ipynb" in xml:
            continue
                
        lines = []
        xml_one = os.path.join(new_xml_dir_,xml)
        ann = parse_xml_rec(xml_one)
        for frame_id_ann in ann:
            points, IDs, Transcriptions,confidences = ann[frame_id_ann]
            for ids, trans, confidence in zip(IDs,Transcriptions,confidences):
                if str(ids) in id_trans:
                    id_trans[str(ids)].append(trans)
                    id_cond[str(ids)].append(float(confidence))
                else:
                    id_trans[str(ids)]=[trans]
                    id_cond[str(ids)]=[float(confidence)]
                    
        id_trans = sort_key(id_trans)
        id_cond = sort_key(id_cond)

        for i in id_trans:
            txts = id_trans[i]
            confidences = id_cond[i]
            txt = max(txts,key=txts.count)
            
            lines.append('"'+i+'"'+","+'"'+txt+'"'+"\n")
        write_lines(os.path.join(new_xml_dir_,xml.replace("xml","txt")),lines)

def pics2video(frames_dir="", fps=25):
    im_names = os.listdir(frames_dir)
    num_frames = len(im_names)
    frames_path = []
    for im_name in tqdm(range(1, num_frames)):
        string = os.path.join( frames_dir, str(im_name) + '.jpg')
        frames_path.append(string)

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(frames_dir+".mp4", codec='libx264')
            
def generate_output_video(image_path):    
    for video_name in os.listdir(image_path):
        if "mp4" in video_name:
            continue
        
        try:
            video_name_one = os.path.join(image_path,video_name)
            pics2video(frames_dir=video_name_one,fps=5)
        except:
            continue

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # # upload a video from the app, video_name is the video name, video_path is the path for saving the uploaded video
    # # define a folder to save the frames extracted from the video
    # frames_dir = os.path.join(frames_dir,video_name)
    # os.makedirs(frames_dir, exist_ok=True)
    # extract_frame_from_video(video_path,frames_dir)
    
    extracted_frame_path = os.path.join(PROCESSED_FOLDER,"results")
    print("extracted_frame_path: ",extracted_frame_path)
    
    annotate_frames(args, EXTRACTED_FRAME_FOLDER)
    # generate_output_video(extracted_frame_path)