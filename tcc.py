from deepface import DeepFace
import cv2
import os
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np
from gfpgan import GFPGANer
from PIL import Image
from datetime import datetime
from basicsr.utils import imwrite

def convert_to_serializable(obj):
    """Converte valores não serializáveis para tipos serializáveis"""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj



# Abre o vídeo
video_path = "dataset/Robbery/Robbery034_x264.mp4"
# video_path = "dataset/Robbery/Robbery034_x264.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()

# Obtém informações do vídeo
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nProcessando vídeo: {os.path.basename(video_path)}")
print(f"Total de frames: {total_frames}")
print(f"FPS: {fps}")
print(f"Resolução: {width}x{height}")

# Cria diretório base para salvar as faces
output_dir = "faces_detected/run-" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
faces_count = 0
start_time = time.time()

# Barra de progresso
pbar = tqdm(total=total_frames, desc="Processando frames")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Cria diretório para o frame atual
    frame_dir = os.path.join(output_dir, f"frame_{frame_count:05d}")
    os.makedirs(frame_dir, exist_ok=True)
    
    # Salva o frame original
    frame_path = os.path.join(frame_dir, "frame.png")
    cv2.imwrite(frame_path, frame)

    temp_frame_path = "temp_frame.png"
    cv2.imwrite(temp_frame_path, frame)

    resultado = DeepFace.extract_faces(
        img_path=temp_frame_path,
        detector_backend='retinaface',
        enforce_detection=False,
        align=True
    )
        
    for i, face in enumerate(resultado):
        face_img = face['face']
        confidence = face.get('confidence', 1.0)
        facial_area = face.get('facial_area', [0, 0, 0, 0])

        face_img = (face_img * 255).astype(np.uint8)

        temp_face_path = os.path.join(frame_dir, f"temp_face_{i:03d}.png")
        cv2.imwrite(temp_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

        image = cv2.imread(temp_frame_path, cv2.IMREAD_COLOR)
        img_alta_resolucao = cv2.resize(image, (512,512), interpolation=cv2.INTER_CUBIC)

        # Inicializa o GFPGAN
        restorer = GFPGANer(model_path='weights/GFPGANv1.4.pth', upscale=1 )

        # Melhora a qualidade (mantém como array NumPy)
        _, restored_face, restored_img = restorer.enhance(img_alta_resolucao, has_aligned=False, only_center_face=False, paste_back=True)
        
        save_restored_face_path = os.path.join(frame_dir, f"restored_face_{i:03d}.png")
        # Salva a imagem restaurada
        imwrite(restored_face[0], save_restored_face_path)
        imwrite(restored_img, os.path.join(frame_dir, f"restored_img_{i:03d}.png") )
           
        demographies = DeepFace.analyze(
            img_path=save_restored_face_path,
            actions=['age', 'gender', 'race'],
            enforce_detection=False,
            detector_backend='retinaface',
        )

        face_info = {
            'face_index': i,
            'confidence': confidence, 
            'facial_area': facial_area, 
            'demographics': convert_to_serializable(demographies[0]) if demographies else None
        }
        face_info_path = os.path.join(frame_dir, f"face_{i:03d}_info.json")
        with open(face_info_path, 'w', encoding='utf-8') as f:
            json.dump(face_info, f, indent=4, ensure_ascii=False)
        break
    break

    frame_count += 1
    pbar.update(1)



   