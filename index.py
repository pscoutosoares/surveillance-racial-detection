from deepface import DeepFace
import cv2
import os
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np

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
output_dir = "faces_detected"
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
        
    # Salva o frame temporariamente para o DeepFace
    temp_frame_path = "temp_frame.png"
    cv2.imwrite(temp_frame_path, frame)
    
    try:
        # Detecta faces no frame usando DeepFace
        resultado = DeepFace.extract_faces(
            img_path=temp_frame_path,
            detector_backend='retinaface',
            enforce_detection=False,
            align=True
        )
        
        if resultado:
            # Salva cada face detectada
            for i, face in enumerate(resultado):
                try:
                    # Verifica o formato da face
                    if isinstance(face, dict):
                        face_img = face['face']
                        confidence = face.get('confidence', 1.0)
                        facial_area = face.get('facial_area', [0, 0, 0, 0])
                    elif isinstance(face, tuple):
                        face_img = face[0]
                        confidence = 1.0
                        facial_area = [0, 0, 0, 0]
                    else:
                        face_img = face
                        confidence = 1.0
                        facial_area = [0, 0, 0, 0]
                    
                    # Converte para numpy array se necessário
                    if not isinstance(face_img, np.ndarray):
                        face_img = np.array(face_img)
                    
                    # Converte para uint8 se necessário
                    if face_img.dtype != np.uint8:
                        face_img = (face_img * 255).astype(np.uint8)
                    
                    # Salva a face temporariamente para análise
                    temp_face_path = os.path.join(frame_dir, f"temp_face_{i:03d}.png")
                    cv2.imwrite(temp_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    
                    # Analisa atributos demográficos da face
                    try:
                        demographies = DeepFace.analyze(
                            img_path=temp_face_path,
                            actions=['age', 'gender', 'race'],
                            enforce_detection=False,
                            silent=True
                        )
                    except Exception as e:
                        print(f"\nErro ao analisar demografia da face {i} do frame {frame_count}: {str(e)}")
                        demographies = None
                    
                    # Remove o arquivo temporário da face
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
                    
                    # Salva a face
                    face_path = os.path.join(frame_dir, f"face_{i:03d}.png")
                    cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    faces_count += 1
                    
                    # Salva informações da face em JSON
                    face_info = {
                        'face_index': i,
                        'confidence': confidence, 
                        'facial_area': facial_area, 
                        'demographics': convert_to_serializable(demographies[0]) if demographies else None
                    }
                    face_info_path = os.path.join(frame_dir, f"face_{i:03d}_info.json")
                    with open(face_info_path, 'w', encoding='utf-8') as f:
                        json.dump(face_info, f, indent=4, ensure_ascii=False)
                    
                except Exception as e:
                    print(f"\nErro ao processar face {i} do frame {frame_count}: {str(e)}")
                    continue
    except Exception as e:
        print(f"\nErro ao processar frame {frame_count}: {str(e)}")
    
    # Remove o arquivo temporário
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)
    
    frame_count += 1
    pbar.update(1)

# Fecha a barra de progresso
pbar.close()

# Mostra estatísticas
total_time = time.time() - start_time
print(f"\nProcessamento concluído!")
print(f"Total de frames processados: {frame_count}")
print(f"Total de faces detectadas: {faces_count}")
print(f"Tempo total: {total_time:.2f} segundos")
print(f"Velocidade média: {frame_count/total_time:.2f} frames/segundo")

# Libera os recursos
cap.release()
