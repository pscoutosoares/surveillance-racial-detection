from deepface import DeepFace
import cv2
import os
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np

def process_video(video_path, output_dir, frame_interval=30):
    """
    Processa um vídeo e extrai faces usando DeepFace
    frame_interval: processa 1 frame a cada N frames (default: 30)
    """
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return
    
    # Obtém informações do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nProcessando vídeo: {os.path.basename(video_path)}")
    print(f"Total de frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Resolução: {width}x{height}")
    
    frame_count = 0
    faces_count = 0
    start_time = time.time()
    
    # Lista para armazenar informações das faces
    faces_info = []
    
    # Barra de progresso
    pbar = tqdm(total=total_frames, desc="Processando frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Processa apenas 1 frame a cada frame_interval frames
        if frame_count % frame_interval == 0:
            # Salva o frame temporariamente
            temp_frame_path = f"temp_frame_{frame_count}.png"
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # Detecta faces no frame
                result = DeepFace.extract_faces(
                    img_path=temp_frame_path,
                    detector_backend='retinaface',
                    enforce_detection=False,
                    align=True
                )
                
                if result:
                    # Salva cada face detectada
                    for i, face in enumerate(result):
                        try:
                            # Verifica o formato da face
                            if isinstance(face, dict):
                                face_img = face['face']
                                confidence = face.get('confidence', 1.0)
                                facial_area = face.get('facial_area', [0, 0, 0, 0])
                            elif isinstance(face, tuple):
                                # Se for uma tupla, assume que o primeiro elemento é a imagem
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
                            
                            # Verifica se a imagem é válida
                            if face_img.size == 0:
                                print(f"\nAviso: Face vazia detectada no frame {frame_count}")
                                continue
                            
                            # Salva a face
                            face_path = os.path.join(output_dir, f"frame_{frame_count}_face_{i}.png")
                            cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                            faces_count += 1
                            
                            # Armazena informações da face
                            face_info = {
                                'frame': frame_count,
                                'face_index': i,
                                'confidence': confidence,
                                'facial_area': facial_area
                            }
                            faces_info.append(face_info)
                            
                        except Exception as e:
                            print(f"\nErro ao processar face {i} do frame {frame_count}: {str(e)}")
                            continue
                
            except Exception as e:
                print(f"\nErro ao processar frame {frame_count}: {str(e)}")
                print(f"Tipo do resultado: {type(result)}")
                if isinstance(result, list):
                    print(f"Tamanho da lista: {len(result)}")
                    for i, item in enumerate(result):
                        print(f"Item {i}: {type(item)}")
                        if isinstance(item, tuple):
                            print(f"Tamanho da tupla: {len(item)}")
            
            # Remove o frame temporário
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        
        frame_count += 1
        pbar.update(1)
    
    # Fecha a barra de progresso
    pbar.close()
    
    # Salva as informações das faces em JSON
    if faces_info:
        json_path = os.path.join(output_dir, "faces_info.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(faces_info, f, indent=4, ensure_ascii=False)
    
    # Libera os recursos
    cap.release()
    
    # Mostra estatísticas
    total_time = time.time() - start_time
    print(f"\nProcessamento concluído!")
    print(f"Total de frames processados: {frame_count}")
    print(f"Total de faces detectadas: {faces_count}")
    print(f"Tempo total: {total_time:.2f} segundos")
    print(f"Velocidade média: {frame_count/total_time:.2f} frames/segundo")

def process_dataset(base_dir, output_base_dir, frame_interval=30):
    """
    Processa todos os vídeos em todas as classes do dataset
    """
    # Lista todas as classes (pastas) no diretório base
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        output_dir = os.path.join(output_base_dir, class_name)
        
        print(f"\nProcessando classe: {class_name}")
        
        # Lista todos os vídeos na classe
        videos = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi'))]
        
        for video in videos:
            video_path = os.path.join(class_dir, video)
            video_output_dir = os.path.join(output_dir, os.path.splitext(video)[0])
            
            process_video(video_path, video_output_dir, frame_interval)

if __name__ == "__main__":
    # Configurações
    DATASET_DIR = "dataset"
    OUTPUT_DIR = "faces_detected"
    FRAME_INTERVAL = 30  # Processa 1 frame a cada 30 frames
    
    # Processa todo o dataset
    process_dataset(DATASET_DIR, OUTPUT_DIR, FRAME_INTERVAL) 