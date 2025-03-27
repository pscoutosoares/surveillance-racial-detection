from gfpgan import GFPGANer
from PIL import Image
import sys
import os
from basicsr.utils import imwrite
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), 'GFPGAN'))

restorer = GFPGANer(model_path='GFPGAN\gfpgan\weights\GFPGANv1.4.pth', upscale=4)

# image = Image.open('Adele_crop.png').convert('RGB')
image = cv2.imread('Adele_crop.png', cv2.IMREAD_COLOR)

# Restaurar rosto
_, restored_img, _ = restorer.enhance(image, has_aligned=False, only_center_face=False)

imwrite(restored_img[0], 'result.png')

