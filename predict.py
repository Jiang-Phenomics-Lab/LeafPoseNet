import os
import cv2
import time
import math
import torch
import argparse
import numpy as np
import pandas as pd
from nets.LANet import LANet
from tqdm import tqdm 

def adjust_image(image, fixed_size):
    h, w, c = image.shape
    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        pad = int(((h / hw_ratio) - w) / 2)
        new_image = np.zeros((h, w + 2 * pad, c))
        new_image[:, pad:w + pad, :] = image
    elif h / w < hw_ratio:
        pad = int(((w * hw_ratio) - h) / 2)
        new_image = np.zeros((h + 2 * pad, w, c))
        new_image[pad:h + pad, :, :] = image
    else:
        new_image = image
    return new_image

def predict(args):
    # 参数
    mean = [0.501642, 0.517964, 0.514330]
    std = [0.139957, 0.155858, 0.153716]
    fixed_size = (768, 576)

    # model loader
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = LANet()
    weights = torch.load(args.model_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # Initialisation results table
    data = pd.DataFrame(columns=['Image_Name', 'Angle', 'x0', 'x1', 'x2', 'y0', 'y1', 'y2'])

    start_time = time.time()
    num = 0

    with torch.no_grad():
        image_list = [f for f in os.listdir(args.data_dir)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

        for img_name in tqdm(image_list, desc="🔍 Predicting", unit="img"):
            img_path = os.path.join(args.data_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {img_name}")
                continue

            num += 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] < img.shape[1]:
                img = cv2.transpose(img)[::-1, :, :]
            img = adjust_image(img, fixed_size)
            img = cv2.resize(img, (576, 768))
            img = (img / 255 - mean) / std
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            outputs = model(img_t)
            b, n, h, w = outputs.shape
            heatmaps = outputs.view(b, n, -1)
            maxvals, idx = torch.max(heatmaps, dim=2)
            idx = idx.float()
            x_v = idx[0] % w
            y_v = torch.floor(idx[0] / w)
            preds = torch.cat((x_v, y_v), dim=0) * 2
            predict = preds.cpu().numpy()

            if predict[5] - predict[4] < 0:
                vector_A = [predict[2] - predict[1], predict[5] - predict[4]]
            else:
                vector_A = [predict[1] - predict[2], predict[4] - predict[5]]
            vector_B = [predict[0] - predict[1], predict[3] - predict[4]]
            dot_product = sum(a * b for a, b in zip(vector_A, vector_B))
            mag_A = math.sqrt(sum(a**2 for a in vector_A))
            mag_B = math.sqrt(sum(b**2 for b in vector_B))
            angle_rad = math.acos(dot_product / (mag_A * mag_B))
            angle_deg = math.degrees(angle_rad)

            temp_df = pd.DataFrame({
                'Image_Name': [img_name[:-4]],
                'Angle': [angle_deg],
                'x0': [predict[0]], 'x1': [predict[1]], 'x2': [predict[2]],
                'y0': [predict[3]], 'y1': [predict[4]], 'y2': [predict[5]]
            })
            data = pd.concat([data, temp_df], ignore_index=True)

    data.to_csv(args.output_csv, index=False)
    end_time = time.time()
    print(f"\n✅ Done! Processed {num} images in {end_time - start_time:.2f} seconds.")
    print(f"📄 Prediction results saved to: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LANet Image Prediction")
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use: "cuda:0" or "cpu"')
    parser.add_argument('--data-dir', type=str, default="datasets/images", help='Directory containing images to predict')
    parser.add_argument('--model-path', type=str, default="weights/LANet_best.pth", help='Path to trained LANet model weights (.pth)')
    parser.add_argument('--output-csv', type=str, default='output/predict_result.csv', help='Path to save output CSV')
    args = parser.parse_args()

    predict(args)
