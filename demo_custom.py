# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

# Modified by Marcelo Contreras

import argparse
import glob 
import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F2

from time import time
from torchvision import transforms
from PIL import Image
from efficientvit.seg_model_zoo import create_seg_model


class Resize(object):
    def __init__(
        self,
        crop_size: tuple[int, int] or None,
        interpolation = cv2.INTER_CUBIC,
    ):
        self.crop_size = crop_size
        self.interpolation = interpolation

    def __call__(self, image:np.ndarray) -> np.ndarray:
        height, width = self.crop_size

        h, w, _ = image.shape
        if width != w or height != h:
            image = cv2.resize(
                image,
                dsize=(width, height),
                interpolation=self.interpolation,
            )
        return image

class ToTensor(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return image


class_colors = (
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32))

def get_canvas(image: np.ndarray,mask: np.ndarray,colors: tuple or list,opacity=0.5) -> np.ndarray:
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        seg_mask[mask == k, :] = color
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/marcelo/Documentos/NODElab/DNN/Semantic_seg/PIDNet/samples/")
    parser.add_argument("--extension",type=str,default=".png",choices=[".png",".jpg"])
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "ade20k"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--model", type=str, default="b0")
    parser.add_argument("--weight_url", type=str, default='/home/marcelo/Documentos/NODElab/DNN/Semantic_seg/efficientvit/weights/b0.pt')
    parser.add_argument("--save_path", type=str, default='/home/marcelo/Documentos/NODElab/DNN/Semantic_seg/efficientvit/results')
    args = parser.parse_args()

    transform = transforms.Compose(
    [
        Resize((args.crop_size, args.crop_size * 2)),
        ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    )
    
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = create_seg_model(args.model, args.dataset, weight_url=args.weight_url)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    images_list = glob.glob(args.path+'*'+args.extension)

    with torch.inference_mode():
        for i,img_path in enumerate(images_list):
            img_name = img_path.split("/")[-1]
            img = cv2.imread(os.path.join(args.path, img_name),cv2.IMREAD_COLOR)
            start = time()
            img = transform(img)
            img = torch.unsqueeze(img, 0).cuda()
            # compute output
            output = model(img)               
            pred = F2.interpolate(output, size=(800,1920), 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            # saving results 
            if args.save_path is not None:
                with open(os.path.join(args.save_path, "summary.txt"), "a") as fout:
                    raw_image = np.array(Image.open(os.path.join(args.path, img_name)).convert("RGB"))
                    canvas = get_canvas(raw_image, pred, class_colors)
                    canvas = Image.fromarray(canvas)
                    end = time()
                    print("Inference time: {:10.4f} ms".format((end-start)*1000.0))
                    canvas.save(os.path.join(args.save_path, f"{i}.png"))
                    fout.write(f"{i}:\t{img_path}\n")

if __name__ == "__main__":
    main()