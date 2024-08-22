import gradio as gr
import torch
import requests

from torchvision import transforms
from architecture import *
import argparse

parser = argparse.ArgumentParser(description="Cassva Image Classification Toolbox")
parser.add_argument('--method', type=str, default='Vit', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default='exp/Vit2024_08_09_18_28_32/models/model_epoch_Vit_fold_0_18.pth', help='pretrained model directory')
parser.add_argument("--img_size", type=int, default=512, help='image size')
opt = parser.parse_args()
if opt.method.find('Vit') >= 0:
    opt.img_size = 224


labels_e = ['Cassava Bacterial Blight (CBB)', 'Cassava Brown Streak Disease (CBSD)', 'Cassava Green Mottle (CGM)', 'Cassava Mosaic Disease (CMD)', 'Healthy']
labels_cn = ['木薯细菌性枯萎病', '木薯褐斑病', '木薯绿斑病', '木薯花叶病', '健康']

model = model_generator(opt.method, opt.pretrained_model_path).eval()


def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels_cn[i]: float(prediction[i]) for i in range(5)}
    # max_index = prediction.argmax()
    # result = labels_cn[max_index]
    return confidences

demo = gr.Interface(fn=predict,
             inputs=[gr.Image(shape=(opt.img_size, opt.img_size),label='上传图像')],
             outputs=[gr.outputs.Label(num_top_classes=5, label='结果'), ],
             # examples=[["cheetah.jpg"]],
             )

demo.launch(server_port=7865)
