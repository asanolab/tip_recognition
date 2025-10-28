from PIL import Image
import torch
from torchvision import datasets, transforms
import os
from model import SimpleCNN
import torch.nn.functional as F
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def predict(image_path, model_path):
    # Load the trained model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and preprocess the image
    if isinstance(image_path, str):
        image = Image.open(image_path)

        image = transform(image).unsqueeze(0).to(device)
    else:
        image = image_path
        image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)

    # Make prediction

    output = model(image)

    softmax_output = F.softmax(output, dim=1)
    confidence, predicted = torch.max(softmax_output, 1)

    classes = ['hole', 'tip']

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()
    probabilities = softmax_output.cpu().detach().numpy()

    return predicted_class, confidence_score, probabilities


def test_image_dir(dir_path, model_path):
    imgs = os.listdir(dir_path)
    results = []
    for img in imgs:
        img_path = os.path.join(dir_path, img)
        pred_class, conf, probs = predict(img_path, model_path)
        result = [pred_class, conf, probs]
        results.append(result)
        print(f'Image predicted as :{pred_class}, '
              f'Confidence: {conf}, '
              f'Class probabilities: {probs}')
    return results


if __name__ == "__main__":

    # image_path = 'dataset/test/tip'
    # image_path='tip_image.jpg'
    image_path='camera_tip_image'
    model_path = '1734448311.8315692/best.pth'
    if os.path.isfile(image_path):
        pred_class, conf, probs = predict(image_path, model_path)
        print(f'Image predicted as :{pred_class}, '
              f'Confidence: {conf}, '
              f'Class probabilities: {probs}')

    else:
        test_image_dir(image_path, model_path)
