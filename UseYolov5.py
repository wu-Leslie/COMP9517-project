import torch
import cv2
import os


Path = os.getcwd()
# Green for class 0(penguin) and Orange for class 1(Turtle)
class_color = {
    0: (0, 255, 0),
    1: (0, 165, 255),
}

def detect_and_label_image_Yolov5(img):
    # Load the trained YOLO v5 model
    weightpath = os.path.join(Path, 'frontendModel','Yolov5_best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weightpath)
   
    # Get predictions
    # img = torch.from_numpy(img)
    results = model(img)

    for img_idx, (img, pred) in enumerate(zip(results.ims, results.pred)):
        # print(f'pred = {pred}') 
        for det in pred:
            x_min, y_min, x_max, y_max, confidence, class_idx = det.tolist()
           
            # Make sure the class index is within the range of available classes and confidence is high enough
            if int(class_idx) < len(model.names) and confidence > 0.5: 
                
                label = model.names[int(class_idx)]
                x1 = int(x_min)
                x2 = int(x_max)
                y1 = int(y_min)
                y2 = int(y_max)
                color = class_color[int(class_idx)]
                # print(f"x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")
                
                # Decorate the image with bounding box and class label
                Img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                Img = cv2.putText(Img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    print("returnImage")
    return Img

if __name__ == '__main__':
    # Load an image
    imgPath = os.path.join(Path, 'frontendModel','testImages', 'Test3.jpg')
    img = cv2.imread(imgPath)
    # Show the decorated image
    img = detect_and_label_image_Yolov5(img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)

