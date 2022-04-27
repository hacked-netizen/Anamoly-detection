from pyimagesearch.features import quantify_image
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained anomaly detection model")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

print("[INFO] loading anomaly detection model...")
model = pickle.loads(open(args["model"], "rb").read())

image = cv2.imread(args["image"])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
features = quantify_image(hsv, bins=(3, 3, 3))

preds = model.predict([features])[0]
label = "anomaly" if preds == -1 else "normal"
color = (0, 0, 255) if preds == -1 else (0, 255, 0)

cv2.putText(image, label, (10,  25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
