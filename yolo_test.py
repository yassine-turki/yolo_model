import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
from PIL import Image
import numpy as np

# Create your workflow for YOLO inference
wf = Workflow()

# Add the YOLOv9 algorithm to your workflow
yolov9 = wf.add_task(name="infer_yolo_v9", auto_connect=True)

# Set parameters for YOLOv9
yolov9.set_parameters({
    "model_weight_file": "/users/eleves-a/2021/yassine.turki/cardamage/15-08-2024T10h01m42s/weights/best.pt",
    "conf_thres": "0.3",
    "iou_thres": "0.25"
})

# Run the workflow on your image
wf.run_on(path="/users/eleves-a/2021/yassine.turki/cardamage/yolo/test/doorouter-sep28-28-_jpg.rf.3c18c9c6212076ebdcd0c292f38e54d6.jpg")

# Get the object detection image output with graphics (bounding boxes, etc.)
img_bbox = yolov9.get_image_with_graphics()

# Convert the image to a format suitable for displaying with matplotlib
img_bbox = Image.fromarray(np.uint8(img_bbox))


# Save the image to a file
output_path = "/users/eleves-a/2021/yassine.turki/cardamage/detected_image.png"
img_bbox.save(output_path)

print(f"Image saved to {output_path}")
