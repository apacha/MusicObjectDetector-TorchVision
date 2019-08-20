import numpy
import torch
from PIL import Image
from PIL.ImageDraw import ImageDraw

from MuscimaPpDataset import MuscimaPpDataset
from torchvision_finetuning_instance_segmentation import get_instance_segmentation_model, get_transform

num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load("model-8.pth"))
model.to(device)

dataset_test = MuscimaPpDataset('muscima_pp/v2.0/data/images', "muscima_pp_masks",
                                get_transform(train=False), False, 3)

img, _ = dataset_test[0]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

"""Printing the prediction shows that we have a list of dictionaries. Each element of the list corresponds to a different image. As we have a single image, there is a single dictionary in the list.
The dictionary contains the predictions for the image we passed. In this case, we can see that it contains `boxes`, `labels`, `masks` and `scores` as fields.
"""

prediction

"""Let's inspect the image and the predicted segmentation masks.

For that, we need to convert the image, which has been rescaled to 0-1 and had the channels flipped so that we have it in `[C, H, W]` format.
"""

# predicted_image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
# predicted_image.save("predicted_image.png")

"""And let's now visualize the top predicted segmentation mask. The masks are predicted as `[N, 1, H, W]`, where `N` is the number of predictions, and are probability maps between 0-1."""

image = numpy.zeros((img.shape[1],img.shape[2]), dtype=numpy.uint8)
for i in range(len(prediction[0]['masks'])):
    if prediction[0]['scores'][i] < 0.9:
        continue
    mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
    image += mask

predicted_mask = Image.fromarray(image, mode="L")
predicted_mask = predicted_mask.convert(mode="RGB")

image_draw = ImageDraw(predicted_mask)
for i in range(len(prediction[0]['boxes'])):
    if prediction[0]['scores'][i] < 0.9:
        continue
    box = prediction[0]['boxes'][i].cpu().numpy()
    image_draw.rectangle(list(box), outline="Red", width=2)

predicted_mask.save("predicted_mask.png")