import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
import torch
import time

im = Image.open('test.jpg').convert('RGB')
im = im.resize((640, 640))
im_data = ToTensor()(im)[None]
print(im_data.shape)
size = torch.tensor([[640, 640]])
sess = ort.InferenceSession("model.onnx")

start_time = time.time()
output = sess.run(
    # output_names=['labels', 'boxes', 'scores'],
    output_names=None,
    input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

labels, boxes, scores = output
print("labels shape = ",labels.shape)
print("boxes shape = ",boxes.shape)
print("scores shape = ",scores.shape)
draw = ImageDraw.Draw(im)
thrh = 0.2

for i in range(im_data.shape[0]):

    scr = scores[i]
    lab = labels[i][scr > thrh]
    box = boxes[i][scr > thrh]

    print(i, sum(scr > thrh))

    for b in box:
        draw.rectangle(list(b), outline='DeepSkyBlue',width=2)
        # draw.text((b[0], b[1]), text=str(scr[i]), fill='cyan')

im.save('output.jpg')