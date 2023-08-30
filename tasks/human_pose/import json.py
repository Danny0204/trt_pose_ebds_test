import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
import ipywidgets


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))
WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
t0 = time.time()
torch.cuda.current_stream().synchronize()

#for i in range(50):
#    y = model_trt(data)


torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# 其他代码...

# 修改execute函数
def execute(change):
    images = change['new']  # 接收多个图像
    data_batch = []  # 存储图像批次的列表

    for image in images:
        data = preprocess(image)
        data_batch.append(data)  # 将图像数据添加到批次列表中

    data_batch = torch.stack(data_batch)  # 将图像数据批次堆叠为张量
    cmap_batch, paf_batch = model_trt(data_batch)  # 批次推理

    # 处理推理结果
    for i in range(len(images)):
        cmap, paf = cmap_batch[i], paf_batch[i]
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        draw_objects(images[i], counts, objects, peaks)
        image_w.value = bgr8_to_jpeg(images[i][:, ::-1, :])

# 其他代码...

camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
# camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)

camera.running = True

image_w = ipywidgets.Image(format='jpeg')

display(image_w)

# 监听相机图像变化并执行推理
camera.observe(execute, names='value')
camera.unobserve_all()
