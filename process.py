import torch
import numpy as np

from dataset import ProcessDataDataset

from saliency import dataset

dataset_list = ("OSIE", )
stim_location_list = ("Datasets/OSIE/data/predicting-human-gaze-beyond-pixels-master/data/stimuli", 
                        "Datasets", "Datasets")
function_list = ("DeepGaze", "PoseEstimation", )

DATASET_CONFIG = {
	'data_path' : "Datasets",
	'dataset_json': 'saliency/data/dataset.json',
	'auto_download' : True
}

def main():
    import torchvision.transforms as T

    for curr_dataset, curr_stim_location in zip(dataset_list, stim_location_list):
        ds = ProcessDataDataset(curr_dataset)

        for i in range(len(ds)):

                master_list = list()
                master_list_blurred = list()

                for blur_effect in (False, True):
                    for function in function_list:
                        result = eval(function + "(ds, i, curr_dataset, blur_effect)")
                        if blur_effect:
                            master_list_blurred.append(result)
                        else:
                            master_list.append(result)

def PoseEstimation(ds, i, curr_dataset, blurred = False):
    import cv2
    import os
    from PIL import Image
    import torchvision.transforms as T

    protoFile = "pose_cv2/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose_cv2/pose/coco/pose_iter_440000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    image = ds[i]
    if blurred:
        image = image['blurred_stimuli']
    else:
        image = image['stimuli']

    '''
    temp = Image.fromarray(image.cpu().numpy())
    temp.save(f"image{i}pose.jpeg")
    '''

    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    pe_map = torch.zeros(image.shape, dtype = torch.int)

    inWidth = image.shape[1]
    inHeight = image.shape[0]
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    out = list()
    for body_val in range(19):
        probMap = output[0, body_val, :, :]
        probMap = cv2.resize(probMap, (inWidth, inHeight))

        out.append(probMap)

    out = np.array(out)
    out = np.transpose(out, (1, 2, 0))

    '''
    temp = Image.fromarray(out[:, :, 0] * 255)
    temp = temp.convert("L")
    temp.save(f"image{i}pose_face.jpeg")
    '''

    smaller_tf = T.Compose([
        T.ToTensor(),
        T.Resize((20, 32))
    ])

    out = smaller_tf(out)

    '''
    temp = Image.fromarray(out[0].cpu().numpy() * 255)
    temp = temp.convert("L")
    temp.save(f"image{i}pose_face_small.jpeg")
    '''

    return out

def FacialDetection(ds, index, curr_dataset, blurred = False):
    import cv2
    import os
    import scipy.misc
    from PIL import Image

    import torchvision.transforms as T

    index = 1

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    image = ds[index]
    if blurred:
        image = image['blurred_stimuli']
    else:
        image = image['stimuli']

    if not torch.is_tensor(image):
        image = torch.tensor(image)
    
    fd_map = torch.zeros(image.shape, dtype = torch.int)

    temp = image.cpu().numpy()
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    '''
    im = Image.fromarray(temp)
    im.save(f"image{index}.jpeg")
    '''

    for face_no, face in enumerate(faces, start = 1):
        fd_map[face[1]:face[1] + face[3], face[0]:face[0] + face[2]] = face_no

    fd_map = fd_map * 255

    smaller_tf = T.Compose([
        T.Grayscale(),
        T.Resize((20, 32))
    ])

    fd_map = np.transpose(fd_map, (2, 0, 1))
    fd_map = smaller_tf(fd_map)

    fd_map = fd_map.type(torch.DoubleTensor) 
    fd_map /= 255

    '''
    temp = Image.fromarray(fd_map[0, :, :].cpu().numpy() * 255)
    temp = temp.convert("L")
    temp.save(f"image{index}ass.jpeg")
    '''

    return fd_map

def Segment(ds, stim_no, curr_dataset, blurred = False):
    from PIL import Image
    import requests
    import io
    import math
    import torch
    from torch import nn
    from torchvision.models import resnet50
    import torchvision.transforms as T
    import numpy as np
    torch.set_grad_enabled(False)
    import itertools

    from copy import deepcopy

    import os

    from detectron2.data import MetadataCatalog
    from detectron2.config import get_cfg

    import panopticapi
    from panopticapi.utils import id2rgb, rgb2id

    CLASSES = [
     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',   
     'Backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 
     'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
     'baseball glove', 'skateboard', 'surfboard', 'tennis racket',    
     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
     'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
     'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 
     'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 
     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] 

    coco2d2 = {}
    count = 0
    for i, c in enumerate(CLASSES):
        if c != "N/A":
            coco2d2[i] = count
            count+=1 

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    smaller_tf = T.Compose([
        T.Resize((20, 32))
    ])

    model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
    model.eval()

    im = ds[stim_no]
    if blurred:
        im = im['blurred_stimuli']
    else:
        im = im['stimuli']

    if torch.is_tensor(im):
        im = im.cpu().numpy()

    '''
    temp = Image.fromarray(im)
    temp.save(f"image{stim_no}.jpeg")
    '''

    im = Image.fromarray(im.astype('uint8'), 'RGB')

    img = transform(im).unsqueeze(0)
    out = model(img)

    scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    keep = scores > 0.85
    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy() 

    panoptic_seg_id = rgb2id(panoptic_seg)

    '''
    temp = Image.fromarray(panoptic_seg_id * 255 / np.max(panoptic_seg_id))
    temp = temp.convert("L")
    temp.save(f"image_segged{stim_no}.png")
    '''

    segments_info = deepcopy(result["segments_info"]) 

    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size 

    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
    panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))     

    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

    panoptic_seg = smaller_tf(panoptic_seg.unsqueeze(0))
    panoptic_seg = panoptic_seg.squeeze(0)

    dcb = torch.zeros([134, 20, 32], dtype = torch.uint8)

    for y in range(panoptic_seg.shape[0]):
        for x in range(panoptic_seg.shape[1]):
            curr_index = panoptic_seg[y, x]

            curr_is_thing = segments_info[curr_index]['isthing']
            curr_category_id = segments_info[curr_index]['category_id']

            curr_dcb_index = curr_category_id
            if not curr_is_thing:
                curr_dcb_index += len(meta.thing_classes)

            dcb[curr_dcb_index, y, x] = 1

    return dcb

def DeepGaze(ds, index, curr_dataset, blurred = False):
    import DeepGaze.deepgaze_pytorch
    import torchvision
    from scipy.special import logsumexp
    from scipy.ndimage import zoom
    import os
    from scipy.misc import face
    from PIL import Image

    DEVICE = "cuda"

    stim = ds[index]
    if blurred:
        stim = stim['blurred_stimuli']
    else:
        stim = stim['stimuli']

    image = stim
    i = index

    im = Image.fromarray(stim.cpu().numpy())
    im.save(f"image{index}.jpeg")

    # you can use DeepGazeI or DeepGazeIIE
    model = DeepGaze.deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
    centerbias_template = np.load('centerbias_mit1003.npy')

    # rescale to match image size
    centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    log_density_prediction = model(image_tensor, centerbias_tensor)

    print(log_density_prediction.shape)

    out = log_density_prediction.cpu().detach().numpy()

    return out

if __name__ == '__main__':
    main()