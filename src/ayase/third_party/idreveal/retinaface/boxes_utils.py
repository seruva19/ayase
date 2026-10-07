import torch
from torchvision.ops import roi_align, roi_pool

def convert_to_square_margin(bboxA, margin_zoom = 0):
    bboxA = torch.clone(bboxA).int()
    h = bboxA[:, 3] - bboxA[:, 1] + 1
    w = bboxA[:, 2] - bboxA[:, 0] + 1
    s = torch.max(w, h)
    s = (s + margin_zoom*s // 100).int()
    bboxA[:,0] = (bboxA[:, 2] + bboxA[:, 0] + 1)//2 - s//2
    bboxA[:,1] = (bboxA[:, 3] + bboxA[:, 1] + 1)//2 - s//2
    bboxA[:,2] = bboxA[:,0] + s - 1
    bboxA[:,3] = bboxA[:,1] + s - 1 
    return bboxA

def recovery_box_margin(dim, margin_zoom = 0, times = 2):
    _, _, h2, w2 = dim
    h = h2//times
    w = w2//times
    s = max(w, h)    
    s = (s + margin_zoom*s // 100)
    bboxA0 = (w2 - s)//2
    bboxA1 = (h2 - s)//2
    bboxA2 = bboxA0 + s - 1
    bboxA3 = bboxA1 + s - 1 
    return torch.tensor([bboxA0,bboxA1,bboxA2,bboxA3])

def roi_linear(imgs, image_inds, boxes, outshape):
    boxes = boxes.float() + torch.tensor([-0.5,-0.5, 0.5,0.5], device=boxes.device, dtype=torch.float32)
    b = torch.cat((image_inds[:,None].float(), boxes),-1).to(imgs.device)
    return roi_align(imgs, b, outshape)

def roi_nearest(imgs, image_inds, boxes, outshape):
    b = torch.cat((image_inds[:,None].float(), boxes.float()),-1).to(imgs.device)
    return roi_pool(imgs, b, outshape)

def roi_nores(imgs, image_inds, boxes):
    return [ imgs[a,:,b[1]:b[3]+1, b[0]:b[2]+1] if a>=0 else None for a,b in zip(image_inds, boxes)]

def apply_box_linear(x, box, outshape):
    num_frame = x.shape[0]
    boxes = box[None,:].repeat(num_frame,1).to(x.device)
    image_inds = torch.arange(0, num_frame, dtype=x.dtype, device=x.device)
    face = roi_linear(x, image_inds, boxes, outshape)
    return face
    
def apply_box_nearest(x, box, outshape):
    num_frame = x.shape[0]
    boxes = box[None,:].repeat(num_frame,1).to(x.device)
    image_inds = torch.arange(0, num_frame, dtype=x.dtype, device=x.device)
    face = roi_nearest(x, image_inds, boxes, outshape)
    return face

def apply_box_nores(x, box, padding=True):
    if padding:
        return apply_box_nearest(x, box, outshape=(box[3]-box[1]+1, box[2]-box[0]+1))
    else:
        box = box.int()
        return x[:,:, box[1]:box[3]+1, box[0]:box[2]+1]

def area(boxes):
    return (boxes[:,2]-boxes[:,0]+1).clamp(min=0) * (boxes[:,3]-boxes[:,1]+1).clamp(min=0)
    
def iou(boxes1, boxes2):
    area_int = area(torch.cat((torch.max(boxes1[:,:2],boxes2[:,:2]),torch.min(boxes1[:,2:],boxes2[:,2:])),-1))
    return area_int / (area(boxes1)+area(boxes2)-area_int)
    

def points2poses(x):
    amp = (x[..., 0] - x[..., 2])**2 + \
          (x[..., 1] - x[..., 3])**2
    pro = (x[..., 4] - (x[..., 0] + x[..., 2]) / 2) * (x[..., 0] - x[..., 2]) + \
          (x[..., 5] - (x[..., 1] + x[..., 3]) / 2) * (x[..., 1] - x[..., 3])
    return pro/amp