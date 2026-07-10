"""Cross-frame mask/object bookkeeping for VMBench Temporal Coherence.

Vendored from IDEA-Research/Grounded-SAM-2 (``utils/mask_dictionary_model.py``,
Apache-2.0). ``MaskDictionaryModel`` holds one frame's per-object masks and
assigns *consistent object ids across frames* by IoU matching: a new frame's
mask inherits an existing object's id when their IoU exceeds a threshold, else it
starts a new object (bumping ``objects_count``). Only the pieces the TCS module
needs are kept (frame annotation, IoU id-assignment, IoU helper).
"""

from dataclasses import dataclass, field

import torch


@dataclass
class ObjectInfo:
    instance_id: int = 0
    mask: any = None
    class_name: str = ""
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    logit: float = 0.0

    def get_mask(self):
        return self.mask

    def get_id(self):
        return self.instance_id

    def update_box(self):
        nonzero_indices = torch.nonzero(self.mask)

        if nonzero_indices.size(0) == 0:
            return []

        y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
        y_max, x_max = torch.max(nonzero_indices, dim=0)[0]

        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]


@dataclass
class MaskDictionaryModel:
    mask_name: str = ""
    mask_height: int = 1080
    mask_width: int = 1920
    promote_type: str = "mask"
    labels: dict = field(default_factory=dict)

    def add_new_frame_annotation(self, mask_list, box_list, label_list, background_value=0):
        mask_img = torch.zeros(mask_list.shape[-2:])
        anno_2d = {}
        for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
            final_index = background_value + idx + 1

            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            mask_img[mask == True] = final_index
            name = label
            box = box
            new_annotation = ObjectInfo(instance_id=final_index, mask=mask, class_name=name,
                                        x1=box[0], y1=box[1], x2=box[2], y2=box[3])
            anno_2d[final_index] = new_annotation

        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d

    def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
        updated_masks = {}

        for seg_obj_id, seg_mask in self.labels.items():
            flag = 0
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue

            for object_id, object_info in tracking_annotation_dict.labels.items():
                iou = self.calculate_iou(seg_mask.mask, object_info.mask)
                if iou > iou_threshold:
                    flag = object_info.instance_id
                    new_mask_copy.mask = seg_mask.mask
                    new_mask_copy.instance_id = object_info.instance_id
                    new_mask_copy.class_name = seg_mask.class_name
                    break

            if not flag:
                objects_count += 1
                flag = objects_count
                new_mask_copy.instance_id = objects_count
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.class_name = seg_mask.class_name
            updated_masks[flag] = new_mask_copy
        self.labels = updated_masks
        return objects_count

    def get_target_class_name(self, instance_id):
        return self.labels[instance_id].class_name

    @staticmethod
    def calculate_iou(mask1, mask2):
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)

        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection

        iou = intersection / union
        return iou
