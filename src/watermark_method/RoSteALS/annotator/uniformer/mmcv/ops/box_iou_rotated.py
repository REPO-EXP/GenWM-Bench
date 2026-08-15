
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['box_iou_rotated'])

def box_iou_rotated(bboxes1, bboxes2, mode='iou', aligned=False):
    
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros((rows * cols))
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    ext_module.box_iou_rotated(
        bboxes1, bboxes2, ious, mode_flag=mode_flag, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious
