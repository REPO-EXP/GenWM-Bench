
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['bbox_overlaps'])

def bbox_overlaps(bboxes1, bboxes2, mode='iou', aligned=False, offset=0):
    
    mode_dict = {'iou': 0, 'iof': 1}
    assert mode in mode_dict.keys()
    mode_flag = mode_dict[mode]
    
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)
    assert offset == 1 or offset == 0

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if aligned else bboxes1.new(rows, cols)

    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros((rows, cols))
    ext_module.bbox_overlaps(
        bboxes1, bboxes2, ious, mode=mode_flag, aligned=aligned, offset=offset)
    return ious
