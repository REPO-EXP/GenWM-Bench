import matplotlib.pyplot as plt
import annotator.uniformer.mmcv as mmcv
import torch
from annotator.uniformer.mmcv.parallel import collate, scatter
from annotator.uniformer.mmcv.runner import load_checkpoint

from annotator.uniformer.mmseg.datasets.pipelines import Compose
from annotator.uniformer.mmseg.models import build_segmentor

def init_segmentor(config, checkpoint=None, device='cuda:0'):
    
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        .format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  
    model.to(device)
    model.eval()
    return model

class LoadImage:
    
    def __call__(self, results):
        
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def inference_segmentor(model, img):
    
    cfg = model.cfg
    device = next(model.parameters()).device  
    
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True):
    
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    
    return mmcv.bgr2rgb(img)
