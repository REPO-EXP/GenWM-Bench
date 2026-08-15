
import os.path as osp
import pickle
import shutil
import tempfile
import time

import torch
import torch.distributed as dist

import annotator.uniformer.mmcv as mmcv
from annotator.uniformer.mmcv.runner import get_dist_info

def single_gpu_test(model, data_loader):
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar.completed
            for _ in range(batch_size_all):
                prog_bar.update()

    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

def collect_results_cpu(result_part, size, tmpdir=None):
    
    rank, world_size = get_dist_info()
    
    if tmpdir is None:
        MAX_LEN = 512
        
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    
    if rank != 0:
        return None
    else:
        
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            
            if part_result:
                part_list.append(part_result)
        
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        
        ordered_results = ordered_results[:size]
        
        shutil.rmtree(tmpdir)
        return ordered_results

def collect_results_gpu(result_part, size):
    
    rank, world_size = get_dist_info()
    
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            
            if part_result:
                part_list.append(part_result)
        
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        
        ordered_results = ordered_results[:size]
        return ordered_results
