
from io import StringIO

from .file_client import FileClient

def list_from_file(filename,
                   prefix='',
                   offset=0,
                   max_num=0,
                   encoding='utf-8',
                   file_client_args=None):
    
    cnt = 0
    item_list = []
    file_client = FileClient.infer_client(file_client_args, filename)
    with StringIO(file_client.get_text(filename, encoding)) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list

def dict_from_file(filename,
                   key_type=str,
                   encoding='utf-8',
                   file_client_args=None):
    
    mapping = {}
    file_client = FileClient.infer_client(file_client_args, filename)
    with StringIO(file_client.get_text(filename, encoding)) as f:
        for line in f:
            items = line.rstrip('\n').split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
