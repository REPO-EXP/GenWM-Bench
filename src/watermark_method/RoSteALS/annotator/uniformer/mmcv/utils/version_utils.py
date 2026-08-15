
import os
import subprocess
import warnings

from packaging.version import parse

def digit_version(version_str: str, length: int = 4):
    
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              )
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)

def _minimal_ext_cmd(cmd):
    
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out

def get_git_hash(fallback='unknown', digits=None):
    
    if digits is not None and not isinstance(digits, int):
        raise TypeError('digits must be None or an integer')

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
        if digits is not None:
            sha = sha[:digits]
    except OSError:
        sha = fallback

    return sha
