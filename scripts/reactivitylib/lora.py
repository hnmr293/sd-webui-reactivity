import os
from modules import extensions, script_loading

try:
    _lora_ext = next(ex for ex in extensions.active() if ex.name == 'Lora')
    _lora_mod = script_loading.load_module(os.path.join(_lora_ext.path, 'lora.py'))
except:
    print('[WARN] MatView: Lora is not activated!')
    _lora_mod = None


def reload_loras():
    if _lora_mod is None:
        raise ValueError('Lora is inactive. See `Extensions` tab.')
    _lora_mod.list_available_loras()
    loras = [''] + list(_lora_mod.available_loras.keys())
    return loras

def available_loras():
    return (
        ([''] + list(_lora_mod.available_loras.keys()))
        if _lora_mod is not None
        else ['']
    )

def lora_path(name: str):
    if _lora_mod is None:
        raise ValueError('Lora is inactive. See `Extensions` tab.')
    filename = _lora_mod.available_loras[name].filename
    return filename

def load_lora(name: str):
    if _lora_mod is None:
        raise ValueError('Lora is inactive. See `Extensions` tab.')
    return _lora_mod.load_lora(name, lora_path(name))
