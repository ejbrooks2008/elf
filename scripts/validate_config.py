"""Quick validation that our Flux LoRA config matches ai-toolkit expectations."""
import sys, os, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai-toolkit'))

config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train', 'flux_lora_elf.yaml')

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

p = cfg['config']['process'][0]
errors = []

# 1. Check structure
if p.get('type') != 'sd_trainer':
    errors.append(f"type should be 'sd_trainer', got: {p.get('type')}")

# 2. low_vram should be in model, NOT in train
if 'low_vram' in p.get('train', {}):
    errors.append("low_vram found in train: section — should be in model: section")
if not p.get('model', {}).get('low_vram', False):
    errors.append("low_vram not set in model: section")

# 3. network_kwargs / target_modules should NOT exist in network
net = p.get('network', {})
if 'network_kwargs' in net and net['network_kwargs']:
    # ai-toolkit does accept network_kwargs as a passthrough dict,
    # but target_modules is not a valid passthrough for its LoRA
    nk = net['network_kwargs']
    if 'target_modules' in nk:
        errors.append("network_kwargs.target_modules is not valid in ai-toolkit LoRA config")

# 4. logging block should not exist at process level
if 'logging' in p:
    errors.append("logging: block is not a standard ai-toolkit process key")

# 5. trigger_word should be at process level
if 'trigger_word' not in p:
    errors.append("trigger_word missing at process level")

# 6. datasets checks
ds = p.get('datasets', [{}])[0]
if 'shuffle_tokens' not in ds:
    errors.append("shuffle_tokens missing in dataset config")
if 'cache_latents_to_disk' not in ds:
    errors.append("cache_latents_to_disk missing in dataset config")

# 7. Required fields
required_model = ['name_or_path', 'is_flux', 'quantize']
for k in required_model:
    if k not in p.get('model', {}):
        errors.append(f"model.{k} missing")

required_train = ['batch_size', 'steps', 'lr', 'optimizer', 'dtype', 'noise_scheduler']
for k in required_train:
    if k not in p.get('train', {}):
        errors.append(f"train.{k} missing")

# 8. Validate against actual ai-toolkit config classes
try:
    from toolkit.config_modules import (
        ModelConfig, TrainConfig, NetworkConfig, DatasetConfig, SaveConfig, SampleConfig
    )
    
    mc = ModelConfig(**p['model'])
    print(f"  ModelConfig OK — low_vram={mc.low_vram}, quantize={mc.quantize}, is_flux={mc.is_flux}")
    
    tc = TrainConfig(**p['train'])
    print(f"  TrainConfig OK — steps={tc.steps}, lr={tc.lr}, optimizer={tc.optimizer}, scheduler={tc.lr_scheduler}")
    
    nc = NetworkConfig(**p['network'])
    print(f"  NetworkConfig OK — type={nc.type}, rank={nc.rank}, linear_alpha={nc.linear_alpha}")
    
    dc = DatasetConfig(**ds)
    print(f"  DatasetConfig OK — folder={dc.folder_path}, resolution={dc.resolution}, shuffle={dc.shuffle_tokens}")
    
    sc = SaveConfig(**p['save'])
    print(f"  SaveConfig OK — dtype={sc.dtype}, save_every={sc.save_every}")
    
    samp = SampleConfig(**p['sample'])
    print(f"  SampleConfig OK — sampler={samp.sampler}, sample_every={samp.sample_every}")
    
except Exception as e:
    errors.append(f"Config class instantiation failed: {e}")

if errors:
    print("\n✗ VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n✓ All config validations passed!")
    sys.exit(0)
