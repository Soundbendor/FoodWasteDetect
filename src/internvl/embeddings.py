"""Inference-only Jina CLIP v2 embeddings. No import-time backends."""
from contextlib import ExitStack
from pathlib import Path
import json
import shutil
import tempfile

def _load_backend(model_id, device):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer
    import torch

    text_snapshot = Path(snapshot_download('jinaai/jina-embeddings-v3',
                         local_files_only=True))
    code_snapshot = Path(snapshot_download('jinaai/xlm-roberta-flash-implementation',
                         local_files_only=True))
    # Keep the nested text loader local: it does not forward local_files_only.
    with tempfile.TemporaryDirectory(prefix='food-jina-config-') as directory:
        local = Path(directory)
        config = json.loads((text_snapshot / 'config.json').read_text())
        config['auto_map'] = {key: value.split('--')[-1] for key, value in config['auto_map'].items()}
        (local / 'config.json').write_text(json.dumps(config))
        sources = list(code_snapshot.glob('*.py'))
        if not sources:
            raise FileNotFoundError('Text encoder Python sources are not cached')
        for source in sources:
            shutil.copyfile(source, local / source.name)
        clip_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        clip_config.text_config.hf_model_name_or_path = directory
        clip_config.use_text_flash_attn = False
        clip_config.use_vision_xformers = False
        backend = AutoModel.from_pretrained(model_id, config=clip_config, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float32).to(device).eval()
    # Override lazy loaders so preprocessing also uses only cached files.
    backend.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    backend.preprocess = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    return backend


class EmbeddingModel:
    def __init__(self, model_id, device, batch_size=32, *, backend=None):
        if model_id != 'jinaai/jina-clip-v2':
            raise ValueError('This experiment requires jinaai/jina-clip-v2')
        if not device or device == 'auto' or type(batch_size) is not int or batch_size < 1:
            raise ValueError('An explicit device and positive batch size are required')
        self.batch_size = batch_size
        self._metadata = dict(model_id=model_id,
                              device=device, batch_size=batch_size, dimension=1024,
                              normalized=True, preprocessing='checkpoint encode_image; RGB',
                              text_task=None, truncate_dim=None, dtype='float32',
                              use_text_flash_attn=False, use_vision_xformers=False)
        if backend is None:
            backend = _load_backend(model_id, device)
        self.model = backend

    @property
    def dimension(self):
        return 1024

    @property
    def metadata(self):
        return dict(self._metadata)

    def _encode(self, inputs, image=False):
        import numpy as np
        from PIL import Image
        inputs = list(inputs)
        outputs = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start:start + self.batch_size]
            with ExitStack() as stack:
                if image:
                    rgb = []
                    for item in batch:
                        original = stack.enter_context(Image.open(item)) if isinstance(item, (str, Path)) else item
                        converted = original.convert('RGB')
                        stack.callback(converted.close)
                        rgb.append(converted)
                    batch = rgb
                method = self.model.encode_image if image else self.model.encode_text
                values = method(batch, batch_size=self.batch_size, truncate_dim=None)
                if hasattr(values, 'detach'):
                    values = values.detach().float().cpu().numpy()
                values = np.asarray(values, dtype=np.float32)
                if values.ndim == 1 and len(batch) == 1:
                    values = values[None, :]
                if values.shape != (len(batch), self.dimension) or not np.isfinite(values).all():
                    raise ValueError('Invalid embedding shape or nonfinite values')
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                if (norms == 0).any():
                    raise ValueError('Zero embedding')
                outputs.append(values / norms)
        return np.concatenate(outputs) if outputs else np.empty((0, self.dimension), dtype=np.float32)

    def encode_text(self, texts):
        return self._encode(texts)

    def encode_images(self, images):
        return self._encode(images, image=True)
