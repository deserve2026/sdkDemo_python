from pathlib import Path


def imread_unicode(path, flags=None):
    import cv2
    import numpy as np
    flag = cv2.IMREAD_UNCHANGED if flags is None else flags
    image = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def preprocess_image(path, size=(224, 224)):
    """Explicit future inference contract: PIL grayscale, bilinear, repeat RGB, /255."""
    import numpy as np
    from PIL import Image
    source = path if hasattr(path, 'read') else Path(path)
    with Image.open(source) as image:
        image = image.convert('L').resize(tuple(size), Image.Resampling.BILINEAR)
        gray = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.repeat(gray[None, :, :], 3, axis=0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (rgb - mean) / std
