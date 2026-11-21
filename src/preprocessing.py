"""Explicit, serializable input policies for reproducible scene checkpoints."""

from PIL import Image
from torchvision import transforms


class Letterbox:
    """Keep the entire image: longest edge -> size, then centered mean-color padding."""

    def __init__(self, size=224, fill=(124, 116, 104)):
        self.size = size
        self.fill = tuple(fill)

    def __call__(self, image):
        image = image.convert('RGB')
        width, height = image.size
        longest = max(width, height)
        target = (max(1, round(width * self.size / longest)),
                  max(1, round(height * self.size / longest)))
        resized = image.resize(target, Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - target[0]) // 2, (self.size - target[1]) // 2))
        return canvas

    def __repr__(self):
        return f'Letterbox(size={self.size}, fill={self.fill}, interpolation=bilinear)'


def make_preprocessing(policy, weights):
    original = weights.transforms()
    spec = {'policy': policy, 'mean': list(original.mean), 'std': list(original.std)}
    if policy == 'pretrained_weights_transforms':
        spec.update(resize_shorter_edge=list(original.resize_size),
                    center_crop=list(original.crop_size),
                    interpolation=original.interpolation.value, antialias=original.antialias)
        return original, spec
    if policy == 'letterbox_224_imagenet_mean':
        spec.update(size=224, fill=[124, 116, 104], resize_longest_edge=224,
                    interpolation='bilinear', rounding='python_round_half_even',
                    padding='center_floor_left_top')
        transform = transforms.Compose([
            Letterbox(spec['size'], spec['fill']), transforms.ToTensor(),
            transforms.Normalize(mean=original.mean, std=original.std),
        ])
        return transform, spec
    raise ValueError(f'Unknown preprocessing policy: {policy}')


def preprocessing_from_checkpoint(checkpoint, weights):
    saved = checkpoint.get('preprocessing')
    if saved is None:
        policy = checkpoint.get('config', {}).get('model', {}).get(
            'preprocessing', 'pretrained_weights_transforms')
        if policy != 'pretrained_weights_transforms':
            raise ValueError('Nonstandard preprocessing requires an explicit saved specification')
        return make_preprocessing(policy, weights)[0]
    transform, expected = make_preprocessing(saved['policy'], weights)
    if saved != expected:
        raise ValueError('Saved preprocessing specification differs from supported implementation')
    return transform
