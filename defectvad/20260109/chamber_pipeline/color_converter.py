import os
import numpy as np
from PIL import Image
from tqdm import tqdm


class DataConverter:
    """ npz to rgb or gray"""

    def __init__(self, data_dir, primaries=None, rotation=0, Y_white=1.0):
        self.data_dir = data_dir
        self.rotation = rotation
        self.Y_white = Y_white

        # D65 defaults
        self.primaries = primaries or {
            "W": (0.3127, 0.3290),
            "R": (0.640, 0.330),
            "G": (0.300, 0.600),
            "B": (0.150, 0.060),
        }

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

    def _parse_dimming(self, filename):
        try:
            name = os.path.splitext(os.path.basename(filename))[0]
            dimming = name.split('_')[-2]
            return float(dimming)
        except Exception:
            raise ValueError(f"Cannot parse dimming from filename: {filename}")

    def _rotate(self, data):
        if self.rotation == 180:
            return np.flip(data, axis=(0, 1))
        elif self.rotation == 90:
            return np.rot90(data, k=1, axes=(0, 1))
        elif self.rotation == 270:
            return np.rot90(data, k=3, axes=(0, 1))
        return data

    def _xy_to_XYZ(self, x, y, Y=1.0):
        X = x * (Y / y)
        Z = (1 - x - y) * (Y / y)
        return np.array([X, Y, Z], dtype=np.float32)

    def _get_RGB2XYZ_matrix(self, primaries, Y_white):
        XYZ_r = self._xy_to_XYZ(*primaries["R"], Y=1.0)
        XYZ_g = self._xy_to_XYZ(*primaries["G"], Y=1.0)
        XYZ_b = self._xy_to_XYZ(*primaries["B"], Y=1.0)
        XYZ_w = self._xy_to_XYZ(*primaries["W"], Y=Y_white)
        matrix = np.stack([XYZ_r, XYZ_g, XYZ_b], axis=-1).astype(np.float32)
        scale = np.linalg.solve(matrix, XYZ_w).astype(np.float32)
        return matrix * scale[np.newaxis, :]

    def _linear_to_srgb(self, linear):
        linear = np.clip(linear, 0.0, 1.0)
        mask = linear <= 0.0031308
        srgb = np.empty_like(linear, dtype=np.float32)
        srgb[mask] = linear[mask] * 12.92
        srgb[~mask] = 1.055 * np.power(linear[~mask], 1.0/2.4) - 0.055
        return srgb

    def _XYZ_to_RGB(self, XYZ):
        M_RGB2XYZ = self._get_RGB2XYZ_matrix(self.primaries, self.Y_white)
        M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
        linear = M_XYZ2RGB @ XYZ.reshape(-1, 3).T
        linear = linear.T.reshape(XYZ.shape)
        linear = np.clip(linear, 0, 1).astype(np.float32)
        return self._linear_to_srgb(linear)

    def _normalize(self, data, vmin, vmax):
        if vmax > vmin:
            return (data - vmin) / (vmax - vmin)
        return data

    def _convert(self, output_dir, mode, normalize=True):
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in sorted(os.listdir(self.data_dir)) if f.endswith("_f16.npz")]
        if not files:
            print("No .npz files found.")
            return

        num_saved = 0
        desc = f" > Converting to {'RGB' if mode == 'rgb' else 'Grayscale'}"

        with tqdm(files, total=len(files), desc=desc, ascii=True, leave=False) as pbar:
            for filename in pbar:
                try:
                    filepath = os.path.join(self.data_dir, filename)
                    data = np.load(filepath)["data"]
                    dimming = max(self._parse_dimming(filename), 1e-6)  # zero division 방지

                    if mode == 'rgb':
                        data = self._rotate(data) / dimming
                        if normalize:
                            data = self._normalize(data, np.min(data[..., 1]), np.max(data[..., 1]))
                        image = self._XYZ_to_RGB(data)
                        image = np.clip(image * 255, 0, 255).astype('uint8')
                        image = Image.fromarray(image, 'RGB')
                        suffix = '_rgb_norm.png' if normalize else '_rgb.png'

                    elif mode == 'gray':
                        data = self._rotate(data[..., 1]) / dimming
                        if normalize:
                            data = self._normalize(data, np.min(data), np.max(data))
                        image = np.clip(image, 0, 1)
                        image = (image * 255).astype('uint8')
                        image = Image.fromarray(image, 'L')
                        suffix = '_gray_norm.png' if normalize else '_gray.png'

                    else:
                        continue

                    image_name = filename.replace('_f16.npz', suffix)
                    image.save(os.path.join(output_dir, image_name))
                    num_saved += 1

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")
                pbar.set_postfix_str(image_name)

        print(f"\n > {num_saved} {mode.upper()} images saved to {output_dir}")

    def to_rgb(self, output_dir, normalize=True):
        self._convert(output_dir, mode='rgb', normalize=normalize)

    def to_gray(self, output_dir, normalize=True):
        self._convert(output_dir, mode='gray', normalize=normalize)
