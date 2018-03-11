import os
from datetime import datetime

from PIL import Image, ImageOps


class PrintPianoRolls(object):
    def __init__(self, **params):
        self._data_dir = params['data_dir']
        result_dir = params['result_dir']
        subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._result_dir = os.path.join(result_dir, subdir)
        if not os.path.exists(self._result_dir):
            os.makedirs(self._result_dir)

    def __call__(self, loader):
        if hasattr(loader, '__name__') and loader.__name__ == "MidiLoader":
            loader = loader(self._data_dir)
        for l in loader():
            genre_name = l['genre']
            folder_name = os.path.splitext(l['file_name'])[0]
            result_dir = os.path.join(
                self._result_dir, genre_name, folder_name
            )
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            for k, i in l['piano_roll'].items():
                filename = "_".join([folder_name, k])+".tiff"
                img = Image.fromarray(i)
                im_flip = ImageOps.flip(img)
                save_path = os.path.join(result_dir, filename)
                im_flip.save(save_path)
