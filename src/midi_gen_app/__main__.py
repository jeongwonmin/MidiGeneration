import argparse

from input_output.loader.midi_loader import MidiLoader
from input_output.loader.midi_processor import MidiProcessor
from input_output.loader.midi_splitter import MidiSplitter
from input_output.loader.settings import *
from analysis.print_piano_rolls.print_piano_rolls import PrintPianoRolls

parser = argparse.ArgumentParser()
parser.add_argument('settings', help='setting path')
args = parser.parse_args()


class Main(object):
    def __init__(self, setting_path):
        conf = load_settings(setting_path)
        routine = conf["routine"]
        params = conf["params"]
        self._routine = {
            "print_piano_rolls": PrintPianoRolls,
            # "deep_learning": DeepLearning,
        }[routine](**params)
        self.path = params["data_dir"]
        self.params = params

if __name__=="__main__":
    main = Main(args.settings)
    # main._routine(MidiProcessor(MidiLoader(main.path)))

    from PIL import Image, ImageOps
    import os

    for m in MidiSplitter(MidiProcessor(MidiLoader(main.path)))():
        mel = m["splitted"]["melody"]
        chrd = m["splitted"]["chord"]
        for i in range(mel.shape[0]):
            folder = main.path
            m_img = Image.fromarray(mel[i])
            chrd_img = Image.fromarray(chrd[i])
            m_flip = ImageOps.flip(m_img)
            chrd_flip = ImageOps.flip(chrd_img)
            fname = os.path.splitext(m["file_name"])[0]
            save_folder = os.path.join(main.params["result_dir"], fname)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            m_savedir = os.path.join(save_folder, 
                "_".join(["melody", str(i)])+".tiff")
            chrd_savedir = os.path.join(save_folder, 
                "_".join(["chord", str(i)])+".tiff")
            m_flip.save(m_savedir)
            chrd_flip.save(chrd_savedir)

