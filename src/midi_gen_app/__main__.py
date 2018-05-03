import argparse
import shutil
import os

from input_output.loader.midi_loader import MidiLoader
from input_output.loader.midi_processor import MidiProcessor
from input_output.loader.midi_splitter import MidiSplitter
from input_output.loader.batch_loader import BatchLoader
from input_output.loader.settings import *
from analysis.print_piano_rolls.print_piano_rolls import PrintPianoRolls
from analysis.deep_learning.deep_learning import DeepLearning

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
            "deep_learning": DeepLearning,
        }[routine](**params)
        self.path = params["data_dir"]
        self.params = params
        if routine == "print_piano_rolls":
            self._loader = MidiSplitter(MidiProcessor(MidiLoader(self.path))) 
        setting_fname = os.path.basename(setting_path)
        result_file = os.path.join(self._routine._path, setting_fname)
        shutil.copyfile(setting_path, result_file)

if __name__=="__main__":
    main = Main(args.settings)
    l = main._loader if main._routine.__class__.__name__=="PrintPianoRolls" \
        else None
    main._routine(l)

