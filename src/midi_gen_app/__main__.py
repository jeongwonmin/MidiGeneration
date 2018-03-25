import argparse

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

if __name__=="__main__":
    main = Main(args.settings)
    main._routine()

