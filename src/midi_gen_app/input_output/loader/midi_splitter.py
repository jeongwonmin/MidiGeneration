from .midi_processor import MidiProcessor


class MidiSplitter(object):
    def __init__(self, processor):
        self._processor = processor

    def __call__(self):
        for p in self._processor():
            yield p
