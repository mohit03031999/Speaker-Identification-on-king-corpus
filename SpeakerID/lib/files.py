import re
import os.path
from collections import defaultdict

def get_files(dir, audio_extensions=('.wav', '.flac')):
    """
    Retrieve files from specified directory that match the file ext
    :param dir:  Search recursively in specified directory for audio files
    :param audio_extensions: Iterable with valid audio extensions
    :return:
    """

    file_list = []

    # Ensure directory exists
    if not os.path.exists(dir):
        raise ValueError(f"directory {dir} does not exist")

    # Traverse directory to grab all files
    for path, directory, files in os.walk(dir):
        # Only keep files that meet criteria
        for f in files:
            basename, ext = os.path.splitext(f)
            if ext.lower() in audio_extensions:
                file_list.append(os.path.join(path, f))  # matched!

    return file_list

def partition_files(files):
    """
    Given a set of files from the King corpus, partition them by speaker
    King corpus files have the following format:

    CSS_MM_T.wav
    C - is wide (w) or narrow (n) band data
    SS - Recording session number
    MM - Speaker identifier
    T - assigned topic

    :param files:
    :return:  Dictionary of lists of files keyed by speaker number.
    """

    # Regular expression for matching King corpus files
    speaker_re = re.compile(
        '[nw](?P<session>\d\d)_(?P<speaker>\d\d)_(?P<topic>\d).*')

    # Set up partition dictionary that defaults to an empty list
    partitions = defaultdict(list)

    for f in files:
        # Determine speaker number from filename
        m = speaker_re.match(os.path.basename(f))
        if m is None:
            print(f"Unable to parse speaker from {f}")
        else:
            speaker = int(m.group('speaker'))
            partitions[speaker].append(f)

    return partitions
