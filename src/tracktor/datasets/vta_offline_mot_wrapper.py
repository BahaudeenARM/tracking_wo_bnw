from torch.utils.data import Dataset

from .vta_offline_mot_sequence import VTAOfflineMotSequence


class VTAOfflineMotWrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split, dataloader):
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        dataloader -- args for the MOT_Sequence dataloader
        """

        train_sequences = []
        test_sequences = ['det']

        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        elif "last3train" == split:
            sequences = train_sequences[-3:]
        elif split in train_sequences or split in test_sequences:
            sequences = [split]
        else:
            raise NotImplementedError("Image set: {}".format(split))

        self._data = []

        for s in sequences:
            self._data.append(VTAOfflineMotSequence(seq_name=s, **dataloader))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
