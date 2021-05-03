from torch.utils.data import Dataset


class DBSdataSet(Dataset):

    def __init__(self, data=None, indices=None):
        #finding the distance from starting time to final time
        self.dispArray = data[:]
        #self.dispArray = self.dispArray.reshape(900, 708)

        ' remember original using indices'
        if indices is not None:
            self.dispArray = self.dispArray[indices]
        # print(indices)
        #print(self.dispArray.shape)
        #print(len(self.dispArray))

    def __len__(self):
        return len(self.dispArray)

    def __getitem__(self, index):
        dispArrayIndex = self.dispArray[index]
        return index, dispArrayIndex