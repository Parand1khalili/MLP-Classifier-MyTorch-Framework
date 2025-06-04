import numpy as np
from mytorch import Tensor
from PIL import Image
from random import shuffle
from typing import List, Tuple

# data count
TRAIN = 10000
TEST = 1000

class DataLoader:

    def __init__(self, train_addr:str, test_addr:str) -> None:
        # train
        self.train_addr = train_addr
        self.train = [] # iterate this field for train
        # test
        self.test_addr = test_addr
        self.test = []  # iterate this field for test

    def load(self, train_batch_size:int=500, test_batch_size:int=100):
        print("loading train...")
        train_data = []
        for i in range(TRAIN):
            label = (int) (i/1000)
            index = (int) ((i%1000) + 1) 
            addr = self.train_addr + '/' + label.__str__() + ' (' + index.__str__() + ')' + '.jpg'
            img = Image.open(addr, mode='r')
            train_data.append((np.array(img), label))
        
        print("loading test...")
        test_data = []
        for i in range(TEST):
            label = (int) (i/100)
            index = (int) ((i%100) + 1) 
            addr = self.test_addr + '/' + label.__str__() + ' (' + index.__str__() + ')' + '.jpg'
            img = Image.open(addr, mode='r')
            test_data.append((np.array(img), label))
        
        print('processing...')
        shuffle(train_data)
        shuffle(test_data)

        for i in range((int)(TRAIN/train_batch_size)):
            batch_data = []
            batch_label = []
            for j in range(train_batch_size):
                index = i * train_batch_size + j
                batch_data.append(train_data[index][0])
                batch_label.append(train_data[index][1])
            self.train.append((Tensor(np.array(batch_data)), Tensor(np.array(batch_label))))
        
        for i in range((int)(TEST/test_batch_size)):
            batch_data = []
            batch_label = []
            for j in range(test_batch_size):
                index = i * test_batch_size + j
                batch_data.append(test_data[index][0])
                batch_label.append(test_data[index][1])
            self.test.append((Tensor(np.array(batch_data)), Tensor(np.array(batch_label))))

    def getTrain(self)->List[Tuple[Tensor, Tensor]]:
        return self.train
    
    def getTest(self)->List[Tuple[Tensor, Tensor]]:
        return self.test
