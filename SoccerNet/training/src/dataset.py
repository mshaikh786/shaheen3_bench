from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time


from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from config.classes import EVENT_DICTIONARY_V1, EVENT_DICTIONARY_V2, K_V1, K_V2

from preprocessing import oneHotToShifts, getTimestampTargets, getChunks_anchors, getChunks_anchors_old



class SoccerNetClips(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", version=1, 
                framerate=2, chunk_size=240, receptive_field=80, chunks_per_epoch=6000):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.version = version
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
            self.K_parameters = K_V1*framerate
            self.num_detections =5
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"
            self.K_parameters = K_V2*framerate  
            self.num_detections =15

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)


        logging.info("Pre-compute clips")

        clip_feats = []
        clip_labels = []

        self.game_feats = list()
        self.game_labels = list()
        self.game_anchors = list()
        for i in np.arange(self.num_classes+1):
            self.game_anchors.append(list())

        game_counter = 0
        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]



                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = 1

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = 1

            shift_half1 = oneHotToShifts(label_half1, self.K_parameters.cpu().numpy())
            shift_half2 = oneHotToShifts(label_half2, self.K_parameters.cpu().numpy())

            anchors_half1 = getChunks_anchors(shift_half1, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

            game_counter = game_counter+1

            anchors_half2 = getChunks_anchors(shift_half2, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

            game_counter = game_counter+1



            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(shift_half1)
            self.game_labels.append(shift_half2)
            for anchor in anchors_half1:
                self.game_anchors[anchor[2]].append(anchor)
            for anchor in anchors_half2:
                self.game_anchors[anchor[2]].append(anchor)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """

        # Retrieve the game index and the anchor
        class_selection = random.randint(0, self.num_classes)
        event_selection = random.randint(0, len(self.game_anchors[class_selection])-1)
        game_index = self.game_anchors[class_selection][event_selection][0]
        anchor = self.game_anchors[class_selection][event_selection][1]

        # Compute the shift for event chunks
        if class_selection < self.num_classes:
            shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
            start = anchor + shift
        # Compute the shift for non-event chunks
        else:
            start = random.randint(anchor[0], anchor[1]-self.chunk_size)
        if start < 0:
            start = 0
        if start+self.chunk_size >= self.game_feats[game_index].shape[0]:
            start = self.game_feats[game_index].shape[0]-self.chunk_size-1

        # Extract the clips
        clip_feat = self.game_feats[game_index][start:start+self.chunk_size]
        clip_labels = self.game_labels[game_index][start:start+self.chunk_size]

        # Put loss to zero outside receptive field
        clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
        clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

        # Get the spotting target
        clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]


        return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets)

    def __len__(self):
        return self.chunks_per_epoch


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="test", version=1, 
                framerate=2, chunk_size=240, receptive_field=80):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate = framerate
        self.version = version
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
            self.K_parameters = K_V1*framerate
            self.num_detections =5
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"
            self.K_parameters = K_V2*framerate		
            self.num_detections =15

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))

        # Load labels
        labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        for annotation in labels["annotations"]:

            time = annotation["gameTime"]
            event = annotation["label"]

            half = int(time[0])

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            frame = self.framerate * ( seconds + 60 * minutes ) 

            if event not in self.dict_event:
                continue
            label = self.dict_event[event]

            value = 1
            if "visibility" in annotation.keys():
                if annotation["visibility"] == "not shown":
                    value = -1

            if half == 1:
                frame = min(frame, feat_half1.shape[0]-1)
                label_half1[frame][label] = value

            if half == 2:
                frame = min(frame, feat_half2.shape[0]-1)
                label_half2[frame][label] = value

        def feats2clip(feats, stride, clip_length, padding = "replicate_last"):

            if padding =="zeropad":
                print("beforepadding", feats.shape)
                pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
                print("pad need to be", clip_length-pad)
                m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
                feats = m(feats)
                print("afterpadding", feats.shape)
                # nn.ZeroPad2d(2)

            idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx+i)
            idx = torch.stack(idxs, dim=1)

            if padding=="replicate_last":
                idx = idx.clamp(0, feats.shape[0]-1)
                # Not replicate last, but take the clip closest to the end of the video
                idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

            return feats[idx,:]
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)
                                  
        return feat_half1, feat_half2, torch.from_numpy(label_half1), torch.from_numpy(label_half2)

    def __len__(self):
        return len(self.listGames)


class SoccerNetClipsOld(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", version=1, 
                framerate=2, chunk_size=240, receptive_field=80):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.version = version
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
            self.K_parameters = K_V1*framerate
            self.num_detections =5
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"
            self.K_parameters = K_V2*framerate  
            self.num_detections =15

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)


        logging.info("Pre-compute clips")

        clip_feats = []
        clip_labels = []

        self.game_feats = list()
        self.game_labels = list()
        self.game_anchors = list()
        self.game_anchors_negatives = list()
        game_counter = 0

        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                if half == 1:
                    label_half1[frame][label] = 1

                if half == 2:
                    label_half2[frame][label] = 1

            shift_half1 = oneHotToShifts(label_half1, self.K_parameters.cpu().numpy())
            shift_half2 = oneHotToShifts(label_half2, self.K_parameters.cpu().numpy())

            anchors_half1, anchors_negative_half_1 = getChunks_anchors_old(shift_half1, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

            game_counter = game_counter+1

            anchors_half2, anchors_negative_half_2 = getChunks_anchors_old(shift_half2, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

            game_counter = game_counter+1



            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(shift_half1)
            self.game_labels.append(shift_half2)
            self.game_anchors.append(anchors_half1)
            self.game_anchors.append(anchors_half2)
            self.game_anchors_negatives.append(anchors_negative_half_1)
            self.game_anchors_negatives.append(anchors_negative_half_2)




    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """

        # Retrieve the game index and the anchor

        clips_feat = list()
        clips_labels = list()
        clips_targets = list()


        for i in np.arange(len(self.game_anchors[index])):
            anchor = self.game_anchors[index][i][1]

            # Compute the shift
            shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
            start = anchor + shift
            if start < 0:
                start = 0
            if start+self.chunk_size >= self.game_feats[index].shape[0]:
                start = self.game_feats[index].shape[0]-self.chunk_size-1

            # Extract the clips
            clip_feat = self.game_feats[index][start:start+self.chunk_size]
            clip_labels = self.game_labels[index][start:start+self.chunk_size]

            # Put loss to zero outside receptive field
            clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
            clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

            # Get the spotting target
            clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]

            clips_feat.append(clip_feat)
            clips_labels.append(clip_labels)
            clips_targets.append(clip_targets)

        number_of_negative_chunks = np.floor(len(self.game_anchors[index])/self.num_classes)+1
        for i in np.arange(number_of_negative_chunks):

            if len(self.game_anchors_negatives) == 0:
                break
            selection = random.randint(0, len(self.game_anchors_negatives[index])-1)
            anchor = self.game_anchors_negatives[index][selection][1]

            # Compute the shift
            start = random.randint(anchor[0], anchor[1]-self.chunk_size)
            if start < 0:
                start = 0
            if start+self.chunk_size >= self.game_feats[index].shape[0]:
                start = self.game_feats[index].shape[0]-self.chunk_size-1

            # Extract the clips
            clip_feat = self.game_feats[index][start:start+self.chunk_size]
            clip_labels = self.game_labels[index][start:start+self.chunk_size]

            # Put loss to zero outside receptive field
            clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
            clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

            # Get the spotting target
            clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]

            clips_feat.append(clip_feat)
            clips_labels.append(clip_labels)
            clips_targets.append(clip_targets)

        clip_feat = np.array(clips_feat)
        clip_labels = np.array(clips_labels)
        clip_targets = np.array(clips_targets)

        return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets)

    def __len__(self):
        return len(self.game_anchors)


class SoccerNet(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", version=1, 
                framerate=2):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
            self.K_parameters = K_V1*framerate
            self.num_detections =5
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"
            self.K_parameters = K_V2*framerate		
            self.num_detections =15

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))

        # Load labels
        labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        for annotation in labels["annotations"]:

            time = annotation["gameTime"]
            event = annotation["label"]

            half = int(time[0])

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            framerate=2
            frame = framerate * ( seconds + 60 * minutes ) 

            if event not in self.dict_event:
                continue
            label = self.dict_event[event]

            if half == 1:
                label_half1[frame][label] = 1

            if half == 2:
                label_half2[frame][label] = 1


        return feat_half1, feat_half2, label_half1, label_half2

    def __len__(self):
        return len(self.listGames)

        

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    # dataset_Train = SoccerNetClips(path="/media/giancos/Football/SoccerNet/" ,features="ResNET_PCA512.npy", split="train")
    # print(len(dataset_Train))
    # feats, labels = dataset_Train[0]
    # print(feats.shape)
    # print(labels.shape)


    # train_loader = torch.utils.data.DataLoader(dataset_Train,
    #     batch_size=8, shuffle=True,
    #     num_workers=4, pin_memory=True)
    # for i, (feats, labels) in enumerate(train_loader):
    #     print(i, feats.shape, labels.shape)

    dataset_Test = SoccerNetClipsTesting(path="/media/giancos/Football/SoccerNet/" ,features="ResNET_PCA512.npy")
    print(len(dataset_Test))
    feats1, feats2, labels1, labels2 = dataset_Test[0]
    print(feats1.shape)
    print(labels1.shape)
    print(feats2.shape)
    print(labels2.shape)
    print(feats1[-1])