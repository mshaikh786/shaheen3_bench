import argparse
import os 
import SoccerNet



import configparser
import math
try:
    from tensorflow.keras.models import Model  # pip install tensorflow (==2.3.0)
    from tensorflow.keras.applications.resnet import preprocess_input
    # from tensorflow.keras.preprocessing.image import img_to_array
    # from tensorflow.keras.preprocessing.image import load_img
    from tensorflow import keras
except:
    print("issue loading TF2")
    pass
import os
# import argparse
import numpy as np
import cv2  # pip install opencv-python (==3.4.11.41)
import imutils  # pip install imutils
import skvideo.io
from tqdm import tqdm

# from SoccerNet import getListTestGames, getListGames

# try:
#     import torch
#     import torchvision
#     from SoccerNet.DataLoaderTorch import ClipDataset, FrameDataset

# except:
#     print("issue loading PT")
#     pass
import json

from SoccerNet.utils import getListGames
# from .DataLoader import getDuration
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.DataLoader import Frame, FrameCV


class FeatureExtractor():
    def __init__(self, rootFolder,
                 feature="ResNET",
                 video="LQ",
                 back_end="TF2",
                 overwrite=False,
                 transform="crop",
                 tmp_HQ_videos=None,
                 grabber="opencv",
                 FPS=2.0):
        self.rootFolder = rootFolder
        self.feature = feature
        self.video = video
        self.back_end = back_end
        self.verbose = True
        self.transform = transform
        self.overwrite = overwrite
        self.grabber = grabber
        self.FPS = FPS

        self.tmp_HQ_videos = tmp_HQ_videos
        if self.tmp_HQ_videos:
            self.mySoccerNetDownloader = SoccerNetDownloader(self.rootFolder)
            self.mySoccerNetDownloader.password = self.tmp_HQ_videos

        if "TF2" in self.back_end:

            # create pretrained encoder (here ResNet152, pre-trained on ImageNet)
            base_model = keras.applications.resnet.ResNet152(include_top=True,
                                                             weights='imagenet',
                                                             input_tensor=None,
                                                             input_shape=None,
                                                             pooling=None,
                                                             classes=1000)

            # define model with output after polling layer (dim=2048)
            self.model = Model(base_model.input,
                               outputs=[base_model.get_layer("avg_pool").output])
            self.model.trainable = False

        elif self.back_end == "PT" and "ResNET" in self.feature:
            class Identity(torch.nn.Module):
                def __init__(self):
                    super(Identity, self).__init__()

                def forward(self, x):
                    return x

            self.model = torchvision.models.resnet152(
                pretrained=True, progress=True).cuda()
            self.model.fc = Identity()

        elif self.back_end == "PT" and "R25D" in self.feature:
            class Identity(torch.nn.Module):
                def __init__(self):
                    super(Identity, self).__init__()

                def forward(self, x):
                    return x

            self.model = torchvision.models.video.r2plus1d_18(
                pretrained=True, progress=True).cuda()
            self.model.fc = Identity()

        

    def extractAllGames(self):
        for i_game, game in tqdm(getListGames("all")):
            self.extractGameIndex(i_game)

    def extractGameIndex(self, index):
        print(getListGames("all")[index])
        if self.video =="LQ":
            for vid in ["1.mkv","2.mkv"]:
                self.extract(video_path=os.path.join(self.rootFolder, getListGames("all")[index], vid))

        elif self.video == "HQ":
            
            # read config for raw HD video
            config = configparser.ConfigParser()
            if not os.path.exists(os.path.join(self.rootFolder, getListGames("all")[index], "video.ini")) and self.tmp_HQ_videos is not None:
                self.mySoccerNetDownloader.downloadVideoHD(
                    game=getListGames("all")[index], file="video.ini")
            config.read(os.path.join(self.rootFolder, getListGames("all")[index], "video.ini"))

            # lopp over videos
            for vid in config.sections():
                video_path = os.path.join(self.rootFolder, getListGames("all")[index], vid)

                # cehck if already exists, then skip
                feature_path = video_path[:-4] + f"_{self.feature}_{self.back_end}.npy"
                if os.path.exists(feature_path) and not self.overwrite:
                    print("already exists, early skip")
                    continue

                #Download video if does not exist, but remove it afterwards
                remove_afterwards = False
                if not os.path.exists(video_path) and self.tmp_HQ_videos is not None:
                    remove_afterwards = True
                    self.mySoccerNetDownloader.downloadVideoHD(game=getListGames("all")[index], file=vid)

                # extract feature for video
                self.extract(video_path=video_path,
                            start=float(config[vid]["start_time_second"]), 
                            duration=float(config[vid]["duration_second"]))
                
                # remove video if not present before
                if remove_afterwards:
                    os.remove(video_path)

    def extract(self, video_path, start=None, duration=None):
        print("extract video", video_path, "from", start, duration)
        # feature_path = video_path.replace(
        #     ".mkv", f"_{self.feature}_{self.back_end}.npy")
        feature_path = video_path[:-4] + f"_{self.feature}_{self.back_end}.npy"

        if os.path.exists(feature_path) and not self.overwrite:
            return
        if "TF2" in self.back_end:
            
            if self.grabber=="skvideo":
                videoLoader = Frame(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
            elif self.grabber=="opencv":
                videoLoader = FrameCV(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)

            # create numpy aray (nb_frames x 224 x 224 x 3)
            # frames = np.array(videoLoader.frames)
            # if self.preprocess:
            frames = preprocess_input(videoLoader.frames)
            
            if duration is None:
                duration = videoLoader.time_second
                # time_second = duration
            if self.verbose:
                print("frames", frames.shape, "fps=", frames.shape[0]/duration)

            # predict the featrues from the frames (adjust batch size for smalled GPU)
            features = self.model.predict(frames, batch_size=64, verbose=1)
            if self.verbose:
                print("features", features.shape, "fps=", features.shape[0]/duration)


        elif self.back_end == "PT" and "ResNET" in self.feature:

            myClips = FrameDataset(video_path=video_path,
                                  FPS_desired=2)

            myClipsLoader = torch.utils.data.DataLoader(myClips, batch_size=1)

            features = []
            for clip in tqdm(myClipsLoader):
                clip = clip.cuda()
                # print(clip.shape)
                feat = self.model(clip)
                # print(feat.shape)
                features.append(feat.detach().cpu())
                # print(len(features))
                # del clip
                # del feat
            print(orch.cat(features).shape)
            features = torch.cat(features).numpy()
            print(features.shape)


        elif self.back_end == "PT" and "R25D" in self.feature:

            myClips = ClipDataset(video_path=video_path, FPS_desired=2, clip_len=16)

            myClipsLoader = torch.utils.data.DataLoader(myClips, batch_size=1)


            features = []
            for clip in tqdm(myClipsLoader):
                clip = clip.cuda()
                # print(clip.shape)
                feat = self.model(clip)
                features.append(feat.detach().cpu())
                # print(len(feats))
                # del clip
                # del feat
            features = torch.cat(features).numpy()
            # print(features.shape)

        # save the featrue in .npy format
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        np.save(feature_path, features)



if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Extract ResNet feature out of SoccerNet Videos.')

    parser.add_argument('--soccernet_dirpath', type=str, default="/media/giancos/Football/SoccerNet/",
                        help="Path for SoccerNet directory [default:/media/giancos/Football/SoccerNet/]")
    # parser.add_argument('--soccernet_featpath', type=str, default="/media/giancos/Football/SoccerNet_features/",
    #                     help="Path for SoccerNet directory [default:/media/giancos/Football/SoccerNet_features/]")
    # parser.add_argument('--soccernet_oldfeatpath', type=str, default="/media/giancos/Football/SoccerNet/",
    #                     help="Path for SoccerNet directory [default:/media/giancos/Football/SoccerNet/]")
    # parser.add_argument('--preprocess', type=str, default="crop",
                        # help="How to preprocess frames: crop or resize [default:crop]")
    # parser.add_argument('--split', type=str, default=['Test_100', 'Valid_100', 'Train_300'], metavar='N', nargs='+',
    #                     help="Split to elaborate? [default:['Test_100', 'Valid_100', 'Train_300']]")
    # parser.add_argument('--array', action="store_true",
    #                     help="convert img to array? [default:False]")
    # parser.add_argument('--preprocess', action="store_true",
    #                     help="preprocess the input? [default:False]")

    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite the features? [default:False]")
    parser.add_argument('--GPU', type=int, default=0,
                        help="ID of the GPU to use [default:0]")
    parser.add_argument('--verbose', action="store_true",
                        help="Print verbose? [default:False]")
    parser.add_argument('--game_ID', type=int, default=None,
                        help="ID of the game from which to extract features. If set to None, then loop over all games. [default:None]")

    # feature setup
    parser.add_argument('--back_end', type=str, default="TF2",
                        help="Backend TF2 or PT [default:TF2]")
    parser.add_argument('--features', type=str, default="ResNET",
                        help="ResNET or R25D [default:ResNET]")
    parser.add_argument('--transform', type=str, default="crop",
                        help="crop or resize? [default:crop]")
    parser.add_argument('--video', type=str, default="LQ",
                        help="LQ or HQ? [default:HQ]")
    parser.add_argument('--grabber', type=str, default="opencv",
                        help="skvideo or opencv? [default:skvideo]")
    parser.add_argument('--tmp_HQ_videos', type=str, default=None,
                        help="enter pawssword to download and store temporally the videos [default:None]")
    parser.add_argument('--FPS', type=float, default=2.0,
                        help="FPS for the features [default:2.0]")

    args = parser.parse_args()
    print(args)

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    myFeatureExtractor = FeatureExtractor(
        args.soccernet_dirpath, 
        feature=args.features,
        video=args.video,
        back_end=args.back_end, 
        # array=args.array,
        transform=args.transform,
        # preprocess=True,
        tmp_HQ_videos=args.tmp_HQ_videos,
        grabber=args.grabber,
        FPS=args.FPS)
    myFeatureExtractor.overwrite= args.overwrite

    # def extractGameIndex(self, index):
    if args.game_ID is None:
        myFeatureExtractor.extractAllGames()
    else:
        myFeatureExtractor.extractGameIndex(args.game_ID)
