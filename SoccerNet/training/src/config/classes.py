import torch

EVENT_DICTIONARY_V1 = {"soccer-ball": 0, "soccer-ball-own": 0, "r-card": 1, "y-card": 1, "yr-card": 1,
                                 "substitution-in": 2}
EVENT_DICTIONARY_V2 = {"Penalty":0,"Kick-off":1,"Goal":2,"Substitution":3,"Offside":4,"Shots on target":5,
                                "Shots off target":6,"Clearance":7,"Ball out of play":8,"Throw-in":9,"Foul":10,
                                "Indirect free-kick":11,"Direct free-kick":12,"Corner":13,"Yellow card":14
                                ,"Red card":15,"Yellow->red card":16}
K_V1 = torch.FloatTensor([[-20,-20,-40],[-10,-10,-20],[60,10,10],[90,20,20]]).cuda()
K_V2 = torch.FloatTensor([[-20,-20,-20,-40,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20],[-10,-10,-10,-20,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10],[10,10,60,10,10,10,10,10,10,10,10,10,10,10,10,10,10],[20,20,90,20,20,20,20,20,20,20,20,20,20,20,20,20,20]]).cuda()