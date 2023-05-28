import torch


####################################################################################################################################################

# Context-aware loss function

####################################################################################################################################################

class ContextAwareLoss(torch.nn.Module):

    def __init__(self, K, hit_radius = 0.1, miss_radius = 0.9):

        super(ContextAwareLoss,self).__init__()

        self.K = K
        self.hit_radius = float(hit_radius)
        self.miss_radius = float(miss_radius)

    def forward(self, gt_label, pred_score):

        params = self.K
        hit_radius = self.hit_radius
        miss_radius = self.miss_radius
        y_true = gt_label
        y_pred = pred_score
        
        loss = torch.sum(-torch.min(0.*y_true,torch.sign(y_true))*torch.max(0.*y_pred,-torch.log(y_pred+(1.-y_pred)*(torch.min(torch.max(y_true,params[0,:]),params[1,:])-params[0,:])/(params[1,:]-params[0,:]))+torch.log(torch.tensor(miss_radius)))+(1.-torch.max(0.*y_true,torch.sign(-y_true)))*torch.max(0.*y_pred,torch.max(-torch.log(1.-y_pred+torch.abs(y_true)/params[2,:])+torch.log(1.-torch.tensor(hit_radius)),-torch.log(y_pred+(torch.min(y_true,params[3,:])-params[3,:])/(params[2,:]-params[3,:]))+torch.log(torch.tensor(miss_radius)))))
        
        return loss  

        """
        zeros = torch.zeros(pred_score.size()).to(pred_score.device).type(torch.float)
        pred_score = torch.clamp(1.-pred_score,0.00001,1.)
        #pred_score = 1.-pred_score
        
        case1 = self.DownStep(gt_label, self.K[0]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + self.miss_radius))
        case2 = self.Interval(gt_label, self.K[0], self.K[1]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(self.PartialIdentity(gt_label,self.K[0],self.K[1])-self.K[0])/(self.K[1]-self.K[0])) + torch.log(zeros + self.miss_radius))
        #note cases 1 and 2 could be regrouped with DownStep(gt_label, self.K[1]) * formula loss of case2
        case3 = self.Interval(gt_label, self.K[1], 0.) * zeros
        case4 = self.Interval(gt_label, 0., self.K[2]) * torch.max(zeros, - torch.log(1.-pred_score + pred_score*(self.PartialIdentity(gt_label,0.,self.K[2])-0.)/(self.K[2]-0.)) + torch.log(zeros + 1.-self.hit_radius))
        case5 = self.Interval(gt_label, self.K[2], self.K[3]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(self.PartialIdentity(gt_label,self.K[2],self.K[3])-self.K[3])/(self.K[2]-self.K[3])) + torch.log(zeros + self.miss_radius))
        case6 = self.UpStep(gt_label, self.K[3]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + self.miss_radius))
        #note cases 5 and 6 could be regrouped 
        
        L = case1 + case2 + case3 + case4 + case5 + case6


        
        return torch.sum(L)
        """

    def UpStep(self,x,a): #0 if x<a, 1 if x >= a

        return 1.-torch.max(0.*x,torch.sign(a-x))

    def DownStep(self,x,a): #1 if x < a, 0 if x >=a

        return torch.max(0.*x,torch.sign(a-x))

    def Interval(self,x,a,b): # 1 if a<= x < b, 0 otherwise
        
        return self.UpStep(x,a) * self.DownStep(x,b)

    def PartialIdentity(self,x,a,b):#a if x<a, x if a<= x <b, b if x >= b

        return torch.min(torch.max(x,0.*x+a),0.*x+b)




####################################################################################################################################################

# Spotting loss

####################################################################################################################################################

class SpottingLoss(torch.nn.Module):

    def __init__(self, lambda_coord, lambda_noobj):
        super(SpottingLoss,self).__init__()

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def forward(self,y_true, y_pred):
        y_pred = self.permute_ypred_for_matching(y_true,y_pred)
        loss = torch.sum(y_true[:,:,0]*self.lambda_coord*torch.square(y_true[:,:,1]-y_pred[:,:,1])  +  y_true[:,:,0]*torch.square(y_true[:,:,0]-y_pred[:,:,0]) +  (1-y_true[:,:,0])*self.lambda_noobj*torch.square(y_true[:,:,0]-y_pred[:,:,0]) +  y_true[:,:,0]*torch.sum(torch.square(y_true[:,:,2:]-y_pred[:,:,2:]),axis=-1)) #-y_true[:,:,0]*torch.sum(y_true[:,:,2:]*torch.log(y_pred[:,:,2:]),axis=-1)
        return loss


    def permute_ypred_for_matching(self, y_true, y_pred):
        
        alpha = y_true[:,:,0]
        x = y_true[:,:,1]
        p = y_pred[:,:,1]
        nb_pred = x.shape[-1]
        
        
        D = torch.abs(x.unsqueeze(-1).repeat(1,1,nb_pred) - p.unsqueeze(-2).repeat(1,nb_pred,1))
        D1 = 1-D
        Permut = 0*D
        
        alpha_filter = alpha.unsqueeze(-1).repeat(1,1,nb_pred)
        
        v_filter = alpha_filter
        h_filter = 0*v_filter + 1 
        D2 = v_filter * D1

        for i in range(nb_pred):
            D2 = v_filter * D2
            D2 = h_filter * D2
            A = torch.nn.functional.one_hot(torch.argmax(D2,axis=-1),nb_pred)
            B = v_filter * A * D2
            C = torch.nn.functional.one_hot(torch.argmax(B,axis=-2),nb_pred).permute(0, 2, 1)
            E = v_filter * A * C
            Permut = Permut + E
            v_filter = (1-torch.sum(Permut,axis=-1))*alpha
            v_filter = v_filter.unsqueeze(-1).repeat(1,1,nb_pred)
            h_filter = 1-torch.sum(Permut, axis=-2)
            h_filter = h_filter.unsqueeze(-2).repeat(1,nb_pred,1)
        
        v_filter = 1-alpha_filter
        D2 = v_filter * D1
        D2 = h_filter * D2
        
        for i in range(nb_pred):
            D2 = v_filter * D2
            D2 = h_filter * D2
            A = torch.nn.functional.one_hot(torch.argmax(D2,axis=-1),nb_pred)
            B = v_filter * A * D2
            C = torch.nn.functional.one_hot(torch.argmax(B,axis=-2),nb_pred).permute(0, 2, 1)
            E = v_filter * A * C
            Permut = Permut + E
            v_filter = (1-torch.sum(Permut,axis=-1))*(1-alpha) #here comes the change
            v_filter = v_filter.unsqueeze(-1).repeat(1,1,nb_pred)
            h_filter = 1-torch.sum(Permut, axis=-2)
            h_filter = h_filter.unsqueeze(-2).repeat(1,nb_pred,1)
        
        permutation = torch.argmax(Permut,axis=-1)
        permuted = torch.gather(y_pred, 1, permutation.unsqueeze(-1).repeat(1,1,y_true.shape[-1]))
            
        return permuted



if __name__ == "__main__":

    K_V1 = torch.FloatTensor([[-20,-20,-40],[-10,-10,-20],[60,10,10],[90,20,20]])
    labels = torch.load("labels.pt")
    output_segmentation = torch.load("output_segmentation.pt")


    criterion_segmentation = ContextAwareLoss(K=K_V1)

    print(labels)
    print(output_segmentation)

    print(output_segmentation.max())
    print(output_segmentation.min())
    loss = criterion_segmentation(labels,output_segmentation)
    print(loss)

    torch.save(labels, "labels.pt")
    torch.save(output_segmentation, "output_segmentation.pt")



"""
def SymmetricLoss(pred_score, gt_label, K, hit_radius = 0.1, miss_radius = 0.9, coef_0 = 1., coef_1 = 1.):
    
    zeros = torch.zeros(pred_score.size()).to(pred_score.device).type(torch.float)
    pred_score = 1.-pred_score
    
    case1 = DownStep(gt_label, self.K[0]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + miss_radius))
    case2 = Interval(gt_label, self.K[0], self.K[1]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(PartialIdentity(gt_label,self.K[0],self.K[1])-self.K[0])/(self.K[1]-self.K[0])) + torch.log(zeros + miss_radius))
    #note cases 1 and 2 could be regrouped with DownStep(gt_label, self.K[1]) * formula loss of case2
    case3 = Interval(gt_label, self.K[1], 0.) * torch.max(zeros, - torch.log(1.-pred_score + pred_score*(PartialIdentity(gt_label,self.K[1],0.)-0.)/(self.K[1]-0.)) + torch.log(zeros + 1.-hit_radius))
    case4 = Interval(gt_label, 0., self.K[2]) * torch.max(zeros, - torch.log(1.-pred_score + pred_score*(PartialIdentity(gt_label,0.,self.K[2])-0.)/(self.K[2]-0.)) + torch.log(zeros + 1.-hit_radius))
    case5 = Interval(gt_label, self.K[2], self.K[3]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(PartialIdentity(gt_label,self.K[2],self.K[3])-self.K[3])/(self.K[2]-self.K[3])) + torch.log(zeros + miss_radius))
    case6 = UpStep(gt_label, self.K[3]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + miss_radius))
    #note cases 5 and 6 could be regrouped 
    
    L = coef_0*case1 + coef_0*case2 + coef_1*case3 + coef_1*case4 + coef_0*case5 + coef_0*case6
    
    return torch.sum(L)

def MirrorLoss(pred_score, gt_label, K, hit_radius = 0.1, miss_radius = 0.9, coef_0 = 1., coef_1 = 1.):
    
    zeros = torch.zeros(pred_score.size()).to(pred_score.device).type(torch.float)
    pred_score = 1.-pred_score
    
    case1 = DownStep(gt_label, self.K[0]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + miss_radius))
    case2 = Interval(gt_label, self.K[0], self.K[1]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(PartialIdentity(gt_label,self.K[0],self.K[1])-self.K[0])/(self.K[1]-self.K[0])) + torch.log(zeros + miss_radius))
    #note cases 1 and 2 could be regrouped with DownStep(gt_label, self.K[1]) * formula loss of case2
    case3 = Interval(gt_label, self.K[1], 0.) * torch.max(zeros, - torch.log(1.-pred_score + pred_score*(PartialIdentity(gt_label,self.K[1],0.)-0.)/(self.K[1]-0.)) + torch.log(zeros + 1.-hit_radius))
    case4 = Interval(gt_label, 0., self.K[2]) * zeros
    case5 = Interval(gt_label, self.K[2], self.K[3]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(PartialIdentity(gt_label,self.K[2],self.K[3])-self.K[3])/(self.K[2]-self.K[3])) + torch.log(zeros + miss_radius))
    case6 = UpStep(gt_label, self.K[3]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + miss_radius))
    #note cases 5 and 6 could be regrouped 
    
    L = coef_0*case1 + coef_0*case2 + coef_1*case3 + coef_1*case4 + coef_0*case5 + coef_0*case6
    
    return torch.sum(L)
"""

#Code for testing
    
#import numpy as np
    
#np.random.seed(25)
#
#y_true = np.random.randint(-20.,20.,size=100)
#y_pred = np.random.uniform(0.,1.,size=100)
#
#params = np.array([-10.,-5.,5.,10.])
#hit_radius = 0.1
#miss_radius = 0.9
#
#L1 = SoccerNetLoss(torch.Tensor(y_pred), torch.Tensor(y_true), params, hit_radius, miss_radius, 1., 1.)
#L2 = SymmetricLoss(torch.Tensor(y_pred), torch.Tensor(y_true), params, hit_radius, miss_radius, 1., 1.)
#L3 = MirrorLoss(torch.Tensor(y_pred), torch.Tensor(y_true), params, hit_radius, miss_radius, 1., 1.)
#
#print(L1)
#print(L2)
#print(L3)




#For testing
 
#import numpy as np
#
#nb_predict = 7
#batch_size = 2
#num_classes = 3
#
#x = np.random.uniform(0,1,(batch_size, nb_predict, 2+num_classes))
#x = np.around(x,2)
#x[...,0] = np.round(x[...,0])
#x[...,2:] = 0
#X = torch.tensor(x)
#
#
#p = np.random.uniform(0,1,(batch_size, nb_predict, 2+num_classes))
#p = np.around(p,2)
#p[...,0] = 0
#p[...,2:] = 0
#P = torch.tensor(p)
#
#
#print(x)
#print(p)
#permutation = torch_permute_ypred_for_matching(X,P)
#
#print(permutation)

