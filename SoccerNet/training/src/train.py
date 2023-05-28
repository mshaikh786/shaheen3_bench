import logging
import os
from metrics_fast import AverageMeter ,average_mAP
from metrics_visibility_fast import average_mAP_visibility
#from metrics_visibility_corrected import average_mAP_visibility_corrected
import time
from tqdm import tqdm
import torch
import numpy as np
import math


def trainer(train_loader,
            val_loader,
            val_metric_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            weights,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20):

    logging.info("start training")

    best_loss = 9e99
    best_metric = -1
    
    e_time=float(0.0)
    b_time=0.0
    d_time=0.0
    btchs=0
    for epoch in range(max_epochs):
        best_model_path = os.path.join("models", model_name, "model.pth.tar")

        s_time = time.time()
        # train for one epoch
        loss_training,t_stamp = train(
            train_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train = True)
        b_time += t_stamp['batch']
        d_time += t_stamp['data']
        btchs = t_stamp['i']
        e_time += (time.time() - s_time)

        # evaluate on validation set
        loss_validation,t_stamp = train(
            val_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train = False)        

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)
        # torch.save(
        #     state,
        #     os.path.join("models", model_name,
        #                  "model_epoch" + str(epoch + 1) + ".pth.tar"))

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)



        # Save the best model based on loss only if the evaluation frequency too long
        if is_better and evaluation_frequency > 50:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(
                val_metric_loader,
                model, 
                model_name)

            if train_loader.dataset.version==2:
                performance_validation = performance_validation[0]
            logging.info("Validation performance at epoch " + str(epoch+1) + " -> " + str(performance_validation))

            is_better_metric = performance_validation > best_metric
            best_metric = max(performance_validation,best_metric)


            # Save the best model based on metric only if the evaluation frequency is short enough
            if is_better_metric and evaluation_frequency <= 50:
                torch.save(state, best_model_path)
                performance_test = test(
                    test_loader,
                    model, 
                    model_name)
                if train_loader.dataset.version==2:
                    performance_test = performance_test[0]

                logging.info("Test performance at epoch " + str(epoch+1) + " -> " + str(performance_test))

        if scheduler is not None:
            prevLR = optimizer.param_groups[0]['lr']
            scheduler.step(loss_validation)
            currLR = optimizer.param_groups[0]['lr']
            if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
                logging.info("Plateau Reached!")

            if (prevLR < 2 * scheduler.eps and
                    scheduler.num_bad_epochs >= scheduler.patience):
                logging.info(
                    "Plateau Reached and no more reduction -> Exiting Loop")
                break
        else:
            current_learning_rate = optimizer.param_groups[0]['lr']
            new_learning_rate = current_learning_rate * 0.993116#- (scheduler[0]-scheduler[1])/max_epochs# * 0.993116
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_learning_rate

            print(new_learning_rate)

        """

        """
    print("\nEPOCHS: %s , Epoch time: %s , Batches: %s, Batch Time: %s , Data Time: %s  \n" %(max_epochs,e_time,btchs,b_time,d_time))
    return

def train(dataloader,
          model,
          criterion, 
          weights,
          optimizer,
          epoch,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_segmentation = AverageMeter()
    losses_spotting = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()
    
    
    b_time=0.0
    d_time=0.0
    btchs=0
    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (feats, labels, targets) in t: 
            # measure data loading time
            data_time.update(time.time() - end)

            if dataloader.dataset.version == 1:
                feats = feats.cuda().squeeze(0)
                labels = labels.cuda().float().squeeze(0)
                targets = targets.cuda().float().squeeze(0)
            else:
                feats = feats.cuda()
                labels = labels.cuda().float()
                targets = targets.cuda().float()


            feats=feats.unsqueeze(1)

            # compute output
            output_segmentation, output_spotting = model(feats)

            loss_segmentation = criterion[0](labels, output_segmentation) 
            loss_spotting = criterion[1](targets, output_spotting)

            loss = weights[0]*loss_segmentation + weights[1]*loss_spotting

            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))
            losses_segmentation.update(loss_segmentation.item(), feats.size(0))
            losses_spotting.update(loss_spotting.item(), feats.size(0))

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            desc += f'Loss Seg {losses_segmentation.avg:.4e} '
            desc += f'Loss Spot {losses_spotting.avg:.4e} '
            t.set_description(desc)
            b_time+=batch_time.val
            d_time+=data_time.val
            btchs+=1
    t_stamp={'i':btchs,'batch':b_time,'data':d_time}
    return losses.avg,t_stamp


def test(dataloader,model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spotting_grountruth = list()
    spotting_grountruth_visibility = list()
    spotting_predictions = list()
    segmentation_predictions = list()

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    count_visible = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
    count_unshown = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
    count_all = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)

    def timestamps2long(output_spotting, video_size, chunk_size, receptive_field):

        start = 0
        last = False
        receptive_field = receptive_field//2

        timestamps_long = torch.zeros([video_size,output_spotting.size()[-1]-2], dtype = torch.float, device=output_spotting.device)-1


        for batch in np.arange(output_spotting.size()[0]):

            tmp_timestamps = torch.zeros([chunk_size,output_spotting.size()[-1]-2], dtype = torch.float, device=output_spotting.device)-1
            
            for i in np.arange(output_spotting.size()[1]):
                tmp_timestamps[torch.floor(output_spotting[batch,i,1]*(chunk_size-1)).type(torch.int) , torch.argmax(output_spotting[batch,i,2:]).type(torch.int) ] = output_spotting[batch,i,0]

            # ------------------------------------------
            # Store the result of the chunk in the video
            # ------------------------------------------

            # For the first chunk
            if start == 0:
                timestamps_long[0:chunk_size-receptive_field] = tmp_timestamps[0:chunk_size-receptive_field]

            # For the last chunk
            elif last:
                timestamps_long[start+receptive_field:start+chunk_size] = tmp_timestamps[receptive_field:]
                break

            # For every other chunk
            else:
                timestamps_long[start+receptive_field:start+chunk_size-receptive_field] = tmp_timestamps[receptive_field:chunk_size-receptive_field]
            
            # ---------------
            # Loop Management
            # ---------------

            # Update the index
            start += chunk_size - 2 * receptive_field
            # Check if we are at the last index of the game
            if start + chunk_size >= video_size:
                start = video_size - chunk_size 
                last = True
        return timestamps_long

    def batch2long(output_segmentation, video_size, chunk_size, receptive_field):

        start = 0
        last = False
        receptive_field = receptive_field//2

        segmentation_long = torch.zeros([video_size,output_segmentation.size()[-1]], dtype = torch.float, device=output_segmentation.device)


        for batch in np.arange(output_segmentation.size()[0]):

            tmp_segmentation = torch.nn.functional.one_hot(torch.argmax(output_segmentation[batch], dim=-1), num_classes=output_segmentation.size()[-1])


            # ------------------------------------------
            # Store the result of the chunk in the video
            # ------------------------------------------

            # For the first chunk
            if start == 0:
                segmentation_long[0:chunk_size-receptive_field] = tmp_segmentation[0:chunk_size-receptive_field]

            # For the last chunk
            elif last:
                segmentation_long[start+receptive_field:start+chunk_size] = tmp_segmentation[receptive_field:]
                break

            # For every other chunk
            else:
                segmentation_long[start+receptive_field:start+chunk_size-receptive_field] = tmp_segmentation[receptive_field:chunk_size-receptive_field]
            
            # ---------------
            # Loop Management
            # ---------------

            # Update the index
            start += chunk_size - 2 * receptive_field
            # Check if we are at the last index of the game
            if start + chunk_size >= video_size:
                start = video_size - chunk_size 
                last = True
        return segmentation_long

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, feat_half2, label_half1, label_half2) in t:
            data_time.update(time.time() - end)

            feat_half1 = feat_half1.cuda().squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.cuda().squeeze(0)
            label_half2 = label_half2.float().squeeze(0)


            feat_half1=feat_half1.unsqueeze(1)
            feat_half2=feat_half2.unsqueeze(1)

            # Compute the output
            output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)
            output_segmentation_half_2, output_spotting_half_2 = model(feat_half2)


            timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)
            segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

            spotting_grountruth.append(torch.abs(label_half1))
            spotting_grountruth.append(torch.abs(label_half2))
            spotting_grountruth_visibility.append(label_half1)
            spotting_grountruth_visibility.append(label_half2)
            spotting_predictions.append(timestamp_long_half_1)
            spotting_predictions.append(timestamp_long_half_2)
            segmentation_predictions.append(segmentation_long_half_1)
            segmentation_predictions.append(segmentation_long_half_2)

            count_all = count_all + torch.sum(torch.abs(label_half1), dim=0)
            count_visible = count_visible + torch.sum((torch.abs(label_half1)+label_half1)/2, dim=0)
            count_unshown = count_unshown + torch.sum((torch.abs(label_half1)-label_half1)/2, dim=0)
            count_all = count_all + torch.sum(torch.abs(label_half2), dim=0)
            count_visible = count_visible + torch.sum((torch.abs(label_half2)+label_half2)/2, dim=0)
            count_unshown = count_unshown + torch.sum((torch.abs(label_half2)-label_half2)/2, dim=0)


    if dataloader.dataset.version == 1:
        a_mAP = average_mAP(spotting_grountruth, spotting_predictions, model.framerate)
        print("Average-mAP: ", a_mAP)
    else:
        a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP_visibility(spotting_grountruth_visibility, spotting_predictions, model.framerate)
        print("a_mAP visibility all: ", a_mAP)
        print("a_mAP visibility all per class: ", a_mAP_per_class)
        print("a_mAP visibility visible: ", a_mAP_visible)
        print("a_mAP visibility visible per class: ", a_mAP_per_class_visible)
        print("a_mAP visibility unshown: ", a_mAP_unshown)
        print("a_mAP visibility unshown per class: ", a_mAP_per_class_unshown)    
        print("Count all: ", torch.sum(count_all))
        print("Count all per class: ", count_all)
        print("Count visible: ", torch.sum(count_visible))
        print("Count visible per class: ", count_visible)
        print("Count unshown: ", torch.sum(count_unshown))
        print("Count unshown per class: ", count_unshown)
        return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown


    return a_mAP
