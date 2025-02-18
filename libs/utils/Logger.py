# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Author  : Haoxin Chen
# # @File    : Logger.py
# import torch
# import numpy as np
# import time
# import sys


# class Tee(object):
#     def __init__(self, name, mode):
#         self.file = open(name, mode)
#         self.stdout = sys.stdout
#         sys.stdout = self

#     def __del__(self):
#         sys.stdout = self.stdout
#         self.file.close()

#     def write(self, data):
#         self.file.write(data)
#         self.stdout.write(data)

#     def flush(self):
#         self.file.flush()


# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n

#     def avg(self):
#         return self.sum / self.count


# class Loss_record():
#     '''save the loss: total(tensor), part1 and part2 can be 0'''

#     def __init__(self):
#         self.total = AverageMeter()
#         self.part1 = AverageMeter()
#         self.part2 = AverageMeter()
#         self.part3 = AverageMeter()
#         self.part4 = AverageMeter()

#     def reset(self):
#         self.total.reset()
#         self.part1.reset()
#         self.part2.reset()
#         self.part3.reset()
#         self.part4.reset()

#     def updateloss(self, loss_val, loss_part1=0, loss_part2=0, loss_part3=0, loss_part4=0):
#         self.total.update(loss_val.data.item(), 1)
#         self.part1.update(loss_part1.data.item(), 1) if isinstance(loss_part1, torch.Tensor) else self.part1.update(0,
#                                                                                                                     1)
#         self.part2.update(loss_part2.data.item(), 1) if isinstance(loss_part2, torch.Tensor) else self.part2.update(0,
#                                                                                                                     1)
#         self.part3.update(loss_part3.data.item(), 1) if isinstance(loss_part3, torch.Tensor) else self.part3.update(0,
#                                                                                                                     1)
#         self.part4.update(loss_part4.data.item(), 1) if isinstance(loss_part4, torch.Tensor) else self.part4.update(0,
#                                                                                                                     1)

#     def getloss(self, epoch, step):
#         ''' get every step loss and reset '''
#         total_avg = self.total.avg()
#         part1_avg = self.part1.avg()
#         part2_avg = self.part2.avg()
#         part3_avg = self.part3.avg()
#         part4_avg = self.part4.avg()
#         out_str = 'epoch %d, step %d : %.4f, %.4f, %.4f, %.4f, %.4f' % \
#                   (epoch, step, total_avg, part1_avg, part2_avg, part3_avg, part4_avg)
#         self.reset()
#         return out_str



# def measure(y_in, pred_in):
#     thresh = .5
#     y = y_in > thresh
#     pred = pred_in > thresh
#     tp = np.logical_and(y, pred).sum()
#     tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
#     fp = np.logical_and(np.logical_not(y), pred).sum()
#     fn = np.logical_and(y, np.logical_not(pred)).sum()
#     return tp, tn, fp, fn


# from libs.utils.davis_JF import db_eval_boundary, db_eval_iou


# class TreeEvaluation():
#     def __init__(self):
#         self.setup()

#     def setup(self):
#         self.tp_list = {}
#         self.total_list = {}
#         self.f_list = {}
#         self.j_list = {}
#         self.n_list = {}
#         self.iou_list = {}
#         self.f_score = {}
#         self.j_score = {}

#     def update_evl(self, idx, query_mask, pred):
#         # B N H W
#         batch = len(idx)
#         for i in range(batch):
#             video_id = idx[i]
#             if video_id not in self.tp_list:
#                 self.tp_list[video_id] = 0
#                 self.total_list[video_id] = 0
#                 self.f_list[video_id] = 0
#                 self.j_list[video_id] = 0
#                 self.n_list[video_id] = 0
#                 self.iou_list[video_id] = 0
#                 self.f_score[video_id] = 0
#                 self.j_score[video_id] = 0

#             tp, total = self.test_in_train(query_mask[i], pred[i])
#             for j in range(query_mask[i].shape[0]):
#                 thresh = .5
#                 y = query_mask[i][j].cpu().numpy() > thresh
#                 predict = pred[i][j].data.cpu().numpy() > thresh
#                 self.f_list[video_id] += db_eval_boundary(predict[0], y)
#                 self.j_list[video_id] += db_eval_iou(y, predict[0])
#                 self.n_list[video_id] += 1

#             self.tp_list[video_id] += tp
#             self.total_list[video_id] += total

#             # Update scores for this video
#             self.iou_list[video_id] = self.tp_list[video_id] / float(max(self.total_list[video_id], 1))
#             self.f_score[video_id] = self.f_list[video_id] / float(max(self.n_list[video_id], 1))
#             self.j_score[video_id] = self.j_list[video_id] / float(max(self.n_list[video_id], 1))

#     def test_in_train(self, query_label, pred):
#         # test N*H*F
#         pred = pred.data.cpu().numpy()
#         query_label = query_label.cpu().numpy()

#         tp, tn, fp, fn = measure(query_label, pred)
#         total = tp + fp + fn
#         return tp, total

#     def logiou(self, epoch=None, step=None):
#         mean_iou = np.mean(list(self.iou_list.values()))
#         mean_f = np.mean(list(self.f_score.values()))
#         mean_j = np.mean(list(self.j_score.values()))
#         out_str = f'Mean IoU: {mean_iou:.4f}, Mean F: {mean_f:.4f}, Mean J: {mean_j:.4f}'
#         return out_str

#     def get_metrics(self):
#         """获取所有评估指标"""
#         return {
#             'mean_iou': np.mean(list(self.iou_list.values())),
#             'mean_f': np.mean(list(self.f_score.values())),
#             'mean_j': np.mean(list(self.j_score.values())),
#             'per_video_iou': self.iou_list,
#             'per_video_f': self.f_score,
#             'per_video_j': self.j_score
#         }

#     def print_metrics(self):
#         """打印详细的评估指标"""
#         metrics = self.get_metrics()
#         print("\nEvaluation Results:")
#         print(f"Mean IoU: {metrics['mean_iou']:.4f}")
#         print(f"Mean F: {metrics['mean_f']:.4f}")
#         print(f"Mean J: {metrics['mean_j']:.4f}")
        
#         print("\nPer-video Metrics:")
#         for video_id in self.iou_list.keys():
#             print(f"Video {video_id}:")
#             print(f"  IoU: {self.iou_list[video_id]:.4f}")
#             print(f"  F-score: {self.f_score[video_id]:.4f}")
#             print(f"  J-score: {self.j_score[video_id]:.4f}")


# class TimeRecord():
#     def __init__(self, maxstep, max_epoch):
#         self.maxstep = maxstep
#         self.max_epoch = max_epoch

#     def gettime(self, epoch, begin_time):
#         step_time = time.time() - begin_time
#         remaining_time = (self.max_epoch - epoch) * step_time * self.maxstep / 3600
#         return step_time, remaining_time


# class LogTime():
#     def __init__(self):
#         self.reset()

#     def t1(self):
#         self.logt1 = time.time()

#     def t2(self):
#         self.logt2 = time.time()
#         self.alltime += (self.logt2 - self.logt1)

#     def reset(self):
#         self.logt1 = None
#         self.logt2 = None
#         self.alltime = 0

#     def getalltime(self):
#         out = self.alltime
#         self.reset()
#         return out

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : Logger.py
import torch
import numpy as np
import time
import sys


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def avg(self):
        return self.sum / self.count


class Loss_record():
    '''save the loss: total(tensor), part1 and part2 can be 0'''

    def __init__(self):
        self.total = AverageMeter()
        self.part1 = AverageMeter()
        self.part2 = AverageMeter()
        self.part3 = AverageMeter()
        self.part4 = AverageMeter()

    def reset(self):
        self.total.reset()
        self.part1.reset()
        self.part2.reset()
        self.part3.reset()
        self.part4.reset()

    def updateloss(self, loss_val, loss_part1=0, loss_part2=0, loss_part3=0, loss_part4=0):
        self.total.update(loss_val.data.item(), 1)
        self.part1.update(loss_part1.data.item(), 1) if isinstance(loss_part1, torch.Tensor) else self.part1.update(0,
                                                                                                                    1)
        self.part2.update(loss_part2.data.item(), 1) if isinstance(loss_part2, torch.Tensor) else self.part2.update(0,
                                                                                                                    1)
        self.part3.update(loss_part3.data.item(), 1) if isinstance(loss_part3, torch.Tensor) else self.part3.update(0,
                                                                                                                    1)
        self.part4.update(loss_part4.data.item(), 1) if isinstance(loss_part4, torch.Tensor) else self.part4.update(0,
                                                                                                                    1)

    def getloss(self, epoch, step):
        ''' get every step loss and reset '''
        total_avg = self.total.avg()
        part1_avg = self.part1.avg()
        part2_avg = self.part2.avg()
        part3_avg = self.part3.avg()
        part4_avg = self.part4.avg()
        out_str = 'epoch %d, step %d : %.4f, %.4f, %.4f, %.4f, %.4f' % \
                  (epoch, step, total_avg, part1_avg, part2_avg, part3_avg, part4_avg)
        self.reset()
        return out_str



def measure(y_in, pred_in):
    thresh = .5
    y = y_in > thresh
    pred = pred_in > thresh
    tp = np.logical_and(y, pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn


from libs.utils.davis_JF import db_eval_boundary, db_eval_iou


class TreeEvaluation():
    '''eval training output'''

    def __init__(self, class_list=None):
        assert class_list is not None
        self.class_indexes = class_list
        self.num_classes = len(class_list)
        self.setup()

    def setup(self):
        # For non-empty masks
        self.tp_list = [0] * self.num_classes
        self.f_list = [0] * self.num_classes
        self.j_list = [0] * self.num_classes
        self.n_list = [0] * self.num_classes
        self.total_list = [0] * self.num_classes
        self.iou_list = [0] * self.num_classes
        self.f_score = [0] * self.num_classes
        self.j_score = [0] * self.num_classes

        # For empty masks
        self.tp_list_empty = [0] * self.num_classes
        self.f_list_empty = [0] * self.num_classes
        self.j_list_empty = [0] * self.num_classes
        self.n_list_empty = [0] * self.num_classes
        self.total_list_empty = [0] * self.num_classes
        self.iou_list_empty = [0] * self.num_classes
        self.f_score_empty = [0] * self.num_classes
        self.j_score_empty = [0] * self.num_classes

    def update_evl(self, idx, query_mask, pred):
        # B N H W
        batch = len(idx)
        for i in range(batch):
            id = idx[i]
            id = self.class_indexes.index(id)

            # Calculate video-level metrics
            y = query_mask[i].cpu().numpy() > 0.5
            predict = pred[i].data.cpu().numpy() > 0.5
            
            # Check if video contains any object
            # is_empty = (y.sum() == 0)
            is_empty = False

            if is_empty:
                # Update empty video metrics
                self.f_list_empty[id] += np.mean([db_eval_boundary(predict[j][0], y[j]) for j in range(y.shape[0])])
                self.j_list_empty[id] += np.mean([db_eval_iou(y[j], predict[j][0]) for j in range(y.shape[0])])
                self.n_list_empty[id] += 1
            else:
                # Update non-empty video metrics  
                self.f_list[id] += np.mean([db_eval_boundary(predict[j][0], y[j]) for j in range(y.shape[0])])
                self.j_list[id] += np.mean([db_eval_iou(y[j], predict[j][0]) for j in range(y.shape[0])])
                self.n_list[id] += 1

            # Calculate video-level TP and total
            tp, total = self.test_in_train(query_mask[i], pred[i])
            if is_empty:
                self.tp_list_empty[id] += tp
                self.total_list_empty[id] += total
            else:
                self.tp_list[id] += tp
                self.total_list[id] += total

        # Calculate metrics for non-empty videos
        self.iou_list = [self.tp_list[ic] / float(max(self.total_list[ic], 1))
                        for ic in range(self.num_classes)]
        self.f_score = [self.f_list[ic] / float(max(self.n_list[ic], 1))
                       for ic in range(self.num_classes)]
        self.j_score = [self.j_list[ic] / float(max(self.n_list[ic], 1))
                       for ic in range(self.num_classes)]

        # Calculate metrics for empty videos
        self.iou_list_empty = [self.tp_list_empty[ic] / float(max(self.total_list_empty[ic], 1))
                              for ic in range(self.num_classes)]
        self.f_score_empty = [self.f_list_empty[ic] / float(max(self.n_list_empty[ic], 1))
                             for ic in range(self.num_classes)]
        self.j_score_empty = [self.j_list_empty[ic] / float(max(self.n_list_empty[ic], 1))
                             for ic in range(self.num_classes)]

    def test_in_train(self, query_label, pred):
        # Calculate video-level metrics
        pred = pred.data.cpu().numpy()
        query_label = query_label.cpu().numpy()

        # Average over all frames
        tp, tn, fp, fn = measure(query_label, pred)
        total = tp + fp + fn
        return tp, total

    def logiou(self, epoch=None, step=None):
        mean_iou = np.mean(self.iou_list)
        mean_iou_empty = np.mean(self.iou_list_empty)
        out_str = 'iou (non-empty): %.4f, iou (empty): %.4f' % (mean_iou, mean_iou_empty)
        self.setup()
        return out_str
    def get_metrics(self):
        """获取所有评估指标"""
        # Get indices of categories with test samples
        valid_indices = [i for i in range(len(self.n_list)) if self.n_list[i] > 0]
        valid_indices_empty = [i for i in range(len(self.n_list_empty)) if self.n_list_empty[i] > 0]
        
        # Calculate means only for categories with samples
        mean_iou = np.mean([self.iou_list[i] for i in valid_indices]) if valid_indices else None
        mean_f = np.mean([self.f_score[i] for i in valid_indices]) if valid_indices else None
        mean_j = np.mean([self.j_score[i] for i in valid_indices]) if valid_indices else None
        
        mean_iou_empty = np.mean([self.iou_list_empty[i] for i in valid_indices_empty]) if valid_indices_empty else None
        mean_f_empty = np.mean([self.f_score_empty[i] for i in valid_indices_empty]) if valid_indices_empty else None
        mean_j_empty = np.mean([self.j_score_empty[i] for i in valid_indices_empty]) if valid_indices_empty else None
        
        return {
            'n_list': self.n_list,
            'mean_iou': mean_iou,
            'mean_f': mean_f, 
            'mean_j': mean_j,
            'mean_iou_empty': mean_iou_empty,
            'mean_f_empty': mean_f_empty,
            'mean_j_empty': mean_j_empty,
            'per_category_metrics': {
                category: {
                    'iou': self.iou_list[i],
                    'f_score': self.f_score[i],
                    'j_score': self.j_score[i],
                    'iou_empty': self.iou_list_empty[i],
                    'f_score_empty': self.f_score_empty[i],
                    'j_score_empty': self.j_score_empty[i],
                    'j_list': self.j_list[i],
                    'j_list_empty': self.j_list_empty[i],
                    'f_list': self.f_list[i],
                    'f_list_empty': self.f_list_empty[i],
                    'n_list': self.n_list[i],
                    "n_list_empty": self.n_list_empty[i]
                }
                for i, category in enumerate(self.class_indexes)
            }
        }


class TimeRecord():
    def __init__(self, maxstep, max_epoch):
        self.maxstep = maxstep
        self.max_epoch = max_epoch

    def gettime(self, epoch, begin_time):
        step_time = time.time() - begin_time
        remaining_time = (self.max_epoch - epoch) * step_time * self.maxstep / 3600
        return step_time, remaining_time


class LogTime():
    def __init__(self):
        self.reset()

    def t1(self):
        self.logt1 = time.time()

    def t2(self):
        self.logt2 = time.time()
        self.alltime += (self.logt2 - self.logt1)

    def reset(self):
        self.logt1 = None
        self.logt2 = None
        self.alltime = 0

    def getalltime(self):
        out = self.alltime
        self.reset()
        return out