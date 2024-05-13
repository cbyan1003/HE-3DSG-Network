#!/usr/bin/env python3
# -*- coding: utf-8 -*-
if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model_base import BaseModel
from network_PointNet import PointNetfeat,PointNetCls,PointNetRelCls,PointNetRelClsMulti
from network_TripletGCN import TripletGCNModel
from network_GNN import GraphEdgeAttenNetworkLayers
from network_GRU import GRUCell
import numpy as np
from collections import defaultdict
import time
from config import Config
import op_utils
import optimizer
import math
'''ulip'''
from utils.tokenizer import SimpleTokenizer
import json
import numpy as np

import utils.utils_ULIP
import os
from threading import Thread
from multiprocessing import Queue, Process
import torch.multiprocessing as mp
''''''
class MyThread(mp.Process):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
class SGFNModel(BaseModel):
    def __init__(self,config:Config,name:str, num_class, num_rel, dim_descriptor=11):
        super().__init__(name,config)
        models = dict()
        self.history_len = config.history_len
        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial
        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_class
        self.num_rel=num_rel
        
        self.flow = 'target_to_source' # we want the msg
        
        dim_point_feature = self.mconfig.point_feature_size
        if self.mconfig.USE_SPATIAL:
            dim_point_feature -= dim_f_spatial-3 # ignore centroid
        
        # Object Encoder
        models['obj_encoder'] = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # models['pc_encoder'] = ULIP(
            
        # )
        # models['text_encoder'] = ULIP(
            
        # )

        ## Relationship Encoder
        models['rel_encoder'] = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=512, #dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        ''' GRU '''
        models['obj_GRU_cell'] = nn.GRUCell(input_size = 512, hidden_size = 512) 
        models['rel_GRU_cell'] = nn.GRUCell(input_size = 512, hidden_size = 512) 

        # models['obj_GRU_cell'] = DataParallel(models['obj_GRU_cell'])
        # models['rel_GRU_cell'] = DataParallel(models['rel_GRU_cell'])
        ''' Message passing between segments and segments '''
        if self.mconfig.USE_GCN:
            if mconfig.GCN_TYPE == "TRIP":
                models['gcn'] = TripletGCNModel(num_layers=mconfig.N_LAYERS,
                                                dim_node = mconfig.point_feature_size,
                                                dim_edge = mconfig.edge_feature_size,
                                                dim_hidden = mconfig.gcn_hidden_feature_size)
            elif mconfig.GCN_TYPE == 'EAN':
                models['gcn'] = GraphEdgeAttenNetworkLayers(self.mconfig.point_feature_size,
                                    self.mconfig.edge_feature_size,
                                    self.mconfig.DIM_ATTEN,
                                    self.mconfig.N_LAYERS, 
                                    self.mconfig.NUM_HEADS,
                                    self.mconfig.GCN_AGGR,
                                    flow=self.flow,
                                    attention=self.mconfig.ATTENTION,
                                    use_edge=self.mconfig.USE_GCN_EDGE,
                                    DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)
            else:
                raise NotImplementedError('')
        
        ''' node feature classifier '''
        models['obj_predictor'] = PointNetCls(num_class, in_size=mconfig.point_feature_size,
                                 batch_norm=with_bn,drop_out=True)
        
        if mconfig.multi_rel_outputs:
            models['rel_predictor'] = PointNetRelClsMulti(
                num_rel, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            models['rel_predictor'] = PointNetRelCls(
                num_rel, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
            
            
        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            if len(config.GPU) > 1:# and name != "gcn":
                model = torch.nn.DataParallel(model, config.GPU)
            self.add_module(name, model)
            params += list(model.parameters())
            print(name,op_utils.pytorch_count_params(model))
        print('')
        
        self.optimizer = optim.AdamW(
            params = params,
            lr = float(config.LR),
            weight_decay=self.config.W_DECAY,
            amsgrad=self.config.AMSGRAD
        )
        self.optimizer.zero_grad()
        
        self.scheduler = None
        if self.config.LR_SCHEDULE == 'BatchSize':
            def _scheduler(epoch, batchsize):
                return 1/math.log(batchsize) if batchsize>1 else 1
            self.scheduler = optimizer.BatchMultiplicativeLR(self.optimizer, _scheduler)
        
        self.contrastive_criterion = nn.MarginRankingLoss(margin=0.2)
    def forward(self, segments_points, edges, descriptor, classsname, input_data_list, global_history, pos, neg, timelen, imgs = None, covis_graph = None, return_meta_data=False):
        print("Timelen:{}".format(timelen))
        for num in range(timelen + 1):
           
            
            input_data = input_data_list[num]
            # obj_feature = self.obj_encoder(segments_points[num][:,:,:256]) # 生成pointnet点云特征  248
        

            # if self.mconfig.USE_SPATIAL:
            #     tmp = descriptor[num][:,3:].clone()
            #     tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            #     obj_feature = torch.cat([obj_feature, tmp],dim=1)#cat向量拼接

            ''' Create edge feature '''
            with torch.no_grad():
                edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor[num], input_data.squeeze(1), edges[num].t().contiguous()) # edge feature 仅仅由长度为11的描述符生成边的特征，
            #点云的点仅仅用来pointnet生成点云特征，后与描述符结合成为节点描述符，经过消息传递生成边的特征
            rel_feature = self.rel_encoder(edge_feature)
            #边的特征经过同样的pointnet decoder生成关系特征 edge特征长度为256
            
            ''' GRU '''
            #以下应该出现在network_GRU的forward中，具体应该封装成 obj_h_0, rel_h_0 = GRU(list)
            # obj_feature_embedding = self.obj_encoder()
            # rel_feature_embedding = self.rel_encoder()
            self.layer_norm = False
            if num == 0:
                tick = time.time()
                self.rel_h_0 = rel_feature
                self.obj_h_0 = torch.zeros(96, 512).cuda(0, non_blocking=True)
                input_data = (F.pad(input_data, (0, 0, 0, 0, 0, 96 - input_data.shape[0]))).permute(1,0,2)
                input_data = input_data.squeeze(0)
                self.obj_h_0 = self.obj_GRU_cell(input_data, self.obj_h_0)
                self.obj_h_0 = F.normalize(self.obj_h_0) if self.layer_norm else self.obj_h_0
                tock = time.time()
                # print("obj_GRU {}".format(tock-tick))
            else:
                # if self.rel_h_0.dim() == 3:
                tick = time.time()
                self.rel_h_0 = (self.rel_h_0[:len(edges[num]),:])
                input_data = (F.pad(input_data, (0, 0, 0, 0, 0, 96 - input_data.shape[0]))).permute(1,0,2)
                input_data = input_data.squeeze(0)
                self.obj_h_0 = F.pad(gcn_obj_feature_embedding, (0, 0, 0, 96 - gcn_obj_feature_embedding.shape[0]))
                self.obj_h_0 = self.obj_GRU_cell(input_data, self.obj_h_0)
                self.obj_h_0 = F.normalize(self.obj_h_0) if self.layer_norm else self.obj_h_0
                tock = time.time()
                # print("obj_GRU {}".format(tock-tick))
            
                
            # self.rel_h_0 = F.pad(rel_feature,)
            tick = time.time()
            gcn_obj_feature_embedding, gcn_rel_feature_embedding, probs = self.gcn((self.obj_h_0[:len(classsname[num]),:]), self.rel_h_0, edges[num].t().contiguous())
            gcn_obj_feature_embedding = F.normalize(self.gcn_obj_feature_embedding) if self.layer_norm else gcn_obj_feature_embedding
            tock = time.time()
            print("GCN {}".format(tock-tick))
            if num < timelen:
                tick = time.time()
                self.rel_h_0 = self.rel_GRU_cell(F.pad(rel_feature, ((0, 0, 0, 9216 - rel_feature.shape[0]))), F.pad(gcn_rel_feature_embedding, (0, 0, 0, 9216 - gcn_rel_feature_embedding.shape[0])))
                self.rel_h_0 = F.normalize(self.rel_h_0) if self.layer_norm else self.rel_h_0
                tock = time.time()
                # print("rel_GRU {}".format(tock-tick))
            
            
                
            
            #之后将GRU的两个特征与经过对比学习的全局历史特征拼接，然后进行decoder，全局历史在dataload过程中就已经生成，只是在SGFN的forward中增加对比学习部分
            '''contrastive'''
        con_pos_dis = F.pairwise_distance(torch.from_numpy(global_history), torch.from_numpy(pos), p=2)
        con_neg_dis = F.pairwise_distance(torch.from_numpy(global_history), torch.from_numpy(neg), p=2)
        
                    
        # ''' GNN '''
        # probs=None
        # if self.mconfig.USE_GCN:
        #     if self.mconfig.GCN_TYPE == 'TRIP':
        #         gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edges[num].t().contiguous()) # [node num ,512]  [edge[num] len ,512] 
        #     elif self.mconfig.GCN_TYPE == 'EAN':
        #         gcn_obj_feature, gcn_rel_feature, probs = self.gcn(obj_feature, rel_feature, edges[num].t().contiguous()) # GCN node edge feature
        # else:
        #     gcn_obj_feature=gcn_rel_feature=probs=None     
                    
        ''' Predict '''
        with open('/home/ycb/3DSSG_old/feature.json', 'w') as f:
            json.dump(gcn_obj_feature_embedding.tolist(), f)
        
        obj_cls = self.obj_predictor(gcn_obj_feature_embedding)
        rel_cls = self.rel_predictor(gcn_rel_feature_embedding)

            
        if return_meta_data:
            return obj_cls, rel_cls, input_data, rel_feature, gcn_obj_feature_embedding, gcn_rel_feature_embedding, probs, con_pos_dis, con_neg_dis
        else:
            return obj_cls, rel_cls
    
    
        
    def process(self, obj_points, edges, descriptor, gt_obj_cls, gt_rel_cls, classsname, ULIP_model, timelen, text_features_list, validation_flag, weights_obj=None, weights_rel=None, ignore_none_rel=False,
                imgs = None, covis_graph = None):
        if validation_flag:
            print("validation process")
        else:
            print("train process")
            self.iteration +=1     
        head_list = []
        tail_list = []
        gt_rel_list = []
        # print('=> loading head list & tail list')
        for num  in range(len(classsname)):
            for i in edges[num].t()[0]:
                head_list.append(gt_obj_cls[num][i])
            for i in edges[num].t()[1]:
                tail_list.append(gt_obj_cls[num][i])
        for gt_rel in gt_rel_cls:
            for rel in gt_rel:
                gt_rel_list.append(rel)

        global_history_list = list(zip(head_list, tail_list, gt_rel_list))

        x = np.zeros((20,20,8))
        for triplet in global_history_list:
            try:
                index = triplet[2].tolist().index(1)
                x[triplet[0].tolist()][triplet[1].tolist()][index] = 1
            except ValueError:
                continue
                

        def find_unique_pair(triplets):
            pairs_count = defaultdict(int)
            paris = defaultdict(list)
            for triplet in triplets:
                pair1 = (triplet[0].tolist(), triplet[1].tolist())
                pairs_count[pair1] = 1
                paris[pair1] = triplet
            unique_pairs = [pair for pair, count in pairs_count.items() if count == 1]
            return unique_pairs
     
        # unique_pairs = find_unique_pair(global_history_list)
        
        # '''
        # make pos neg history
        # '''
        # # pos 将有关系的r数值增强
        pos = x.copy()
        pos = pos.astype(int)
        for head in pos:
            for tail in head:
                try:
                    index = tail.tolist().index(1)
                    if np.random.rand() > 0.3:
                        change_value = np.random.randint(1, 5, size=None, dtype=int) 
                        tail[index] = change_value
                except ValueError:
                    continue
        # # neg 删除首尾节点，或者增加首位节点
        neg = x.copy()
        neg = neg.astype(int)
        for head_index, head in enumerate(neg):
            for tail_index, tail in enumerate(head):
                try:
                    index = tail.tolist().index(1)
                    if np.random.rand() > 0.3:
                        if np.random.rand() > 0.5:
                            change_head = np.random.randint(0, 20, size=None, dtype=int)
                            change_value = np.random.randint(1, 5, size=None, dtype=int)
                            neg[change_head][tail_index][index] = change_value
                            neg[head_index][tail_index][index] = 0
                        else:
                            change_tail = np.random.randint(0, 20, size=None, dtype=int)
                            change_value = np.random.randint(1, 5, size=None, dtype=int)
                            neg[head_index][change_tail][index] = change_value
                            neg[head_index][tail_index][index] = 0
                except ValueError:
                    continue
                    
        def get_text_feature(classsname, ULIP_model):
            '''
            text features & pc features
            '''
            tokenizer = SimpleTokenizer()
            print('=> encoding captions')
            with open(os.path.join("/home/ycb/3DSSG/3RScan_gen_data", 'templates.json')) as f:
                templates = json.load(f)['3RScan']
            text_features = []
            text_list = []
            with torch.no_grad():
                for l in classsname:
                    texts = [t.format(l) for t in templates]
                    texts = tokenizer(texts).cuda(0, non_blocking=True)
                    if len(texts.shape) < 2:
                        texts = texts[None,...]
                    text_list.append(texts)
                class_embeddings_list = utils.utils_ULIP.get_model(ULIP_model).encode_text(text_list)
                for class_embeddings in class_embeddings_list:
                    class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                    class_embeddings = class_embeddings.mean(dim=0)
                    class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                    text_features.append(class_embeddings)
                text_features = torch.stack(text_features, dim=0)     # [3,512]
            return text_features
        
        def get_text_feature_from_list(names):
            return torch.stack([text_features_list[name] for name in names])
        method = "single"
        def get_pc_direct(segments_points, ULIP_model):
            if method == "mutil thread":
                '''mutil thread / Process''' 
                thread_list = []
                features_list = []
                device_ids = list(range(torch.cuda.device_count()))
                
                thread_tick = time.time()
                count = 0
                for num in range(len(segments_points)):
                    thread = MyThread(func=self.get_pc_feature, args=(segments_points[num].to(device_ids[num % 4]), ULIP_model.to(device_ids[num % 4]), device_ids[num % 4]))
                    thread.start()
                    count +=1
                    print("start thread:{}".format(num))
                    thread_list.append(thread)
                    # if count == len(segments_points):
                    #     print("wait {}".format(count - 1))
                    #     thread_list[count - 1].join()
                thread_tock = time.time()
                
                pre_tick = time.time()
                for t in thread_list:
                    t.join()
                pre_tock = time.time()
                
                print("Thread creation time:{}".format(thread_tock - thread_tick))
                print("Thread execution time:{}".format(pre_tock - pre_tick))

                for t in thread_list:
                    features_list.append(t.get_result())
                return features_list
            elif method == "direct":   
                '''direct'''
                with torch.no_grad():
                    if len(segments_points) == 10:
                        features_list = []
                        devices = [0,1,2,3]
                        c0 = segments_points[0].permute(0, 2, 1)[:,:,:3].to(devices[0])
                        c1 = segments_points[1].permute(0, 2, 1)[:,:,:3].to(devices[1])
                        c2 = segments_points[2].permute(0, 2, 1)[:,:,:3].to(devices[2])
                        c3 = segments_points[3].permute(0, 2, 1)[:,:,:3].to(devices[3])
                        c4 = segments_points[4].permute(0, 2, 1)[:,:,:3].to(devices[0])
                        c5 = segments_points[5].permute(0, 2, 1)[:,:,:3].to(devices[1])
                        c6 = segments_points[6].permute(0, 2, 1)[:,:,:3].to(devices[2])
                        c7 = segments_points[7].permute(0, 2, 1)[:,:,:3].to(devices[3])
                        c8 = segments_points[8].permute(0, 2, 1)[:,:,:3].to(devices[0])
                        c9 = segments_points[9].permute(0, 2, 1)[:,:,:3].to(devices[1])
                        ULIP_model = torch.nn.DataParallel(ULIP_model, device_ids=devices)
                        ULIP_model.eval()
                        c0 = c0.to(devices[0])
                        c1 = c1.to(devices[1])
                        c2 = c2.to(devices[2])
                        c3 = c3.to(devices[3])
                        c4 = c4.to(devices[0])
                        c5 = c5.to(devices[1])
                        c6 = c6.to(devices[2])
                        c7 = c7.to(devices[3])
                        c8 = c8.to(devices[0])
                        c9 = c9.to(devices[1])
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[0])).encode_pc(c0)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[1])).encode_pc(c1)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[2])).encode_pc(c2)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[3])).encode_pc(c3)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[0])).encode_pc(c4)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[1])).encode_pc(c5)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[2])).encode_pc(c6)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[3])).encode_pc(c7)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[0])).encode_pc(c8)).to(devices[0]))
                        features_list.append((utils.utils_ULIP.get_model(ULIP_model.to(devices[1])).encode_pc(c9)).to(devices[0]))
                    else:
                        for num in range(len(segments_points)):
                            features_list.append(self.get_pc_feature(segments_points[num], ULIP_model, 0))
                    return features_list
            elif method == "single":
                '''single'''
                features_list = []
                for num in range(len(segments_points)):
                    features_list.append(self.get_pc_feature(segments_points[num], ULIP_model, 0))
                return features_list
        text_features_from_list = []
        pc_features_list = []
        tick = time.time()
        # text_list = []
        # tokenizer = SimpleTokenizer()
        # with open(os.path.join("/home/ycb/3DSSG/3RScan_gen_data", 'templates.json')) as f:
        #         templates = json.load(f)['3RScan']
        # for num in range(timelen + 1):
        #     if obj_points[num].shape[0] > 8:
        #         device_ids = [0,1,2,3]
        #     elif obj_points[num].shape[0] > 6 and obj_points[num].shape[0] <= 8:
        #         device_ids = [0,1,2]
        #     elif obj_points[num].shape[0] > 4 and obj_points[num].shape[0] <= 6:
        #         device_ids = [0,1]  
        #     elif obj_points[num].shape[0] <= 4:
        #         device_ids = [0] 
        # device_ids = [0,1] 
        # ULIP_model = nn.DataParallel(ULIP_model, device_ids=device_ids)
        # for num in range(timelen + 1):
        #     with torch.no_grad():
                
        #         # for l in classsname[num]:
        #         #     texts = [t.format(l) for t in templates]
        #         #     texts = tokenizer(texts).cuda(0, non_blocking=True)
        #         #     if len(texts.shape) < 2:
        #         #         texts = texts[None,...]
        #         #     text_list.append(texts)
        #         #     test = torch.stack(text_list, dim = 0)
        #         current_segments_points = obj_points[num].permute(0, 2, 1)
        #         pc_no_normal = current_segments_points[:,:,:3]   
        #         output = ULIP_model.forward(pc = pc_no_normal.to("cuda"), image = None)
        #         # text_features_from_list.append(output["text_embed"])
        #         pc_features_list.append(output["pc_embed"])
        
        
        
        
        
        
        

        input_data_list = []
        for num in range(timelen + 1):
            
            text_tick = time.time()
            text_features_from_list.append(get_text_feature_from_list(classsname[num]))
            text_tock = time.time()
            # print("text representation {}".format(text_tock - text_tick))
            # pc_features_list.append(get_pc_feature(obj_points, ULIP_model))
        pc_tick = time.time()
        
        pc_features_list = get_pc_direct(obj_points, ULIP_model)
        pc_tock = time.time()
        # print("pc representation {}".format(pc_tock - pc_tick))
            
        torch.cuda.empty_cache()
            
        for num, (text_features, pc_features) in enumerate(zip(text_features_from_list, pc_features_list)):
            logits_per_pc = pc_features.cuda(0) @ text_features.t()
            _, indice = torch.topk(logits_per_pc, dim=1, k=1)
            obj_feature1 = []
            for i in range(len(classsname[num])):
                obj_feature1.append(torch.add(pc_features[i].cuda(0), text_features[indice[i]]))
            input_data_list.append(torch.stack(obj_feature1, dim=0)) # [ num_node, 1, 512]
        
        tock = time.time()
        print("unified representation {}".format(tock - tick))
        
        forward_tick = time.time()
        if validation_flag:    
            obj_pred, rel_pred, _, _, _, _, probs, con_pos_dis, con_neg_dis =\
                self.forward(obj_points, edges, descriptor, classsname, input_data_list, x, pos, neg, timelen, return_meta_data=True)
            return obj_pred, rel_pred, _, _, _, _, probs, con_pos_dis, con_neg_dis
        else:
            obj_pred, rel_pred, _, _, _, _, probs, con_pos_dis, con_neg_dis = \
                self(obj_points, edges, descriptor, classsname, input_data_list, x, pos, neg, timelen, return_meta_data=True, imgs=imgs, covis_graph=covis_graph) #model(input)等效model.forward(input)
        forward_tock = time.time()
        print("forward {}".format(forward_tock - forward_tick))
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                # batch_mean = torch.sum(torch.stack([torch.sum(gt_rel, dim=(0))  for gt_rel in gt_rel_cls]), dim=(0))
                batch_mean = torch.sum(gt_rel_cls[timelen], dim=(0))
                # gt_rel_cls1 = torch.cat(gt_rel_cls)
                zeros = (gt_rel_cls[timelen] ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                    # print('set weight of none to 0')
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]
                
            else:
                raise NotImplementedError("unknown weight_edge type")

            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls[timelen], weight=weight)
        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = self.num_rel)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[-1] = 0 # assume none is the last relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            else:
                raise NotImplementedError("unknown weight_edge type")

            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                loss_rel = torch.zeros(1,device=rel_pred.device, requires_grad=False)
            else:
                loss_rel = F.nll_loss(rel_pred, gt_rel_cls, weight = weight)

        loss_obj = F.nll_loss(obj_pred, gt_obj_cls[timelen], weight = weights_obj)
        loss_contrastive = self.contrastive_criterion(con_pos_dis.cuda(0, non_blocking=True), con_neg_dis.cuda(0, non_blocking=True), torch.ones(con_pos_dis.size()).cuda(0, non_blocking=True))
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max
        lambda_con = 0.3

        if 'USE_REL_LOSS' in self.mconfig and not self.mconfig.USE_REL_LOSS:
            loss = loss_obj
        elif 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
            loss = loss_obj
        else:
            loss = lambda_o * loss_obj + lambda_r * loss_rel + lambda_con * loss_contrastive
            
        if self.scheduler is not None:
            self.scheduler.step(batchsize=edges[timelen].shape[1])
        self.backward(loss)
        
        logs = [("Loss/cls_loss",loss_obj.detach().item()),
                ("Loss/rel_loss",loss_rel.detach().item()),
                ("Loss/con_loss",loss_contrastive.item()),
                ("Loss/loss", loss.detach().item())]
        return logs, obj_pred.detach(), rel_pred.detach(), probs
    # @classmethod
    def get_pc_feature(self,segments_points, ULIP_model, device):
        with torch.no_grad():
            torch.cuda.set_device(device)
            current_segments_points = segments_points.permute(0, 2, 1)
            pc_no_normal = current_segments_points[:,:,:3]#(64,8192,3)
            pc_no_normal = pc_no_normal.to(device)
            ULIP_model = ULIP_model.to(device)

            # ULIP_model = torch.nn.DataParallel(ULIP_model, device_ids=device_ids)                
            pc_features = utils.utils_ULIP.get_model(ULIP_model).encode_pc(pc_no_normal) # [3,512]
            # print("end,{}".format(device))
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
        return pc_features
    def backward(self, loss):
        loss.backward()        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def calculate_metrics(self, preds, gts):
        assert(len(preds)==2)
        assert(len(gts)==2)
        obj_pred = preds[0].detach()
        rel_pred = preds[1].detach()
        obj_gt   = gts[0]
        rel_gt   = gts[1]
        
        pred_cls = torch.max(obj_pred.detach(),1)[1]
        acc_obj = (obj_gt[len(obj_gt) - 1] == pred_cls).sum().item() / obj_gt[len(obj_gt) - 1].nelement()
        
        if self.mconfig.multi_rel_outputs:
            pred_rel= rel_pred.detach() > 0.5
            acc_rel = (rel_gt[len(rel_gt) - 1]==pred_rel).sum().item() / rel_gt[len(rel_gt) - 1].nelement()
        else:
            pred_rel = torch.max(rel_pred.detach(),1)[1]
            acc_rel = (rel_gt[len(rel_gt) - 1]==pred_rel).sum().item() / rel_gt[len(rel_gt) - 1].nelement()
            
        
        logs = [("Accuracy/obj_cls",acc_obj), 
                ("Accuracy/rel_cls",acc_rel)]
        return logs
    
    def trace(self,path):
        op_utils.create_dir(path)
        params = dict()
        params['USE_GCN']=self.mconfig.USE_GCN
        params['USE_RGB']=self.mconfig.USE_RGB
        params['USE_NORMAL']=self.mconfig.USE_NORMAL
        params['dim_point']=self.dim_point
        params['dim_edge'] =self.dim_edge
        params["DIM_ATTEN"]=self.mconfig.DIM_ATTEN
        params['obj_pred_from_gcn']=self.mconfig.OBJ_PRED_FROM_GCN
        params['dim_o_f']=self.mconfig.point_feature_size
        params['dim_r_f']=self.mconfig.edge_feature_size
        params['dim_hidden_feature']=self.mconfig.gcn_hidden_feature_size
        params['num_classes']=self.num_class
        params['num_relationships']=self.num_rel
        params['multi_rel_outputs']=self.mconfig.multi_rel_outputs
        params['flow'] = self.flow
        
        self.eval()
        params['enc_o'] = self.obj_encoder.trace(path,'obj')
        params['enc_r'] = self.rel_encoder.trace(path,'rel')
        params['cls_o'] = self.obj_predictor.trace(path,'obj')
        params['cls_r'] = self.rel_predictor.trace(path,'rel')
        if self.mconfig.USE_GCN:
            params['n_layers']=self.gcn.num_layers
            if self.mconfig.GCN_TYPE == 'EAN':
                for i in range(self.gcn.num_layers):
                    params['gcn_'+str(i)] = self.gcn.gconvs[i].trace(path,'gcn_'+str(i))
            elif self.mconfig.GCN_TYPE == 'TRIP':
                for i in range(self.gcn.num_layers):
                    params['gcn_'+str(i)] = self.gcn.gconvs[i].trace(path,'gcn_'+str(i))
            else:
                raise NotImplementedError()
        return params
        
if __name__ == '__main__':
    use_dataset = False
    
    config = Config('../config_example.json')
    
    if not use_dataset:
        num_obj_cls=40
        num_rel_cls=26
    else:
        from src.dataset_builder import build_dataset
        config.dataset.dataset_type = 'point_graph'
        dataset =build_dataset(config, 'validation_scans', True, multi_rel_outputs=True, use_rgb=False, use_normal=False)
        num_obj_cls = len(dataset.classNames)
        num_rel_cls = len(dataset.relationNames)

    # build model
    mconfig = config.MODEL
    network = SGFNModel(config,'SceneGraphFusionNetwork',num_obj_cls,num_rel_cls)
    
    network.trace('./tmp')
    import sys
    sys.exit()

    if not use_dataset:
        max_rels = 80    
        n_pts = 10
        n_rels = n_pts*n_pts-n_pts
        n_rels = max_rels if n_rels > max_rels else n_rels
        obj_points = torch.rand([n_pts,3,128])
        rel_points = torch.rand([n_rels, 4, 256])
        edge_indices = torch.zeros(n_rels, 2,dtype=torch.long)
        counter=0
        for i in range(n_pts):
            if counter >= edge_indices.shape[0]: break
            for j in range(n_pts):
                if i==j:continue
                if counter >= edge_indices.shape[0]: break
                edge_indices[counter,0]=i
                edge_indices[counter,1]=i
                counter +=1
    
    
        obj_gt = torch.randint(0, num_obj_cls-1, (n_pts,))
        rel_gt = torch.randint(0, num_rel_cls-1, (n_rels,))
    
        # rel_gt
        adj_rel_gt = torch.rand([n_pts, n_pts, num_rel_cls])
        rel_gt = torch.zeros(n_rels, num_rel_cls, dtype=torch.float)
        
        
        for e in range(edge_indices.shape[0]):
            i,j = edge_indices[e]
            for c in range(num_rel_cls):
                if adj_rel_gt[i,j,c] < 0.5: continue
                rel_gt[e,c] = 1
            
        network.process(obj_points,edge_indices.t().contiguous(),obj_gt,rel_gt)
        
    for i in range(100):
        if use_dataset:
            scan_id, instance2mask, obj_points, edge_indices, obj_gt, rel_gt = dataset.__getitem__(i)
            
        logs, obj_pred, rel_pred = network.process(obj_points,edge_indices.t().contiguous(),obj_gt,rel_gt)
        logs += network.calculate_metrics([obj_pred,rel_pred], [obj_gt,rel_gt])
        print('{:>3d} '.format(i),end='')
        for log in logs:
            print('{0:} {1:>2.3f} '.format(log[0],log[1]),end='')
        print('')
            
