import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import to_hetero
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
import numpy as np
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
import torch.distributed as dist
from src.utils import seed_worker

from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from src.losses import *
from torch.cuda.amp import autocast
from .utils import count_parameters, AverageMeter, AVGMeter, Reporter, Timer

# torch.autograd.set_detect_anomaly(True)

class Oven():

    def __init__(self, cfg, metadata):
        self.cfg = cfg
        self.ngpus = cfg.get('ngpus', 1)
        if self.ngpus == 0:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        
        if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
            self.reporter = Reporter(cfg)

        self.matrix = self._init_matrix()
        self.train_loader, self.valid_loader = self._init_data()
        self.criterion = self._init_criterion()
        self.model = self._init_model(metadata)
        self.optim, self.scheduler = self._init_optim()


        checkpt_path = self.cfg['model'].get("resume_ckpt_path", "")
        # self.resume_training = True if os.path.exists(os.path.join(self.cfg['log_path'], 'ckpt_latest.pt')) else False
        self.resume_training = True if os.path.exists(checkpt_path) else False
        self.checkpt_path = checkpt_path

        # using ema info
        self.flag_use_ema_model = self.cfg['model'].get("flag_use_ema", False)
        
    def _init_matrix(self):
        if self.cfg['model']['matrix'] == 'vm_va':
            return vm_va_matrix
        else:
            raise TypeError(f"No such of matrix {self.cfg['model']['matrix']}")

    def _init_model(self, metadata):        
        # base base_fusion base_fusion_vnode base_fusion_sffn
        
        if self.cfg['model']['type'] == 'senseflow':
            from src.models.senseflow import IterGCN
            model = IterGCN(**self.cfg['model'])
        elif self.cfg['model']['type'] == 'test_gnn':
            from src.models.powerflow import IterGCN
            model = IterGCN(**self.cfg['model'])
        else:
            raise NotImplementedError('model not support')

        model = model.to(self.device)
        return model

    def _init_training_wt_checkpoint(self, filepath_ckp):
        if not os.path.exists(filepath_ckp):
            return np.Infinity, -1, 0
        
        checkpoint_resum = torch.load(filepath_ckp)
        self.model.load_state_dict(checkpoint_resum['model_state'])
        epoch = checkpoint_resum['epoch']
        previous_best = checkpoint_resum['best_performance']
        previous_best_epoch = checkpoint_resum["best_epoch"]
        return previous_best, previous_best_epoch, epoch

    def _init_optim(self):
        if self.cfg['train'].get("optimizer_type", "Adam").lower() in "adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=float(self.cfg['train']['learning_rate']),
                                   weight_decay=self.cfg['train'].get("weight_decay", 1e-5)
                                   )
        else: # SGD by defalut
            optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.cfg['train']['learning_rate'], 
                                momentum=self.cfg['train'].get("momentum", 0.9), 
                                weight_decay=self.cfg['train'].get("weight_decay", 1e-5))

        # scheduler = StepLR(optimizer, step_size=int(self.cfg['train']['epochs']*2/3), gamma=0.1)
        if self.cfg['scheduler']['type'] == 'Cosine':
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=self.cfg['train']['epochs'],
                                          eta_min=float(self.cfg['scheduler']['eta_min']))
        elif self.cfg['scheduler']['type'] == 'Exponential':
            scheduler = ExponentialLR(optimizer, gamma=self.cfg['scheduler']['gamma'], last_epoch=-1, verbose=False)
        elif self.cfg['scheduler']['type'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-5)
        else: # otherwise: Fixed lr
            scheduler = None
        return optimizer, scheduler

    def _init_criterion(self):
        if self.cfg['loss']['type'] == "deltapq_loss":
            return deltapq_loss
        elif self.cfg['loss']['type'] == "bi_deltapq_loss":
            return bi_deltapq_loss
        elif self.cfg['loss']['type'] == "thetaij_deltapq_loss":
            return thetaij_deltapq_loss
        else:
            raise TypeError(f"No such of loss {self.cfg['loss']['type']}")

    def _init_data(self):
        train_dataset = self.get_dataset(**self.cfg['data']['train'])
        val_dataset = self.get_dataset(**self.cfg['data']['val'])

        if not self.cfg['distributed']:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg['data']['batch_size'],
                num_workers=self.cfg['data']['num_workers'],
                shuffle=True,
                worker_init_fn=seed_worker,
                drop_last=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg['data'].get("batch_size_test", self.cfg['data']['batch_size']),
                num_workers=self.cfg['data']['num_workers'],
                shuffle=False,
                drop_last=True,
                worker_init_fn=seed_worker
            )
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(train_dataset, 
                                  batch_size=self.cfg['data']['batch_size'], 
                                  num_workers=self.cfg['data']['num_workers'], 
                                  sampler=train_sampler,
                                  drop_last=True,
                                  worker_init_fn=seed_worker)
            
            valid_sampler = DistributedSampler(val_dataset, shuffle=False)
            val_loader = DataLoader(val_dataset, 
                                      batch_size=self.cfg['data'].get("batch_size_test", self.cfg['data']['batch_size']), 
                                      num_workers=self.cfg['data']['num_workers'], 
                                      sampler=valid_sampler, 
                                      drop_last=True,
                                      worker_init_fn=seed_worker)

        return train_loader, val_loader

    def get_dataset(self, dataset_type, **kwargs):
        if dataset_type == 'PowerFlowDataset':
            from src.dataset.powerflow_dataset import PowerFlowDataset
            return PowerFlowDataset(
                data_root=kwargs['data_root'],
                split_txt=kwargs['split_txt'],
                pq_len=kwargs['pq_len'],
                pv_len=kwargs['pv_len'],
                slack_len=kwargs['slack_len'],
                mask_num=kwargs['mask_num']
            )

    def exec_epoch(self, epoch, flag, flag_infer_ema=False):
        flag_return_losses = self.cfg.get("flag_return_losses", False)
        if flag == 'train':
            if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
                logger.info(f'-------------------- Epoch: {epoch+1} --------------------')
            self.model.train()
            if self.cfg['distributed']:
                self.train_loader.sampler.set_epoch(epoch)
            
            # record vars
            train_loss = AVGMeter()
            train_matrix = dict()
            total_batch = len(self.train_loader)
            print_period = self.cfg['train'].get('logs_freq', 8)
            print_freq = total_batch // print_period 
            print_freq_lst = [i * print_freq for i in range(1, print_period)] + [total_batch - 1]
            
            # start loops
            for batch_id, batch in enumerate(self.train_loader):
                # data
                batch.to(self.device, non_blocking=True)
                
                # forward
                self.optim.zero_grad()
                if flag_return_losses:
                    pred, loss, record_losses = self.model(batch, flag_return_losses=True)
                else:
                    pred, loss = self.model(batch)

                # records
                cur_matrix = self.matrix(pred)
                if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
                    # logger.info(f"Iter:{batch_id}/{total_batch} - {str(cur_matrix)}")
                    # print(cur_matrix)
                    pass
                if batch_id == 0:
                    for key in cur_matrix:
                        train_matrix[key] = AVGMeter()

                for key in cur_matrix:
                    train_matrix[key].update(cur_matrix[key])
                
                # backwards
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                train_loss.update(loss.item())

                # update ema
                if self.flag_use_ema_model:
                    if self.cfg['distributed']:
                        self.model.module.update_ema_model(epoch, batch_id + epoch * total_batch, total_batch)
                    else:
                        self.model.update_ema_model(epoch, batch_id + epoch * total_batch, total_batch)

                # print stats
                if (batch_id in print_freq_lst) or ((batch_id + 1) == total_batch):
                    if self.cfg['distributed']:
                        if dist.get_rank() == 0:
                            if flag_return_losses:
                                ret_loss_str = " ".join(["{}:{:.5f}".format(x, y) for x,y in record_losses.items()])
                                logger.info(f"Epoch[{str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}], iter[{str(batch_id+1).zfill(3)}/{total_batch}], loss_total:{loss.item():.5f}, {ret_loss_str}")
                            else:
                                logger.info(f"Epoch[{str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}], iter[{str(batch_id+1).zfill(3)}/{total_batch}], loss_total:{loss.item():.5f}")
                    else:
                        if flag_return_losses:
                            ret_loss_str = " ".join(["{}:{:.5f}".format(x, y) for x,y in record_losses.items()])
                            logger.info(f"Epoch[{str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}], iter[{str(batch_id+1).zfill(3)}/{total_batch}], loss_total:{loss.item():.5f}, {ret_loss_str}")
                        else:
                            logger.info(f"Epoch[{str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}], iter[{str(batch_id+1).zfill(3)}/{total_batch}], loss_total:{loss.item():.5f}")
            return train_loss, train_matrix
        elif flag == 'valid':
            n_loops_test = self.cfg['model'].get("num_loops_test", 1)
            self.model.eval()
            if self.cfg['distributed']:
                world_size = dist.get_world_size()
                self.valid_loader.sampler.set_epoch(epoch)

            valid_loss = AVGMeter()
            val_matrix = dict()
            # start data loops
            with torch.no_grad():
                for batch_id, batch in enumerate(self.valid_loader):
                    batch.to(self.device)
                    if self.flag_use_ema_model:
                        pred, loss = self.model(batch, num_loop_infer=n_loops_test, flag_use_ema_infer=flag_infer_ema)
                    else:
                        pred, loss = self.model(batch, num_loop_infer=n_loops_test)
                    cur_matrix = self.matrix(pred, mode='val')
                    # collect performance 1 --- matrix
                    if self.cfg['distributed']:
                        # get all res from multiple gpus 
                        for key in cur_matrix:
                            # tmp_value = cur_matrix[key].clone().detach().requires_grad_(False).cuda()
                            tmp_value = torch.tensor(cur_matrix[key]).cuda()
                            dist.all_reduce(tmp_value)
                            cur_matrix[key] = tmp_value.cpu().item() / world_size
                    if batch_id == 0: # record into val_matrix
                        for key in cur_matrix:
                            val_matrix[key] = AVGMeter()
                    for key in cur_matrix:
                            val_matrix[key].update(cur_matrix[key])
                    # collect performance 2 --- loss
                    if self.cfg['distributed']:
                        tmp_loss = loss.clone().detach()
                        dist.all_reduce(tmp_loss)
                        valid_loss.update(tmp_loss.cpu().item() / world_size)
                    else:
                        valid_loss.update(loss.cpu().item())
            
            return valid_loss, val_matrix
        else:
            raise ValueError(f'flag == {flag} not support, choice[train, valid]')

    def summary_epoch(self,
                      epoch,
                      train_loss, train_matrix,
                      valid_loss, val_matrix,
                      timer, local_best, 
                      local_best_ep=-1,
                      local_best_ema=100, local_best_ep_ema=-1,
                      valid_loss_ema=None, val_matrix_ema=None):
        
        if self.cfg['distributed']:
            if dist.get_rank() == 0:
                cur_lr = self.optim.param_groups[0]["lr"]
                # self.reporter.record({'epoch': epoch+1, 'train_loss': train_loss, 'valid_loss': valid_loss, 'lr': cur_lr})
                self.reporter.record({'loss/train_loss': train_loss}, epoch=epoch)
                self.reporter.record({'loss/val_loss': valid_loss}, epoch=epoch)
                self.reporter.record({'lr': cur_lr}, epoch=epoch)
                self.reporter.record(train_matrix, epoch=epoch)
                self.reporter.record(val_matrix, epoch=epoch)

                # logger.info(f"Epoch {str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}, lr: {cur_lr: .8f}, eta: {timer.eta}h, train_loss: {train_loss: .5f}, valid_loss: {valid_loss: .5f}")
                logger.info(f"Epoch {str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']},"
                        + f" lr: {cur_lr: .8f}, eta: {timer.eta}h, "
                        + f"train_loss: {train_loss.agg(): .5f}, "
                        + f"valid_loss: {valid_loss.agg(): .5f}")
                
                train_matrix_info = "Train: "
                for key in train_matrix.keys():
                    tkey = str(key).split("/")[-1]
                    train_matrix_info += f"{tkey}:{train_matrix[key].agg(): .6f}  "
                logger.info(f"\t{train_matrix_info}")

                val_matrix_info = "ZTest: "
                performance_record = dict()
                for key in val_matrix.keys():
                    tkey = str(key).split("/")[-1]
                    val_matrix_info += f"{tkey}:{val_matrix[key].agg(): .6f}  "
                    performance_record[key] = val_matrix[key].agg()
                logger.info(f"\t{val_matrix_info}")

                if val_matrix_ema is not None:
                    val_matrix_info_ema = "ZTest-ema: "
                    performance_record_ema = dict()
                    for key in val_matrix_ema.keys():
                        tkey = str(key).split("/")[-1]
                        val_matrix_info_ema += f"{tkey}:{val_matrix_ema[key].agg(): .6f}  "
                        performance_record_ema[key] = val_matrix_ema[key].agg()
                    logger.info(f"\t{val_matrix_info_ema}")

                    checked_performance_ema = {x:y for x,y in performance_record_ema.items() if "rmse" in x}
                    best_performance_ema = max(checked_performance_ema.values())
                    if best_performance_ema < local_best_ema:
                        local_best_ema = best_performance_ema
                        local_best_ep_ema = epoch
                    logger.info(f"\t           ValOfEMA:{best_performance_ema:.6f}/{local_best_ema:.6f},  Epoch:{epoch+1}/{local_best_ep_ema+1}")
                
                # best_performance = max(performance_record.values())
                checked_performance = {x:y for x,y in performance_record.items() if "rmse" in x}
                best_performance = max(checked_performance.values())
                if best_performance < local_best:
                    local_best = best_performance
                    local_best_ep = epoch
                    # torch.save(self.model.module, os.path.join(self.cfg['log_path'], 'ckpt_{}_{}.pt'.format(epoch, round(local_best,4))))
                    torch.save(self.model.module, os.path.join(self.cfg['log_path'], 'ckpt_best.pt'))
                
                state = {
                    "epoch": epoch + 1,
                    # "model_state": self.model.module.state_dict(),
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optim.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "best_performance": local_best,
                    "best_epoch":local_best_ep,
                }
                torch.save(state, os.path.join(self.cfg['log_path'], 'ckpt_latest.pt'))
                logger.info(f"\tTime(ep):{int(timer.elapsed_time)}s,  Val(curr/best):{best_performance:.6f}/{local_best:.6f},  Epoch(curr/best):{epoch+1}/{local_best_ep+1}")
            # else:
            #     return local_best, local_best_ep
        else:
            cur_lr = self.optim.param_groups[0]["lr"]
            self.reporter.record({'loss/train_loss': train_loss}, epoch=epoch)
            self.reporter.record({'loss/val_loss': valid_loss}, epoch=epoch)
            self.reporter.record({'lr': cur_lr}, epoch=epoch)
            self.reporter.record(train_matrix, epoch=epoch)
            self.reporter.record(val_matrix, epoch=epoch)

            logger.info(f"Epoch {epoch}/{self.cfg['train']['epochs']},"
                        + f" lr: {cur_lr: .8f}, eta: {timer.eta}h, "
                        + f"train_loss: {train_loss.agg(): .5f}, "
                        + f"valid_loss: {valid_loss.agg(): .5f}")

            train_matrix_info = "Train: "
            for key in train_matrix.keys():
                tkey = str(key).split("/")[-1]
                train_matrix_info += f"{tkey}:{train_matrix[key].agg(): .8f}  "
            logger.info(f"\t{train_matrix_info}")

            val_matrix_info = "ZTest: "
            performance_record = dict()
            for key in val_matrix.keys():
                tkey = str(key).split("/")[-1]
                val_matrix_info += f"{tkey}:{val_matrix[key].agg(): .8f}  "
                performance_record[key] = val_matrix[key].agg()
            logger.info(f"\t{val_matrix_info}")

            if val_matrix_ema is not None:
                val_matrix_info_ema = "ZTest-ema: "
                performance_record_ema = dict()
                for key in val_matrix_ema.keys():
                    tkey = str(key).split("/")[-1]
                    val_matrix_info_ema += f"{tkey}:{val_matrix_ema[key].agg(): .6f}  "
                    performance_record_ema[key] = val_matrix_ema[key].agg()
                logger.info(f"\t{val_matrix_info_ema}")
                
                checked_performance_ema = {x:y for x,y in performance_record_ema.items() if "rmse" in x}
                best_performance_ema = max(checked_performance_ema.values())
                if best_performance_ema < local_best_ema:
                    local_best_ema = best_performance_ema
                    local_best_ep_ema = epoch
                logger.info(f"\t           ValOfEMA:{best_performance_ema:.6f}/{local_best_ema:.6f},  Epoch:{epoch+1}/{local_best_ep_ema+1}")

            # best_performance = max(performance_record)
            checked_performance = {x:y for x,y in performance_record.items() if "rmse" in x}
            best_performance = max(checked_performance.values())
            if best_performance < local_best:  # save best
                local_best = best_performance
                local_best_ep = epoch
                # torch.save(self.model, os.path.join(self.cfg['log_path'], 'ckpt_{}_{}.pt'.format(epoch, round(local_best,4))))
                torch.save(self.model, os.path.join(self.cfg['log_path'], 'ckpt_best.pt'))
            state = {
                "epoch": epoch + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optim.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_performance": local_best,
                "best_epoch":local_best_ep,
            }
            torch.save(state, os.path.join(self.cfg['log_path'], 'ckpt_latest.pt'))
            logger.info(f"\tTime(ep):{int(timer.elapsed_time)}s,  Val(curr/best):{best_performance:.6f}/{local_best:.6f},  Epoch(curr/best):{epoch+1}/{local_best_ep+1}")
        
        if val_matrix_ema is not None:
            return local_best, local_best_ep, local_best_ema, local_best_ep_ema    
        else:
            return local_best, local_best_ep

    def train(self):
        if self.ngpus > 1:
            dummy_batch_data = next(iter(self.train_loader))
            dummy_batch_data.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.flag_use_ema_model:
                    _ = self.model(dummy_batch_data, num_loop_infer=1)
                    _ = self.model(dummy_batch_data, num_loop_infer=1, flag_use_ema_infer=True)
                else:
                    _ = self.model(dummy_batch_data, num_loop_infer=1)
            
            if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
                logger.info(f'==================== Total number of parameters: {count_parameters(self.model):.3f}M')

            local_rank = int(os.environ["LOCAL_RANK"])
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                #  find_unused_parameters=False
            )
        else:
            dummy_batch_data = next(iter(self.train_loader))
            dummy_batch_data.to(self.device, non_blocking=True)
            with torch.no_grad():
                # _ = self.model(dummy_batch_data, num_loop_infer=1)
                if self.flag_use_ema_model:
                    _ = self.model(dummy_batch_data, num_loop_infer=1)
                    _ = self.model(dummy_batch_data, num_loop_infer=1, flag_use_ema_infer=True)
                else:
                    _ = self.model(dummy_batch_data, num_loop_infer=1)
            logger.info(f'==================== Total number of parameters: {count_parameters(self.model):.3f}M')

        
        if not self.resume_training:    
            self.perform_best = np.Infinity
            self.perform_best_ep = -1
            self.start_epoch = 0
        else:
            self.perform_best, self.perform_best_ep, self.start_epoch = self._init_training_wt_checkpoint(self.checkpt_path)
        
        local_best = self.perform_best
        local_best_ep = self.perform_best_ep

        if self.flag_use_ema_model:
            local_best_ema = self.perform_best
            local_best_ep_ema = self.perform_best_ep

        for epoch in range(self.start_epoch, self.cfg['train']['epochs']):
            with Timer(rest_epochs=self.cfg['train']['epochs'] - (epoch + 1)) as timer:
                train_loss, train_matrix = self.exec_epoch(epoch, flag='train')
                valid_loss, val_matrix = self.exec_epoch(epoch, flag='valid')
                if self.flag_use_ema_model:
                    valid_loss_ema, valid_matrix_ema = self.exec_epoch(epoch, flag='valid', 
                                                             flag_infer_ema=True)
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(valid_loss.agg())
                    else:
                        self.scheduler.step()
            if self.flag_use_ema_model:
                local_best, local_best_ep, local_best_ema, local_best_ep_ema = self.summary_epoch(epoch,
                                            train_loss, train_matrix,
                                            valid_loss, val_matrix,
                                            timer, local_best, local_best_ep, 
                                            local_best_ema=local_best_ema, 
                                            local_best_ep_ema=local_best_ep_ema,
                                            valid_loss_ema=valid_loss_ema, 
                                            val_matrix_ema=valid_matrix_ema)
            else:
                local_best, local_best_ep = self.summary_epoch(epoch,
                                            train_loss, train_matrix,
                                            valid_loss, val_matrix,
                                            timer, local_best, local_best_ep)

        if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
            self.reporter.close()
