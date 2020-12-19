from torch.utils.tensorboard import SummaryWriter
import datetime


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, par):
        self.log_dir = 'logs/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'-'+par
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.acc_all = AverageMeter()
        self.mAP_all = AverageMeter()
        self.mAR_all = AverageMeter()

        self.last_log = {
            'loss_total': 0,
            'lr': 0,
            'train_time': 0,
            'step': 0
        }

    def update(self, k, v, iter):
        self.writer.add_scalar(k, v, iter)

    def update_all(self, v_dict, iter):
        for key, value in v_dict.items():
            self.update(key, value, iter)
            self.last_log[key] = value
        self.last_log['step'] = iter

    def update_acc(self, acc_list, iter):
        for i, value in enumerate(acc_list):
            self.acc_all.update(value)
            if i < 6:
                self.mAP_all.update(value)
                key = 'mAP' + str(i)
            else:
                self.mAR_all.update(value)
                key = 'mAR' + str(i)
            self.update(key, value, iter)
        self.update('Acc.avg', self.acc_all.avg, iter)
        self.update('mAP.avg', self.mAP_all.avg, iter)
        self.update('mAR.avg', self.mAR_all.avg, iter)

    def show_last(self):
        # TODO: correct format of acc
        print('Step: {},\tTotal_loss: {},\tAcc_avg: {},\tmAP_avg: {},\t\tmAR_avg: {},\tlr: {},\tTrain_time: {:0.2f}'.
              format(self.last_log['step'], self.last_log['loss_total'], self.acc_all.avg, self.mAP_all.avg,
                     self.mAR_all.avg, self.last_log['lr'], self.last_log['train_time']))

    def close(self):
        self.writer.close()


if __name__ == '__main__':
    print('Ready.')