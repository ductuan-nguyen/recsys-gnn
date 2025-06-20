from absl import app
from absl import flags

from train import TrainManager
from utils import VisManager, ContentManager
from torch.utils.data import DataLoader

FLAGS = flags.FLAGS

#all model flags
flags.DEFINE_enum('model','MF',['MF','MBGCN'],'Model name')
flags.DEFINE_string('name','MF-experiment','Experiment name')
flags.DEFINE_string('dataset_name','Tmall_release',"Dataset's name")
flags.DEFINE_float('lr',0.0001,'Learning rate')
flags.DEFINE_float('L2_norm',0.001,'L2 norm')
flags.DEFINE_bool('gpu','True','Use GPU or not')
flags.DEFINE_integer('gpu_id',6,'GPU ID')
flags.DEFINE_integer('num_workers', 8, 'Number of processes for training and testing')
flags.DEFINE_integer('epoch', 400, 'Max epochs for training')
flags.DEFINE_string('path','/home/tuannd/MBGCN/','The path where the data is')
flags.DEFINE_string('output','/home/tuannd/MBGCN/output','The path to store output message')
flags.DEFINE_integer('port',33337,'Port to show visualization results for visdom')
flags.DEFINE_integer('batch_size',2048,'Batch size')
flags.DEFINE_integer('test_batch_size',512,'Test batch size')
flags.DEFINE_integer('embedding_size',32,'Embedding Size')
flags.DEFINE_integer('es_patience',10,'Early Stop Patience')
flags.DEFINE_enum('loss_mode','mean',['mean','sum'],'Loss Mode')
flags.DEFINE_multi_string('relation', ['buy','wanted','clicked','detail_view'], 'Relations')
#pretrain model flags
flags.DEFINE_bool('pretrain_frozen','False','Froze the pretrain parameter or not')
flags.DEFINE_string('create_embeddings','True','Pretrain or not? If not create embedding here!')
flags.DEFINE_string('pretrain_path','/home/tuannd/MBGCN/output/Steam/MF-Steam-lr3-L4@jinbowen',"Path where the pretrain model is.")
#graph based
flags.DEFINE_float('node_dropout',0.2,'Node dropout ratio')
flags.DEFINE_float('message_dropout',0.2,'Message dropout ratio')
#Ours
flags.DEFINE_float('lamb',0.1,'Lambda for the loss for MultiGNN with item space calculation')
flags.DEFINE_multi_float('mgnn_weight',[1,1,1,1],'Weight for MGNN')
flags.DEFINE_string('trained_mbgcn_model','/home/tuannd/MBGCN/output/Recsys2025/Recsys2025-MBGCN_lr3e-4-L1e-4-size64-lamb1-md0.2-nd0.2-3-relation@tuannd/model_epoch_6.pkl','trained model for loading')
flags.DEFINE_string('output_test','submits/exp_add_remove_relation','Output path for test result')
flags.DEFINE_string('mode','train','Mode of the program, train or test')

def main(argv):

    flags_obj = FLAGS
    vm = VisManager(flags_obj)
    cm = ContentManager(flags_obj)
    Train = TrainManager(flags_obj, vm, cm)

    if flags_obj.mode == 'train':
        Train.train()
    if flags_obj.mode == 'test':
        Train.test()
    # Train.test()

if __name__=='__main__':

    app.run(main)
