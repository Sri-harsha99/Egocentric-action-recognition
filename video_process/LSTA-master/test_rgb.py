from __future__ import print_function, division
from attentionModel import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from tensorboardX import SummaryWriter
from makeDataset import *
import sys
import argparse
from gen_splits import *

#TODO: create separate dirs for stage1 and stage 2

def main_run(dataset, root_dir, checkpoint_path, seqLen, testBatchSize, memSize, outPool_size, split):

    root_dir='D:/study/btp-project/btp/video_process/LSTA-master/dataset'

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)

    test_split = split
    seqLen = seqLen
    memSize = memSize
    c_cam_classes = outPool_size
    dataset = dataset
    testBatchSize = testBatchSize
    checkpoint_path = checkpoint_path

    targetPath = 'video_process/LSTA-MASTER/dataset/gtea_61/frames/S2'
    files = os.listdir(targetPath)
    target_action_names=[]
    for name in files:
        target_action_names.append(name)
    
    # print(target_action_names)
    
    predicted_action_names=['close_chocolate', 'close_coffee', 'close_honey', 'close_jam', 'close_ketchup', 'close_mayonnaise', 'close_mustard', 'close_peanut', 'close_sugar', 'close_water', 'fold_bread', 'open', 'open_cheese', 'open_chocolate', 'open_coffee', 'open_honey', 'open_jam', 'open_ketchup', 'open_mayonnaise', 'open_mustard', 'open_peanut', 'open_sugar', 'open_tea', 'open_water', 'pour_chocolate,bread', 'pour_coffee,spoon,cup', 'pour_honey,bread', 'pour_honey,cup', 'pour_ketchup,hotdog,bread', 'pour_mayonnaise,cheese,bread', 'pour_mustard,cheese,bread', 'pour_mustard,hotdog,bread', 'pour_sugar,spoon,cup', 'pour_water,cup', 'put_bread,bread', 'put_bread,cheese,bread', 'put_cheese,bread', 'put_hotdog,bread', 'scoop_coffee,spoon', 'scoop_jam,spoon', 'scoop_peanut,spoon', 'scoop_sugar,spoon', 'shake_tea,cup', 'spread_jam,spoon,bread', 'spread_peanut,spoon,bread', 'stir_spoon,cup', 'take_bread', 'take_cheese', 'take_chocolate', 'take_coffee', 'take_cup', 'take_honey', 'take_hotdog', 'take_jam', 'take_ketchup', 'take_mayonnaise', 'take_mustard', 'take_peanut', 'take_spoon', 'take_sugar', 'take_tea', 'take_water']
        
    if dataset == 'gtea_61':
        num_classes = 61
    elif dataset == 'gtea_71':
        num_classes = 71
    elif dataset == 'egtea_gaze+':
        num_classes = 106
    else:
        # print('Wrong dataset')
        sys.exit()
    dataset_dir = os.path.join(root_dir, dataset)
    # print('Preparing dataset...')

    if dataset == 'egtea_gaze+':
        trainDatasetF, testDatasetF, trainLabels, testLabels, trainNumFrames, testNumFrames = gen_split_egtea_gazePlus(dataset_dir,
                                                                                                               test_split)
    else:
        trainDatasetF, testDatasetF, trainLabels, testLabels, trainNumFrames, testNumFrames, _ = gen_split(dataset_dir,
                                                                                                  test_split)


    vid_seq_test = makeDataset(testDatasetF, testLabels, testNumFrames,
                               spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                               fmt='.jpg', seqLen=seqLen)

    # print('Number of test samples = {}'.format(vid_seq_test.__len__()))


    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=testBatchSize,
                            shuffle=False, num_workers=0, pin_memory=True)


    model = attentionModel(num_classes=num_classes, mem_size=memSize, c_cam_classes=c_cam_classes)
    # if os.path.exists(checkpoint_path):
    #         # print('Loading weights from checkpoint file {}'.format(checkpoint_path))
    # else:
    #         print('Checkpoint file {} does not exist'.format(checkpoint_path))
    #         sys.exit()
    last_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(last_checkpoint['model_state_dict'])
    model.cuda()
    model.train(False)

    # print('Testing...')
    test_iter = 0
    test_samples = 0
    numCorr = 0
    for j, (inputs, targets) in enumerate(test_loader):
        # print('testing inst = {}'.format(j))
        
        test_iter += 1
        test_samples += inputs.size(0)
        inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda(), volatile=True)
        output_label, _ = model(inputVariable)
        _, predicted = torch.max(output_label.data, 1)
        
        numCorr += (predicted == targets.cuda()).sum()
        inp = test_loader.dataset.__getitem__(j)
        # print("target: ")
        # print(target_action_names[targets.cpu().numpy()[0]])
        # print("predicted: ")
        print(predicted_action_names[predicted.cpu().numpy()[0]])
        return predicted_action_names[predicted.cpu().numpy()[0]]
    # print(numCorr)
    # test_accuracy = torch.true_divide(numCorr,test_samples) * 100
    # print('Test Accuracy after = {}%'.format(test_accuracy))
    

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--checkpoint', type=str, default='model_epoch150.pth.tar', help='Dataset')
    parser.add_argument('--root_dir', type=str, default='./dataset/',
                        help='Dataset directory')
    parser.add_argument('--seqLen', type=int, default=18, help='Length of sequence')
    parser.add_argument('--testBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--outPoolSize', type=int, default=100, help='Output pooling size')
    parser.add_argument('--split', type=int, default=1, help='Split')

    args = parser.parse_args()

    dataset = args.dataset
    checkpoint = args.checkpoint
    root_dir = args.root_dir
    seqLen = args.seqLen
    testBatchSize = args.testBatchSize
    memSize = args.memSize
    outPool_size = args.outPoolSize
    split = args.split

    main_run(dataset=dataset, root_dir=root_dir, seqLen=seqLen, testBatchSize=testBatchSize, memSize=memSize,
             outPool_size=outPool_size, split=split, checkpoint_path=checkpoint)

__main__()