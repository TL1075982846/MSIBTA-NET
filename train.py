import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import StepLR

from model3 import resnet50
#from SE_resnet import se_resnet50
#from SE_TripAtt_resnet import se_resnet50
#from SE_TripAtt_resnet2 import se_resnet50
#from SE_TripAtt_resnet3 import se_resnet50
#from se_change_residual_resnet import se_resnet50
import matplotlib.pyplot as plt
from torchstat import stat

from confusion import  ConfusionMatrix

leaf_dict = {'0': 'Alternaria_Boltch', '1': 'Brown_Spot', '2': 'Grey_spot', '3': 'Healthy', '4': 'Mosaic','5': 'Rust', '6': 'Scab'}
label = [label for _,label in leaf_dict.items()]
confusion = ConfusionMatrix(num_classes=7, labels=label)

Acc,Loss=[],[]

writer = SummaryWriter("runs")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "apple")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    #net = se_resnet50()
    net =resnet50()

    total = sum([param.nelement() for  param in net.parameters()])
    print(("Number of parameters: %.2fM" %(total/1e6)))
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    #model_weight_path = "./resnet34-pre.pth"
    #assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    #net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 7)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    #scheduler = StepLR(optimizer,step_size=25,gamma=0.5)


    epochs = 200
    best_acc = 0.0
    save_path = './se-resnet.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))

            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            torch.cuda.empty_cache()


        #scheduler.step()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                confusion.update(predict_y.cpu().numpy(),labels.cpu().numpy())
            confusion.plot()
            confusion.summary()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        #Loss.append(running_loss / train_steps)
        #Acc.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        writer.add_scalars("Loss", {"Train": running_loss / train_steps}, epoch)
        writer.add_scalars("Accuracy", {"Val": val_accurate}, epoch)

   # plt.xlabel("Epoch", fontsize=14)
   # plt.ylabel("Value", fontsize=18)
   # plt.plot(Acc, color="red", label="val accuracy")
   # plt.plot(Loss, color="green", label="train loss")
   # plt.legend()

   # plt.rcParams['savefig.dpi'] = 1024  # 像素
   # plt.rcParams['figure.dpi'] = 1024  # 分辨率
   # plt.savefig('F:/WORK/code/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet/result/change-acc-loss.png')

    plt.show()

    print('Finished Training')


if __name__ == '__main__':
    main()


