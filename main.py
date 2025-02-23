import time
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm  

import config
from model import VGG, CNN, WeightedCombinationLoss
from utils import GetData, chebyshev_distance, canberra_distance, cosine_similarity, acc


if __name__ == '__main__':
    print("Training started...")
    train_set = GetData(config.PATH + r"/train.tf")
    test_set = GetData(config.PATH + r"/test.tf")
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=True)
    device = torch.device("cuda" if config.USE_GPU else "cpu")

    # net = CNN(num_classes=config.NUM_CLASSES)
    net = VGG(num_classes=config.NUM_CLASSES)
    loss_function = WeightedCombinationLoss(lmd=config.LMD)
    # optimizer：全连接层lr=0.01 卷积层lr=0.001
    # 设置不同层不同的lr
    optimizer = torch.optim.SGD([
        {'params': net.features.parameters()},
        {'params': net.classifier.parameters(), 'lr': config.LR_FC},
    ], lr=config.LR_CNN, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)

    if config.USE_GPU:
        net.to(device)
        loss_function.to(device)

    start_tick = time.time()
    for ep in range(1, config.EPOCH + 1):
        net.train()  # 设置模型为训练模式
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {ep}/{config.EPOCH}", unit="batch") as tepoch:
            for img, label in tepoch:
                if config.USE_GPU:
                    img = img.to(device)
                    label = label.to(device)

                out = net(img)  # 前向传播
                loss = loss_function(out, label)  # 计算损失
                epoch_loss += loss.item()

                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                # 更新 tqdm 的描述信息
                tepoch.set_postfix(loss=loss.item())

        scheduler.step()  # 更新学习率

        # 测试阶段
        net.eval()  # 设置模型为评估模式
        num_c_score = num_acc = num_ca_score = num_cos_score = 0  # 初始化各种score
        with torch.no_grad():
            for img, label in tqdm(test_loader, desc="Testing", unit="batch"):
                if config.USE_GPU:
                    img = img.to(device)
                    label = label.to(device)

                prediction = net(img)  # 获得输出
                num_c_score += chebyshev_distance(prediction, label)  # 计算得分的函数
                num_ca_score += canberra_distance(prediction, label)  # 计算得分的函数
                num_cos_score += cosine_similarity(prediction, label)  # 计算得分的函数
                num_acc += acc(prediction, label)

        c_score = num_c_score.cpu().detach().numpy() / len(test_set)  # 计算正确率，先转换成cpu变量
        ca_score = num_ca_score.cpu().detach().numpy() / len(test_set)  # 计算正确率，先转换成cpu变量
        cos_score = num_cos_score.cpu().detach().numpy() / len(test_set)  # 计算正确率，先转换成cpu变量
        t_acc = num_acc.cpu().detach().numpy() / len(test_set)

        timeSpan = time.time() - start_tick

        # 打印结果
        print(f"Epoch {ep}/{config.EPOCH}:")
        print(f"切比雪夫距离: {c_score:.4f}")
        print(f"堪培拉距离: {ca_score:.4f}")
        print(f"余弦相似度: {cos_score:.4f}")
        print(f"准确性: {t_acc:.4f}")
        print(f"已用时间: {timeSpan:.2f}s")

        # 保存模型
        model_save_path = config.SAVE_FILE + str(ep)
        torch.save(net.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}\n")