import numpy as np
import matplotlib.pyplot as plt

def _log_plot(log_path):
    train_loss = []
    val_loss = []
    # 根据Log文件进行画图
    for line in open(log_path,"r",encoding='UTF-8'):
        train_loss_data = line.split('Train_Avegrage_loss: ')[-1]
        if train_loss_data[0:4] != str(2022):
            train_loss.append(float(train_loss_data[0:6]))
        
    # 创建一个绘图窗口
    plt.figure()
    
    epochs = range(len(train_loss))
    
    # plt.plot(epochs, acc, 'bo', label='Training acc') # 'bo'为画蓝色圆点，不连线
    # plt.plot(epochs, val_acc, 'b', label='Validation acc') 
    # plt.title('Training and validation accuracy')
    # plt.legend() # 绘制图例，默认在右上角
    
    plt.figure()
    
    plt.plot(epochs, train_loss,marker = "o",markersize=2, label='Training loss')
    #plt.plot(epochs, val_loss,marker = "o",markersize=2, label='Validation loss')
    plt.title('Training loss')
    plt.legend()
    
    plt.show()
    plt.savefig('imgs/plot_data/loss2.png')
    
_log_plot('logs/RefineNet-21-lr-0.003-20221017-00.log')