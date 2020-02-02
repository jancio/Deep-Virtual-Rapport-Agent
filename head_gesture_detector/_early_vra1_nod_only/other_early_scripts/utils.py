from matplotlib import pyplot as plt
import numpy as np
import pickle


def plot_train_history(hist):
    
    print(f'\tMin train loss: {np.min(hist.history["loss"])} @epoch {np.argmin(hist.history["loss"])} \t Max train acc: {np.max(hist.history["acc"])} @epoch {np.argmax(hist.history["acc"])}')
    print(f'\tMin valid loss: {np.min(hist.history["val_loss"])} @epoch {np.argmin(hist.history["val_loss"])} \t Max valid acc: {np.max(hist.history["val_acc"])} @epoch {np.argmax(hist.history["val_acc"])}')
    
    # Loss Curves
    plt.figure(figsize=[16,6])
    plt.subplot(121)
    plt.plot(hist.history['loss'],'ro-')#,linewidth=2.0)
    plt.plot(hist.history['val_loss'],'bo-')#,linewidth=2.0)
    plt.legend(['Training loss', 'Validation loss'])#,fontsize=18)
    # plt.xticks(x, x)
    plt.xlabel('Epoch')#,fontsize=16)
    plt.ylabel('Loss')#,fontsize=16)
    # plt.ylim(0.35, 0.95)
    # plt.title('Loss Curves',fontsize=16)
#     plt.show()

    # Accuracy Curves
    plt.subplot(122)
    plt.plot(hist.history['acc'],'ro-')#,linewidth=2.0)
    plt.plot(hist.history['val_acc'],'bo-')#,linewidth=2.0)
    plt.legend(['Training accuracy', 'Validation accuracy'])#,fontsize=18)
    # plt.xticks(x, x)
    plt.xlabel('Epoch')#,fontsize=16)
    plt.ylabel('Accuracy')#,fontsize=16)
    # plt.title('Accuracy Curves',fontsize=16)
    # plt.ylim(0.35, 0.95)
    plt.show()
    
def save_train_history(hist, hist_filename):
    with open(f'./checkpoints/{hist_filename}/{hist_filename}.pkl', 'wb') as pickle_filehandler:
        pickle.dump(hist, pickle_filehandler)
          
def load_train_history(hist_filename):   
    with open(f'./checkpoints/{hist_filename}/{hist_filename}.pkl', 'rb') as pickle_filehandler:
        hist = pickle.load(pickle_filehandler)   
    print(hist_filename)
    plot_train_history(hist)
    return hist
          