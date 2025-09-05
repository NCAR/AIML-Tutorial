import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def yfunction(x):
    return np.sin(4*np.pi*x)


def make_fakedata(N_SAMPLES, train_frac, val_frac, test_frac):
    # the full dataset
    xtrue = np.arange(0, 1, .001)
    ytrue = yfunction(xtrue)

    # Grab subset of sine wave:
    #   randomly select x values (without replacement)
    x = np.random.choice(a=np.arange(0, 1, 1/(N_SAMPLES)), size=N_SAMPLES, replace=False)

    # split x and y into training and testing data
    xtrain = x[:int(N_SAMPLES*train_frac)]
    ytrain = yfunction(xtrain)

    xval = x[int(N_SAMPLES*train_frac):int(N_SAMPLES*train_frac)+int(N_SAMPLES*val_frac)]
    yval = yfunction(xval)

    xtest = x[-1*int(N_SAMPLES*test_frac):]
    ytest = yfunction(xtest)

    return xtrue, ytrue, xtrain, ytrain, xval, yval, xtest, ytest

def plot_datasplit(xtrain, ytrain, xval, yval, xtest, ytest, xtrue, ytrue):
    plt.plot(xtrain, ytrain, 'darkgrey', marker='o', linewidth=0, alpha=0.1, label='training')
    plt.plot(xval, yval, 'orange', marker='o', linewidth=0, alpha=0.75, label='validation')
    plt.plot(xtest, ytest, 'teal', marker='o', linewidth=0, alpha=0.75, label='testing')
    plt.plot(xtrue, ytrue, 'w', linewidth=1, label='true')
    plt.legend()
    plt.show()
    # plt.savefig('sinewave.png',dpi=200,transparent=True)

def plot_performance(m1_name, m1, val_m1, 
                     m2_name, m2, val_m2,
                     num_epochs=100,
                     ):
    '''
    m1_name (str): first metric name
    m1 (array): array of first metric values at each epoch
    val_m1 (array): m1 for validation data
    m2_name (str): second metric name
    m2 (array): array of second metric values at each epoch
    val_m2 (array): m2 for validation data
    num_epochs (int): number of epochs
    '''

    trainColor = 'k'
    valColor = (141/255, 171/255, 127/255, 1.)
    FS = 14
    plt.figure(figsize=(12, 4))

    # plot first metric
    plt.subplot(1, 2, 1)
    plt.plot(m1, color=trainColor, label='Training', alpha=0.9, linewidth=2)
    plt.plot(val_m1, color=valColor, label='Validation', alpha=1, linewidth=2)

    plt.xlabel('EPOCH')
    plt.xticks(np.arange(0, num_epochs, num_epochs/10), labels=np.arange(0, num_epochs, num_epochs/10))
    plt.xlim(-1, num_epochs)

    plt.yticks(np.arange(0, 1.1, .1),
               labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylim(0, 1)

    plt.title(m1_name)
    plt.grid(True)
    plt.legend(frameon=True, fontsize=FS)

    # plot second metric
    plt.subplot(1, 2, 2)
    plt.plot(m2, color=trainColor, label='Training', alpha=0.9, linewidth=2)
    plt.plot(val_m2, color=valColor, label='Validation', alpha=1, linewidth=2)

    plt.xlabel('EPOCH')
    plt.xticks(np.arange(0, num_epochs, num_epochs/10), labels=np.arange(0, num_epochs, num_epochs/10))
    plt.xlim(-1, num_epochs)

    plt.yticks(np.arange(0, 1.1, .1),
               labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylim(0, 1)

    plt.title(m2_name)
    plt.grid(True)
    plt.legend(frameon=True, fontsize=FS)

    plt.show()

def plot_prediction(xtrue, ytrue, xtest, ytest, pred):

    plt.plot(xtrue,ytrue,'k',linestyle='-',linewidth = 0.2,label='true curve')
    plt.plot(xtest,ytest,'teal',marker='o',linewidth = 0,alpha=0.5,label='correct')
    plt.plot(xtest,pred,'purple',marker='o',linewidth = 0,alpha=0.5,label='model prediction')
    plt.legend()
    plt.show()



