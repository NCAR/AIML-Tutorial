import wget
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def get_data(download=True):
    if download:
        url = 'https://eabarnes-data.atmos.colostate.edu/tutorials/ERSSTv5_deseasoneddetrended_5monthrunningmean_1950-2021.nc'
        wget.download(url)
    ddir = '/glade/work/kjmayer/ML_demos/EdEc_ML/utils/'
    filename = 'ERSSTv5_deseasoneddetrended_5monthrunningmean_1950-2021.nc'
    sstds = xr.open_dataset(ddir+filename,decode_times=False)
    sst = sstds.sst
    return sst

def get_nino34(sst):
    ninolat1 = -5
    ninolat2 = 5
    ninolon1 = 190
    ninolon2 = 240
    
    sstnino = np.asarray(sst.sel(lat=slice(ninolat1,ninolat2),lon=slice(ninolon1,ninolon2)))
    nino34 = np.nanmean(sstnino,axis=(1,2))
    return nino34

def get_nino34events(sst, nino34, enso_magnitude, vectorize=True):
    x = sst[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude)),:,:] # grab sst samples where nino occurs
    if vectorize == True:
        x = np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2])) # reshape latxlon to vectors
    
    y = nino34[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))]
    return x,y


def split_data(num_samples,train_val_test,x,y):
    trainind = range(int(train_val_test[0]*num_samples))
    valind = range(int(train_val_test[0]*num_samples),int(train_val_test[1]*num_samples))
    testind = range(int(train_val_test[1]*num_samples),int(train_val_test[2]*num_samples))
    
    # divide into training/validation/testing
    xtrain = x[trainind,:]
    xval = x[valind,:]
    xtest = x[testind,:]
    
    ytrain  = y[trainind]
    yval = y[valind]
    ytest = y[testind]

    return xtrain, xval, xtest, ytrain, yval, ytest, trainind, valind, testind

def standardize_data(xtrain, xval, xtest):
    xstd = np.std(xtrain,axis=0) 
    
    xtrain = np.divide(xtrain,xstd)
    xtrain[np.isnan(xtrain)] = 0 # set all nans to zeros (they are learned to be ignored)
    
    xval = np.divide(xval,xstd)
    xval[np.isnan(xval)] = 0
    
    xtest = np.divide(xtest,xstd)
    xtest[np.isnan(xtest)] = 0

    return xtrain, xval, xtest

def make_cat(y, enso_magnitude):
    y[y>enso_magnitude] = 1 
    y[y<(-1*enso_magnitude)] = 0
    return y



def plot_input(sst, lon, lat):
    projection = ccrs.PlateCarree(central_longitude=180)
    transform = ccrs.PlateCarree()
    # plt.figure(figsize=(8,3))
    plt.figure(figsize=(8,4))
    ax1=plt.subplot(2,1,1,projection=projection)
    ax1.coastlines()
    ax1.set_frame_on(False) 
    ax1.pcolormesh(lon,lat,sst,vmin=-2,vmax=2,cmap='RdBu_r',transform=transform)
    ax1.coastlines(color='gray')
    plt.title('Example Input')
    plt.show()

def plot_inputvector(x):
    # plt.figure(figsize=(8,3))
    fig, ax = plt.subplots(figsize=(8,3))
    # Remove top and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(range(np.shape(x)[0]),x,color='xkcd:slate blue',linewidth=0,marker='.')
    plt.xlim(0,np.shape(x)[0])
    plt.ylim(-3.5,3.5)
    plt.ylabel("SST Anomaly")
    plt.show()

def plot_nino34(nino34):
    # plt.figure(figsize=(8,3))
    fig, ax = plt.subplots(figsize=(8,3))
    # Remove top and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(1950+(5/12),2022,1/12),nino34,color='xkcd:slate blue')
    plt.hlines(0.5,1950,2022,linestyle='dashed',color='grey')
    plt.hlines(-0.5,1950,2022,linestyle='dashed',color='grey')
    plt.xlim(1950,2022)
    plt.ylim(-2.5,2.5)
    plt.ylabel("Nino 3.4 Index")
    plt.show()

def plot_datasplit(nino34,enso_magnitude,trainind,valind,testind):
    timevec = np.arange(1950+(5/12),2022,1/12)
    # plt.figure(figsize=(8,3))
    fig, ax = plt.subplots(figsize=(8,3))
    # Remove top and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(timevec[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][trainind],
                nino34[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][trainind],
                color='xkcd:slate blue', s=8, label = "train")
    plt.scatter(timevec[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][valind],
                nino34[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][valind],
                color='indianred', s=8, label = "validation")
    plt.scatter(timevec[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][testind],
                nino34[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][testind],
                color='coral', s=8, label = "test")
    plt.hlines(enso_magnitude,1950,2022,linestyle='dashed',color='grey')
    plt.hlines(-1*enso_magnitude,1950,2022,linestyle='dashed',color='grey')
    plt.xlim(1950,2022)
    plt.ylim(-2.5,2.5)
    plt.ylabel("Nino 3.4 Index")
    plt.legend(bbox_to_anchor=(1, 0.4, .1, 0.2))
    plt.show()

def plot_datasplitcat(ytrain,yval,ytest,nino34,enso_magnitude,trainind,valind,testind):
    timevec = np.arange(1950+(5/12),2022,1/12)
    # plt.figure(figsize=(8,3))
    fig, ax = plt.subplots(figsize=(8,3))
    # Remove top and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(timevec[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][trainind],
                ytrain,color='xkcd:slate blue', s=8, label = "train")
    plt.scatter(timevec[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][valind],
                yval,color='indianred', s=8, label = "validation")
    plt.scatter(timevec[(nino34>enso_magnitude) | (nino34<(-1*enso_magnitude))][testind],
                ytest,color='coral', s=8, label = "test")
    plt.xlim(1950,2022)
    plt.ylim(-0.5,1.5)
    plt.ylabel("Nino 3.4 Categorical")
    plt.legend(bbox_to_anchor=(1, 0.4, .1, 0.2))
    plt.show()
    
# def plot_performance(m1_name, m1, val_m1, 
#                      m2_name, m2, val_m2,
#                      num_epochs=100,
#                      ):
#     '''
#     m1_name (str): first metric name
#     m1 (array): array of first metric values at each epoch
#     val_m1 (array): m1 for validation data
#     m2_name (str): second metric name
#     m2 (array): array of second metric values at each epoch
#     val_m2 (array): m2 for validation data
#     num_epochs (int): number of epochs
#     '''

#     trainColor = 'k'
#     valColor = (141/255, 171/255, 127/255, 1.)
#     FS = 14
#     plt.figure(figsize=(12, 4))

#     # plot first metric
#     plt.subplot(1, 2, 1)
#     plt.plot(m1, color=trainColor, label='Training', alpha=0.9, linewidth=2)
#     plt.plot(val_m1, color=valColor, label='Validation', alpha=1, linewidth=2)

#     plt.xlabel('EPOCH')
#     plt.xticks(np.arange(0, num_epochs, num_epochs/10), labels=np.arange(0, num_epochs, num_epochs/10))
#     plt.xlim(-1, num_epochs)

#     plt.yticks(np.arange(0, 1.1, .1),
#                labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     plt.ylim(0, 1)

#     plt.title(m1_name)
#     plt.grid(True)
#     plt.legend(frameon=True, fontsize=FS)

#     # plot second metric
#     plt.subplot(1, 2, 2)
#     plt.plot(m2, color=trainColor, label='Training', alpha=0.9, linewidth=2)
#     plt.plot(val_m2, color=valColor, label='Validation', alpha=1, linewidth=2)

#     plt.xlabel('EPOCH')
#     plt.xticks(np.arange(0, num_epochs, num_epochs/10), labels=np.arange(0, num_epochs, num_epochs/10))
#     plt.xlim(-1, num_epochs)

#     plt.yticks(np.arange(0, 1.1, .1),
#                labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     plt.ylim(0, 1)

#     plt.title(m2_name)
#     plt.grid(True)
#     plt.legend(frameon=True, fontsize=FS)

#     plt.show()

# def plot_prediction(xtrue, ytrue, xtest, ytest, pred):

#     plt.plot(xtrue,ytrue,'k',linestyle='-',linewidth = 0.2,label='true curve')
#     plt.plot(xtest,ytest,'teal',marker='o',linewidth = 0,alpha=0.5,label='correct')
#     plt.plot(xtest,pred,'purple',marker='o',linewidth = 0,alpha=0.5,label='model prediction')
#     plt.legend()
#     plt.show()



