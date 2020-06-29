def bin_data(X_window, bin_size, bin_overlap):
    for i in range(0,int(X_window.shape[2]/bin_size)):
        X_bin = np.concatenate[X_bin, np.mean(X_window[:,:,i:i+bin_size],axis=2)]
    return X_bin

def binArray(data, axis, binstep, binsize):
    out = [np.mean(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)), axis=2), axis=2) for i in np.arange(data.shape[2]//binstep)-1]
    out = np.array(out)
    out = np.rollaxis(out,0,3)
    return out
