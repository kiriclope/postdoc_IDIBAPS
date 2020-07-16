def bin_data(data, axis, bin_step, bin_size):
    # bin_step number of pts btw bins, bin_size number of size in each bin
    bin_array = [np.mean(np.take(data,np.arange(int(i*bin_step),int(i*bin_step+bin_size)), axis=2), axis=2) for i in np.arange(data.shape[2]//bin_step)-1]
    bin_array = np.array(bin_array)
    bin_array = np.rollaxis(out,0,3)
    return bin_array
