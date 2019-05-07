import numpy as np

def read_file(filepath):
    #Inputs:
    # filepath = path to wavefront sensor image file to be read
    #Returns:
    # images_array = numpy array of image frames, 16-bit unsigned integer pixel values
    # time_list = time stamps for each frame
    # version = file format version
    # bitdepth = camera bit depth


    if(not filepath.endswith('.dat') and not filepath.endswith('.npy')):
        raise IOError('Invalid file type')
    
    with open(filepath,"rb") as f:
        filename = filepath.split('/')[-1]
        
        if(filepath.endswith('.dat')):
            if(filename.startswith('wavefront') or filename.startswith('track')):
                dtype = np.double
            elif(filename.startswith('data')):
                dtype = np.uint16

            version = np.fromfile(f,count=1,dtype=np.double)
            bitdepth = np.fromfile(f,count=1,dtype=np.int32)
            
            time_list = []
            images_array = []

            while(True):
                frameID = np.fromfile(f,count=1,dtype=np.uint64)
                if(len(frameID)==0):
                    break
                time = np.fromfile(f,count=1,dtype=np.double)
                time_list.append(time)
                width = int(np.fromfile(f,count=1,dtype=np.uint32))
                height = int(np.fromfile(f,count=1,dtype=np.uint32))
                buffer_size = np.fromfile(f,count=1,dtype=np.uint32)
                #buffer size is size of a single frame in bytes. Frame reads out as a stream of pixel values.
                
                image_data = np.fromfile(f,count=int(width*height),dtype=dtype)
                if(len(image_data) == (width*height)):
                    image = image_data.reshape([height,width])
                    images_array.append(image)
                else:
                    print("Frame %s Skipped!!!" % (frameID))
            images_array = np.array(images_array)
            return images_array
        elif(filepath.endswith('.npy')):
            images_array = np.load(f)
            return images_array