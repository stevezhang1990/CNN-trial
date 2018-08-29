import tensorflow as tf
import numpy as np
import math
from scipy import interpolate
import netCDF4 as nc
import random

from glob import glob
import param

surf = 'surftp_uv_rough_Tbot_TMQ.nc'
trop = 'temp_UV_geopot_500_200.nc'
flux = 'ver_geop_moist_flux.nc'

def get_nc_db(neg_dir):
    neg_file_list = []
    timeind_list = []
    subdirs = glob('./neg_train/*/*/')

    for s in subdirs:
        fh = nc.Dataset(s + 'surftp_uv_rough_Tbot_TMQ.nc', 'r')
        num_of_times = fh.variables['time'].shape[0]
        fh.close()

        neg_file_list += [s] * param.neg_per_month

        timeind_list += random.sample(range(num_of_times), param.neg_per_month)


    return neg_file_list, timeind_list


def open_ng_nc(filename):
    tmp = np.zeros((321, 481, param.input_channel))

    fh = nc.Dataset(filename + 'surftp_uv_rough_Tbot_TMQ.nc', 'r')
    tmp[:,:, 0] = fh.variables['t2m'][timeind, :, :]
    tmp[:,:, 1] = fh.variables['u10'][timeind, :, :]
    tmp[:,:, 2] = fh.variables['v10'][timeind, :, :]

    fh.close()

    return tmp    


def img_flip(img):
    return np.fliplr(img)


def open_nc(filename, timeind):
    tmp = np.zeros((321, 481, param.input_channel))

    fh = nc.Dataset(filename + 'surftp_uv_rough_Tbot_TMQ.nc', 'r')
    tmp[:,:, 0] = fh.variables['t2m'][timeind, :, :]
    tmp[:,:, 1] = fh.variables['u10'][timeind, :, :]
    tmp[:,:, 2] = fh.variables['v10'][timeind, :, :]

    fh.close()

    return tmp


def _check_factor(factor):
    if factor <= 1:
        raise ValueError('scale factor must be greater than 1')

def img_enlarge(img, dimx, dimy):

    x = np.arange(0, img.shape[0], 1)
    y = np.arange(0, img.shape[1], 1)
    new_x = np.linspace(0, img.shape[0]-1, dimx)
    new_y = np.linspace(0, img.shape[1]-1, dimy)

    tmp = np.zeros((dimx, dimy, param.input_channel))

    for i in range(param.input_channel):
        
        f = interpolate.interp2d(x, y, img[:,:, i], kind='linear')
        new_img = f(new_x, new_y)
        tmp[:,:, i] = new_img

    return tmp



def img_resize(img, dimx, dimy):

    if img.shape[0] < dimx or img.shape[1] < dimy:
        return img_enlarge(img, dimx, dimy)

    num_win_x = img.shape[0]/dimx
    num_win_y = img.shape[1]/dimy
    residual_x = img.shape[0]%dimx
    residual_y = img.shape[1]%dimy

    tmp = np.zeros((dimx, dimy, param.input_channel))

    x_start = 0
    x_end = num_win_x
    for i in range(dimx):
        if i < residual_x:
            x_end += 1

        y_start = 0
        y_end = num_win_y
        for j in range(dimy):
            if j < residual_y:
                y_end += 1
            
            tmp[i,j,:] = np.mean(img[x_start:x_end, y_start:y_end, :], axis=(0, 1))

            y_start = y_end
            y_end = y_start + num_win_y

        x_start = x_end
        x_end = x_start + num_win_x

    return tmp


def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                     mode='reflect', cval=0, multichannel=None):

    layer = 0
    current_shape = image.shape

    prev_layer_image = image
    yield image

    while layer != max_layer:
        layer += 1

        layer_image = img_resize(img, dimx, dimy)

        prev_shape = np.asarray(current_shape)
        prev_layer_image = layer_image
        current_shape = np.asarray(layer_image.shape)

        # no change to previous pyramid layer
        if np.all(current_shape == prev_shape):
            break

        yield layer_image




def img_crop(img, xmin, ymin, xmax, ymax):

    return img[xmin:xmax+1, ymin:ymax+1, :]


def img2array(img,dim):
     
    if dim == param.img_size_12:    
        if img.size[0] != param.img_size_12 or img.size[1] != param.img_size_12:
            # img = img.resize((param.img_size_12,param.img_size_12))
            img = img_resize(img, param.img_size_12,param.img_size_12)
        img = np.asarray(img).astype(np.float32)/img.max()
    elif dim == param.img_size_24:
        if img.size[0] != param.img_size_24 or img.size[1] != param.img_size_24:
            # img = img.resize((param.img_size_24,param.img_size_24))
            img = img_resize(img, param.img_size_24, param.img_size_24)
        img = np.asarray(img).astype(np.float32)/img.max()
    elif dim == param.img_size_48:
        if img.size[0] != param.img_size_48 or img.size[1] != param.img_size_48:
            # img = img.resize((param.img_size_48,param.img_size_48))
            img = img_resize(img, param.img_size_48, param.img_size_48)
        img = np.asarray(img).astype(np.float32)/img.max()
    return img

def calib_box(result_box,result,img):
    

    for id_,cid in enumerate(np.argmax(result,axis=1).tolist()):
        s = cid / (len(param.cali_off_x) * len(param.cali_off_y))
        x = cid % (len(param.cali_off_x) * len(param.cali_off_y)) / len(param.cali_off_y)
        y = cid % (len(param.cali_off_x) * len(param.cali_off_y)) % len(param.cali_off_y) 
                
        s = param.cali_scale[s]
        x = param.cali_off_x[x]
        y = param.cali_off_y[y]
    
        
        new_ltx = result_box[id_][0] + x*(result_box[id_][2]-result_box[id_][0])
        new_lty = result_box[id_][1] + y*(result_box[id_][3]-result_box[id_][1])
        new_rbx = new_ltx + s*(result_box[id_][2]-result_box[id_][0])
        new_rby = new_lty + s*(result_box[id_][3]-result_box[id_][1])
        
        result_box[id_][0] = int(max(new_ltx,0))
        result_box[id_][1] = int(max(new_lty,0))
        result_box[id_][2] = int(min(new_rbx,img.size[0]-1))
        result_box[id_][3] = int(min(new_rby,img.size[1]-1))
        result_box[id_][5] = img.crop((result_box[id_][0],result_box[id_][1],result_box[id_][2],result_box[id_][3]))

    return result_box 

def NMS(box):
    
    if len(box) == 0:
        return []
    
    #xmin, ymin, xmax, ymax, score, cropped_img, scale
    box.sort(key=lambda x :x[4])
    box.reverse()

    pick = []
    x_min = np.array([box[i][0] for i in range(len(box))],np.float32)
    y_min = np.array([box[i][1] for i in range(len(box))],np.float32)
    x_max = np.array([box[i][2] for i in range(len(box))],np.float32)
    y_max = np.array([box[i][3] for i in range(len(box))],np.float32)

    area = (x_max-x_min)*(y_max-y_min)
    idxs = np.array(range(len(box)))

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x_min[i],x_min[idxs[1:]])
        yy1 = np.maximum(y_min[i],y_min[idxs[1:]])
        xx2 = np.minimum(x_max[i],x_max[idxs[1:]])
        yy2 = np.minimum(y_max[i],y_max[idxs[1:]])

        w = np.maximum(xx2-xx1,0)
        h = np.maximum(yy2-yy1,0)

        overlap = (w*h)/(area[idxs[1:]] + area[i] - w*h)

        idxs = np.delete(idxs, np.concatenate(([0],np.where(((overlap >= 0.5) & (overlap <= 1)))[0]+1)))
    
    return [box[i] for i in pick]

def sliding_window(img, thr, net, input_12_node):

    # pyramid = tuple(pyramid_gaussian(img, downscale = param.downscale))
    pyramid = tuple(pyramid_gaussian(img, downscale = param.downscale))

    detected_list = [0 for _ in xrange(len(pyramid))]
    for scale in xrange(param.pyramid_num):
        
        X = pyramid[scale]

        resized = Image.fromarray(np.uint8(X*255)).resize(
            (   
                int(np.shape(X)[1] * float(param.img_size_12)/float(param.face_minimum) ), \
                int(np.shape(X)[0] * float(param.img_size_12)/float(param.face_minimum) )
            )
            )
        
        X = np.asarray(resized).astype(np.float32)/255

        img_row = np.shape(X)[0]
        img_col = np.shape(X)[1]

        X = np.reshape(X,(1,img_row,img_col,param.input_channel))
        
        if img_row < param.img_size_12 or img_col < param.img_size_12:
            break
        
        #predict and get rid of boxes from padding
        win_num_row = math.floor((img_row-param.img_size_12)/param.window_stride)+1
        win_num_col = math.floor((img_col-param.img_size_12)/param.window_stride)+1

        result = net.prediction.eval(feed_dict={input_12_node:X})
        result_row = np.shape(result)[1]
        result_col = np.shape(result)[2]

        result = result[:,\
                int(math.floor((result_row-win_num_row)/2)):int(result_row-math.ceil((result_row-win_num_row)/2)),\
                int(math.floor((result_col-win_num_col)/2)):int(result_col-math.ceil((result_col-win_num_col)/2)),\
                :]

        feature_col = np.shape(result)[2]

        #feature_col: # of predicted window num in width dim
        #win_num_col: # of box(gt)
        assert(feature_col == win_num_col)

        result = np.reshape(result,(-1,1))
        result_id = np.where(result > thr)[0]
        
        #xmin, ymin, xmax, ymax, score
        detected_list_scale = np.zeros( (len(result_id), 5) ,np.float32)
        detected_list_scale[:,0] = (result_id%feature_col) * param.window_stride
        detected_list_scale[:,1] = np.floor(result_id/feature_col) * param.window_stride
        detected_list_scale[:,2] = np.minimum(detected_list_scale[:, 0] + param.img_size_12 - 1, img_col-1)
        detected_list_scale[:,3] = np.minimum(detected_list_scale[:, 1] + param.img_size_12 - 1, img_row-1)

        # project back to original image...
        detected_list_scale[:,0] = detected_list_scale[:,0] / (img_col-1) * (img.size[0]-1)
        detected_list_scale[:,1] = detected_list_scale[:,1] / (img_row-1) * (img.size[1]-1)
        detected_list_scale[:,2] = detected_list_scale[:,2] / (img_col-1) * (img.size[0]-1)
        detected_list_scale[:,3] = detected_list_scale[:,3] / (img_row-1) * (img.size[1]-1)
        detected_list_scale[:,4] = result[result_id,0] # score...

        detected_list_scale = detected_list_scale.tolist()
       
        #xmin, ymin, xmax, ymax, score, cropped_img, scale
        detected_list_scale = [elem + [img.crop((int(elem[0]),int(elem[1]),int(elem[2]),int(elem[3]))), scale] for id_,elem in enumerate(detected_list_scale)]
        
        if len(detected_list_scale) > 0:
            detected_list[scale] = detected_list_scale 
            
    detected_list = [elem for elem in detected_list if type(elem) != int] # remove empty 
    result_box = [detected_list[i][j] for i in xrange(len(detected_list)) for j in xrange(len(detected_list[i]))]
    
    return result_box
