import numpy as np
import ctypes
import time
#from matplotlib import pyplot as plt
import pfb  #Richard Shaw's PFB library, used for taking the forward transform

mylib=ctypes.cdll.LoadLibrary("libcuda_ipfb.so")

#cufft_c2r=mylib.cufft_c2r_host
#cufft_c2r.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)

#cufft_r2c=mylib.cufft_r2c_host
#cufft_r2c.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int)

cuipfb=mylib.ipfb
cuipfb.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float)


#python version of the IPFB, can be used for comparison
def myipfb(inft,win,ntap):
    in_unft=np.fft.irfft(inft,axis=1)
    n=in_unft.shape[1]
    winmat=np.zeros(in_unft.shape)

    winmat[:ntap,:]=np.reshape(win,[ntap,n])
    winft=np.fft.rfft(winmat,axis=0)
    myft=np.fft.rfft(in_unft,axis=0)
    out=np.fft.irfft(myft/np.conj(winft),axis=0)
    return out,winft
#plt.ion()
ntap=4
nchan=1025
nslice=1024*32
nn=2*(nchan-1)
x=np.random.randn(nslice+ntap-1,nn)
xx=np.ravel(x)

#CUDA expects the window function as an input, so calculate it here.
win=pfb.sinc_hamming(ntap,nn)
win=np.asarray(win,dtype='float32')

#get the PFB of 
dxpfb=pfb.pfb(xx,nchan,ntap,pfb.sinc_hamming)
#cuda version is fp32, so cast double to single
xpfb=np.asarray(dxpfb,dtype='complex64')

out=np.empty([nslice,nn],dtype='float32')
filt=0  #you might want this to be ~0.1 if you're using quantized data
        #set to 0 for no Wiener filtering at all.  The proper value
        #depends on how many bits you use in quantization

#do a few loops for timing since the first iteration is slow
for i in range(10):
    t1=time.time()
    cuipfb(out.ctypes.data,xpfb.ctypes.data,win.ctypes.data,nchan,nslice,ntap,filt)
    t2=time.time()
    print('took ',t2-t1,' seconds to do ipfb, rate of ',out.size/(t2-t1)/1e6,' Msamp/s')

to_cut=100
imax=out.shape[0]-to_cut
print("RMS error in reconstruction is ",np.std(out[to_cut:imax,:]-x[to_cut:imax,:]))

