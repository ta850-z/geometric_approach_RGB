# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 19:52:21 2024

@author: YPU
"""

import numpy as np
import cv2

def inv_gamma_srgb(rgb):
    rgb2=np.zeros((rgb.shape[0],rgb.shape[1]),dtype=np.float64)
    rgb2[rgb[:,0]<=0.04045,0] = rgb[rgb[:,0]<=0.04045,0]/12.92
    rgb2[rgb[:,1]<=0.04045,1] = rgb[rgb[:,1]<=0.04045,1]/12.92
    rgb2[rgb[:,2]<=0.04045,2] = rgb[rgb[:,2]<=0.04045,2]/12.92
    rgb2[rgb[:,0]>0.04045,0] = ((rgb[rgb[:,0]>0.04045,0]+0.055)/1.055)**2.4
    rgb2[rgb[:,1]>0.04045,1] = ((rgb[rgb[:,1]>0.04045,1]+0.055)/1.055)**2.4
    rgb2[rgb[:,2]>0.04045,2] = ((rgb[rgb[:,2]>0.04045,2]+0.055)/1.055)**2.4
    return rgb2

def gamma_srgb(rgb):
    rgb2=np.zeros((rgb.shape[0],rgb.shape[1]),dtype=np.float64)
    rgb2[rgb[:,0]<=0.0031308,0] = 12.92*rgb[rgb[:,0]<=0.0031308,0]
    rgb2[rgb[:,1]<=0.0031308,1] = 12.92*rgb[rgb[:,1]<=0.0031308,1]
    rgb2[rgb[:,2]<=0.0031308,2] = 12.92*rgb[rgb[:,2]<=0.0031308,2]
    rgb2[rgb[:,0]>0.0031308,0] = 1.055*rgb[rgb[:,0]>0.0031308,0]**(1/2.4)-0.055
    rgb2[rgb[:,1]>0.0031308,1] = 1.055*rgb[rgb[:,1]>0.0031308,1]**(1/2.4)-0.055
    rgb2[rgb[:,2]>0.0031308,2] = 1.055*rgb[rgb[:,2]>0.0031308,2]**(1/2.4)-0.055
    return rgb2

def P_srgb2aalbe_S(rgb):
    Prgb_to_aalbe_S=np.array([[5.4721,-0.4229,1],[-1.1246,0.3271,1],[0.0299,1.0514,1]])
    PaalbeS=np.linalg.solve(Prgb_to_aalbe_S,rgb.T)
    return PaalbeS.T

def P_srgb2aalbe_L(rgb):
    Prgb_to_aalbe_L=np.array([[5.4721,1.6756,1],[-1.1246,0.7951,1],[0.0299,-0.1337,1]])
    PaalbeL=np.linalg.solve(Prgb_to_aalbe_L,rgb.T)
    return PaalbeL.T

def D_srgb2aalbe_S(rgb):
    Drgb_to_aalbe_S=np.array([[-4.6419,-0.4229,1],[2.2925,0.3271,1],[-0.1932,1.0514,1]])
    DaalbeS=np.linalg.solve(Drgb_to_aalbe_S,rgb.T)
    return DaalbeS.T

def D_srgb2aalbe_L(rgb):
    Drgb_to_aalbe_L=np.array([[-4.6419,1.6756,1],[2.2925,0.7951,1],[-0.1932,-0.1337,1]])
    DaalbeL=np.linalg.solve(Drgb_to_aalbe_L,rgb.T)
    return DaalbeL.T

file_name='flowers'
#file_name='train'
file_inp=file_name+'.jpg'
file_out_pe=file_name+'_out_pe.jpg'
file_out_de=file_name+'_out_de.jpg'
file_out_pe_ptype=file_name+'_out_pe_ptype.jpg'
file_out_de_dtype=file_name+'_out_de_dtype.jpg'
file_out_ptype=file_name+'_ptype.jpg'
file_out_dtype=file_name+'_dtype.jpg'

data=cv2.imread(file_inp,1)

PL=np.array([5.4721,-1.1246,0.0299])
PM=np.array([-4.6419,2.2925,-0.1932])
Xb=np.array([-0.4229,0.3271,1.0514])
Xy=np.array([1.6756,0.7951,-0.1337])
e=np.array([1,1,1])

x, y, c=data.shape[:3]
img_inp=cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
img_inp=img_inp/255
img_inp=img_inp.reshape((x*y,3),order="F")
img_inp=inv_gamma_srgb(img_inp)
PaalbeS = P_srgb2aalbe_S(img_inp)
PaalbeL = P_srgb2aalbe_L(img_inp)
Paalbe=np.array(PaalbeL)
Paalbe[Paalbe[:,1]<0,:]=PaalbeS[Paalbe[:,1]<0,:]

PY=PaalbeL[:,1].reshape(-1,1)*Xy+PaalbeL[:,2].reshape(-1,1)*e
PY[PaalbeL[:,1]<0,:]=Paalbe[PaalbeL[:,1]<0,1].reshape(-1,1)*Xb+Paalbe[PaalbeL[:,1]<0,2].reshape(-1,1)*e

s=3
rgb_Pe=s*Paalbe[:,0].reshape(-1,1)*(PL+e)+PY
rgb_Pe_Ptype=s*Paalbe[:,0].reshape(-1,1)*(+e)+PY

rgb_Pe=gamma_srgb(rgb_Pe)
rgb_Pe=255*rgb_Pe
rgb_Pe[rgb_Pe>255]=255
rgb_Pe[rgb_Pe<0]=0
rgb_Pe=rgb_Pe.astype(np.uint8)
rgb_Pe=rgb_Pe.reshape((x,y,c),order="F")
rgb_Pe=cv2.cvtColor(rgb_Pe,cv2.COLOR_RGB2BGR)

rgb_Pe_Ptype=gamma_srgb(rgb_Pe_Ptype)
rgb_Pe_Ptype=255*rgb_Pe_Ptype
rgb_Pe_Ptype[rgb_Pe_Ptype>255]=255
rgb_Pe_Ptype[rgb_Pe_Ptype<0]=0
rgb_Pe_Ptype=rgb_Pe_Ptype.astype(np.uint8)
rgb_Pe_Ptype=rgb_Pe_Ptype.reshape((x,y,c),order="F")
rgb_Pe_Ptype=cv2.cvtColor(rgb_Pe_Ptype,cv2.COLOR_RGB2BGR)

DaalbeS = D_srgb2aalbe_S(img_inp)
DaalbeL = D_srgb2aalbe_L(img_inp)
Daalbe=np.array(DaalbeL)
Daalbe[Daalbe[:,1]<0,:]=DaalbeS[Daalbe[:,1]<0,:]

DY=DaalbeL[:,1].reshape(-1,1)*Xy+DaalbeL[:,2].reshape(-1,1)*e
DY[DaalbeL[:,1]<0,:]=Daalbe[DaalbeL[:,1]<0,1].reshape(-1,1)*Xb+Daalbe[DaalbeL[:,1]<0,2].reshape(-1,1)*e

rgb_De=s*Daalbe[:,0].reshape(-1,1)*(PM-e)+DY
rgb_De_Dtype=s*Daalbe[:,0].reshape(-1,1)*(-e)+DY

rgb_De=gamma_srgb(rgb_De)
rgb_De=255*rgb_De
rgb_De[rgb_De>255]=255
rgb_De[rgb_De<0]=0
rgb_De=rgb_De.astype(np.uint8)
rgb_De=rgb_De.reshape((x,y,c),order="F")
rgb_De=cv2.cvtColor(rgb_De,cv2.COLOR_RGB2BGR)

rgb_De_Dtype=gamma_srgb(rgb_De_Dtype)
rgb_De_Dtype=255*rgb_De_Dtype
rgb_De_Dtype[rgb_De_Dtype>255]=255
rgb_De_Dtype[rgb_De_Dtype<0]=0
rgb_De_Dtype=rgb_De_Dtype.astype(np.uint8)
rgb_De_Dtype=rgb_De_Dtype.reshape((x,y,c),order="F")
rgb_De_Dtype=cv2.cvtColor(rgb_De_Dtype,cv2.COLOR_RGB2BGR)

PY=gamma_srgb(PY);
PY=255*PY
PY[PY>255]=255
PY[PY<0]=0
PY=PY.astype(np.uint8)
rgb_Ptype=PY.reshape((x,y,c),order="F")
rgb_Ptype=cv2.cvtColor(rgb_Ptype,cv2.COLOR_RGB2BGR)

DY=gamma_srgb(DY);
DY=255*DY
DY[DY>255]=255
DY[DY<0]=0
DY=DY.astype(np.uint8)
rgb_Dtype=DY.reshape((x,y,c),order="F")
rgb_Dtype=cv2.cvtColor(rgb_Dtype,cv2.COLOR_RGB2BGR)

cv2.imwrite(file_out_pe,rgb_Pe)
cv2.imwrite(file_out_de,rgb_De)
cv2.imwrite(file_out_pe_ptype,rgb_Pe_Ptype)
cv2.imwrite(file_out_de_dtype,rgb_De_Dtype)
cv2.imwrite(file_out_ptype,rgb_Ptype)
cv2.imwrite(file_out_dtype,rgb_Dtype)

cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow('original',data)

cv2.namedWindow('rgb_ptype', cv2.WINDOW_NORMAL)
cv2.imshow('rgb_ptype',rgb_Ptype)

cv2.namedWindow('rgb_dtype', cv2.WINDOW_NORMAL)
cv2.imshow('rgb_dtype',rgb_Dtype)

cv2.namedWindow('out_pe', cv2.WINDOW_NORMAL)
cv2.imshow('out_pe',rgb_Pe)

cv2.namedWindow('out_pe_ptype', cv2.WINDOW_NORMAL)
cv2.imshow('out_pe_ptype',rgb_Pe_Ptype)

cv2.namedWindow('out_de', cv2.WINDOW_NORMAL)
cv2.imshow('out_de',rgb_De)

cv2.namedWindow('out_de_dtype', cv2.WINDOW_NORMAL)
cv2.imshow('out_de_dtype',rgb_De_Dtype)

cv2.waitKey(0)
cv2.destroyAllWindows()