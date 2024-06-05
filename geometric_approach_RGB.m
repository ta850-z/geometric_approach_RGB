% geometric approach in RGB color space
clear all;
close all;

%Read image file
%rgb=imread('flowers.jpg');
rgb=imread('train.jpg');

PL=[5.4721, -1.1246, 0.0299];
PM=[-4.6419, 2.2925, -0.1932];
Xb=[-0.4229, 0.3271, 1.0514];
Xy=[1.6756, 0.7951, -0.1337];
e=[1,1,1];

figure; imshow(uint8(rgb),'Border','tight');
rgb = double(rgb);

[s1,s2,s3]=size(rgb);
rgb = rgb./255;
m = size(rgb,1);
n = size(rgb,2);
rgb = reshape(rgb(:),m*n,3);
rgb = inv_gamma_srgb(rgb);
PaalbeS = P_srgb2aalbe_S(rgb);
PaalbeL = P_srgb2aalbe_L(rgb);
Paalbe=PaalbeL;
Paalbe(Paalbe(:,2)<0,:)=PaalbeS(Paalbe(:,2)<0,:);

PY=PaalbeL(:,2).*Xy+PaalbeL(:,3).*e;
PY(PaalbeL(:,2)<0,:)=Paalbe(PaalbeL(:,2)<0,2).*Xb+Paalbe(PaalbeL(:,2)<0,3).*e;

s=3;
rgb_Pe=s*Paalbe(:,1)*(PL+e)+PY;
rgb_Pe_Ptype=s*Paalbe(:,1)*(+e)+PY;
rgb_Pe=gamma_srgb(rgb_Pe);
rgb_Pe=reshape(rgb_Pe,m,n,3);
rgb_Pe_Ptype=gamma_srgb(rgb_Pe_Ptype);
rgb_Pe_Ptype=reshape(rgb_Pe_Ptype,m,n,3);

DaalbeS = D_srgb2aalbe_S(rgb);
DaalbeL = D_srgb2aalbe_L(rgb);
Daalbe=DaalbeL;
Daalbe(Daalbe(:,2)<0,:)=DaalbeS(Daalbe(:,2)<0,:);

DY=DaalbeL(:,2).*Xy+DaalbeL(:,3).*e;
DY(DaalbeL(:,2)<0,:)=Daalbe(DaalbeL(:,2)<0,2).*Xb+Daalbe(DaalbeL(:,2)<0,3).*e;

rgb_De=s*Daalbe(:,1)*(PM-e)+DY;
rgb_De_Dtype=s*Daalbe(:,1)*(-e)+DY;
rgb_De=gamma_srgb(rgb_De);
rgb_De=reshape(rgb_De,m,n,3);
rgb_De_Dtype=gamma_srgb(rgb_De_Dtype);
rgb_De_Dtype=reshape(rgb_De_Dtype,m,n,3);

PY=gamma_srgb(PY);
rgb_Ptype=reshape(PY,m,n,3);
DY=gamma_srgb(DY);
rgb_Dtype=reshape(DY,m,n,3);

figure; imshow(rgb_Ptype,'Border','tight');
figure; imshow(rgb_Dtype,'Border','tight');
figure; imshow(rgb_Pe,'Border','tight');
figure; imshow(rgb_De,'Border','tight');
figure; imshow(rgb_Pe_Ptype,'Border','tight');
figure; imshow(rgb_De_Dtype,'Border','tight');

function rgb_l = inv_gamma_srgb(rgb_n)
rgb=rgb_n;
rgb(rgb_n(:,1)<=0.04045,1)=rgb_n(rgb_n(:,1)<=0.04045,1)/12.92;
rgb(rgb_n(:,2)<=0.04045,2)=rgb_n(rgb_n(:,2)<=0.04045,2)/12.92;
rgb(rgb_n(:,3)<=0.04045,3)=rgb_n(rgb_n(:,3)<=0.04045,3)/12.92;
rgb(rgb_n(:,1)>0.04045,1)=((rgb_n(rgb_n(:,1)>0.04045,1)+0.055)/1.055).^2.4;
rgb(rgb_n(:,2)>0.04045,2)=((rgb_n(rgb_n(:,2)>0.04045,2)+0.055)/1.055).^2.4;
rgb(rgb_n(:,3)>0.04045,3)=((rgb_n(rgb_n(:,3)>0.04045,3)+0.055)/1.055).^2.4;
rgb_l=rgb;
end

function rgb_n = gamma_srgb(rgb_l)
rgb=rgb_l;
rgb(rgb_l(:,1)<=0.0031308,1)=12.92*rgb_l(rgb_l(:,1)<=0.0031308,1);
rgb(rgb_l(:,2)<=0.0031308,2)=12.92*rgb_l(rgb_l(:,2)<=0.0031308,2);
rgb(rgb_l(:,3)<=0.0031308,3)=12.92*rgb_l(rgb_l(:,3)<=0.0031308,3);
rgb(rgb_l(:,1)>0.0031308,1)=1.055*(rgb_l(rgb_l(:,1)>0.0031308,1)).^(1.0/2.4)-0.055;
rgb(rgb_l(:,2)>0.0031308,2)=1.055*(rgb_l(rgb_l(:,2)>0.0031308,2)).^(1.0/2.4)-0.055;
rgb(rgb_l(:,3)>0.0031308,3)=1.055*(rgb_l(rgb_l(:,3)>0.0031308,3)).^(1.0/2.4)-0.055;
rgb_n=rgb;
end

function PaalbeS = P_srgb2aalbe_S(rgb)
Prgb_to_aalbe_S=[5.4721,-0.4229,1;-1.1246,0.3271,1;0.0299,1.0514,1];
PaalbeS=(Prgb_to_aalbe_S\rgb')';
end

function PaalbeL = P_srgb2aalbe_L(rgb)
Prgb_to_aalbe_L=[5.4721,1.6756,1;-1.1246,0.7951,1;0.0299,-0.1337,1];
PaalbeL=(Prgb_to_aalbe_L\rgb')';
end

function DaalbeS = D_srgb2aalbe_S(rgb)
Drgb_to_aalbe_S=[-4.6419,-0.4229,1;2.2925,0.3271,1;-0.1932,1.0514,1];
DaalbeS=(Drgb_to_aalbe_S\rgb')';
end

function DaalbeL = D_srgb2aalbe_L(rgb)
Drgb_to_aalbe_L=[-4.6419,1.6756,1;2.2925,0.7951,1;-0.1932,-0.1337,1];
DaalbeL=(Drgb_to_aalbe_L\rgb')';
end