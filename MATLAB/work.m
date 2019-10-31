clc

img_in = imread('Test1.pdf', 0);
class(img_in), size(img_in)
subplot(2,1,1); 
imshow(img_in)
subplot(2,1,3)
imhist(img_in)