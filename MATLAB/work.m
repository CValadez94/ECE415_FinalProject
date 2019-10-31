clc

img_in = imread('../Test1.png');
img_in = rgb2gray(img_in);
[H, W] = size(img_in);
subplot(2,2,1); 
imshow(img_in)
subplot(2,2,3)
imhist(img_in)

t = 127;                % Threshold picked from histogram
img = zeros(H,W);      % Store in a new matrix
for i=1:H
    for j=1:W
        if (img_in(i,j) < t)
            img(i,j) = 0;
        else
            img(i,j) = 1;
        end
    end
end

subplot(2,2,[2 4]);
imshow(img)
imwrite(img, 'Test1_binary.png');