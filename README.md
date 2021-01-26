# Non-Local-Means
Filters gaussian noise in image, with non-local-means algorithm

~ Non Local Means ~

The central idea of the algorithm is that, given a noisy image "Im", we can assume that the non-noised image "If" is basically a weighted average of all the pixels
from the initial image "Im". Specifically my function implements this theory using the following equation: If(x) = SUM{ w(x)(y) * Im(y) } where w(x)(y) denotes a 
weight that represents the similarity between two neighborhouds of NxN pixels each (patches of the image). We have w(x)(y) = 1/Z(x) * exp{ -|| P(Nx)-P(Ny) ||^2 / sigma^2 }
where P(Nx) is a neighborhoud with central pixel "x" and Z is the normalization factor. So the algorithm calculates the Euclidean distance of all patches from all the others.
The distance essentially represents the level of similarity between the two neighborhouds (small distance means similar tone of black thus similar color). For more informations
about the algorithm i highly suggest to visit https://www.csd.uoc.gr/~hy371/bibliography/Non-localMeans.pdf which analyzes in detail everything that my code is base on.

~ Sequential code ~

After it reads the image from file "house.txt", it uses the "addNoise" function to add Gaussian noise to all the image pixels. The "Non-Local-Means" algorithm
filters the added noise and returns the image with much higher resolution. You can experiment with different values of SIGMA (variance), and see which delivers the best 
results. It is highly advisable to use MATLAB to print the image in order to observe the results of the algorithm. I have included functions that write to txt file 
so you can simply follow these steps:

1. Move the txt file of the denoised image to your matlab workspace-directory

2. Type:
         
         a) Ifilt = dlmread('denoised.txt');

         b) save('denoised.mat', 'Ifilt');
         
         c) load('denoised.mat');
         
         d) imshow(Ifilt, []);
         
3. The denoised image should appear on your screen.

