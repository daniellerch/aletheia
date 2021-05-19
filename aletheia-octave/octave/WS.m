function beta_hat = WS(path, channel)
%
% Weighted-Sstego LSB change-rate estimator with moderated weights as 
% described in:
% Andrew D. Ker and Rainer Böhme: "Revisiting Weighted Stego-Image 
% Steganalysis." In: Security, Forensics, Steganography, and Watermarking 
% of Multimedia Contents X, Proc. SPIE Electronic Imaging, vol. 6819, 
% San Jose, CA, pp. 0501-0517, 2008.
%
% If bias == 'yes', bias correction is applied (should be done with images
% with correlated parities (e.g., saturated).
% S must be an array of grayscale pixel values. For color images, run sp.m
% for all three channels separately.
%
% 2011 Copyright by Jessica Fridrich, fridrich@binghamton.edu,
% http:\\ws.binghamton.edu\fridrich
%

bias = 'no';
S = imread(path);
if(size(size(S))!=2)
    S=S(:,:,channel);
end

[M, N] = size(S);

I = 2:M-1;
J = 2:N-1;

S = double(S);
Sbar = S + 1 - 2 * mod(S, 2);         % stego image with flipped LSBs

% Computing local weights
varS = localvar(S,3);                 % Local variance
w = 1./(5 + varS(2:end-1,2:end-1));   % Moderated weights
w = w / sum(w(:));                    % Normalization to 1

% Cover image estimate using the KB kernel
X_hat = 1/4 * ( -S(I-1,J-1)-S(I+1,J-1)-S(I+1,J+1)-S(I-1,J+1) + 2 * (S(I,J-1)+S(I,J+1)+S(I-1,J)+S(I+1,J)) );

% Estimated change rate
beta_hat = sum(sum(w.*(S(I,J) - X_hat).*(S(I,J) - Sbar(I,J))));

% Bias correction
if strcmp(bias,'yes')
    D = Sbar - S;
    FD = 1/4 * ( -D(I-1,J-1)-D(I+1,J-1)-D(I+1,J+1)-D(I-1,J+1) + 2 * (D(I,J-1)+D(I,J+1)+D(I-1,J)+D(I+1,J)) );
    b = beta_hat * sum(sum(w.*FD.*(S(I,J) - Sbar(I,J))));
    beta_hat = beta_hat + b;
end


function v = localvar(X,K)

% This function calculates the local standard deviation for matrix X.
% The standard deviation (std) is evaluated for a square region KxK
% pixels surrounding each pixel. At the boundary, the matrix is NOT
% padded. Instead, the std is calculated from available pixels only.
% If K is not an odd integer, it is floored to the closest odd integer.
%
% Input:  X   MxN matrix
%         K   size of the square region for calculating std
% Output: s   local std calculated for KxK regions
% Typical use: s=localstd(X,3);

[M N] = size(X);

if mod(K,2)~=1			% K must be an odd number
  K = K + 1;
end

kern = ones(K,K);			% Kernel for calculating std
x1 = conv2(X, kern, 'same');	% Local sums
x2 = conv2(X.*X,kern,'same');	% Local quadratic sums
R  = conv2(ones(M,N),kern,'same');	% Number of matrix elements in each square region
v  = x2./R-(x1./R).^2;	        % Local variance


