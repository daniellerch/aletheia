function beta = AUMP(path, m, d, channel)
%
% AUMP LSB detector as described by L. Fillatre, "Adaptive Steganalysis of
% Least Significant Bit Replacement in Grayscale Natural Images", IEEE
% TSP, October 2011.
%
% X = image to be analyzed
% m = pixel block size
% d = q - 1 = polynomial degree for fitting (predictor)
% beta = \hat{\Lambda}^\star(X) detection statistic
%

m = 16
d = 5

X = double(imread(path));
if(size(size(X))==2)
    X=X(:,:,channel)
end

%X = double(X);
[Xpred,~,w] = Pred_aump(X,m,d);       % Polynomial prediction, w = weights
r = X - Xpred;                        % Residual
Xbar = X + 1 - 2 * mod(X,2);          % Flip all LSBs
beta = sum(sum(w.*(X-Xbar).*r));      % Detection statistic


function [Xpred,S,w] = Pred_aump(X,m,d)
%
% Pixel predictor by fitting local polynomial of degree d = q - 1 to
% m pixels, m must divide the number of pixels in the row.
% OUTPUT: predicted image Xpred, local variances S, weights w.
%
% Implemention follows the description in: L. Fillantre, "Adaptive 
% Steganalysis of Least Significant Bit Replacement in Grayscale Images",
% IEEE Trans. on Signal Processing, 2011.
%

sig_th = 1;               % Threshold for sigma for numerical stability
q = d + 1;                % q = number of parameters per block
Kn = numel(X)/m;          % Number of blocks of m pixels
Y = zeros(m,Kn);          % Y will hold block pixel values as columns
S = zeros(size(X));       % Pixel variance
Xpred = zeros(size(X));   % Predicted image

H = zeros(m,q);           % H = Vandermonde matrix for the LSQ fit
x1 = (1:m)/m;
for i = 1 : q, H(:,i) = (x1').^(i-1); end

for i = 1 : m             % Form Kn blocks of m pixels (row-wise) as
    aux = X(:,i:m:end);   % columns of Y
    Y(i,:) = aux(:);
end

p = H\Y;                  % Polynomial fit
Ypred = H*p;              % Predicted Y

for i = 1 : m             % Predicted X
    Xpred(:,i:m:end) = reshape(Ypred(i,:),size(X(:,i:m:end))); % Xpred = l_k in the paper
end

sig2 = sum((Y - Ypred).^2) / (m-q);           % sigma_k_hat in the paper (variance in kth block)
sig2 = max(sig_th^2 * ones(size(sig2)),sig2); % Assuring numerical stability
% le01 = find(sig2 < sig_th^2);
% sig2(le01) = (0.1 + sqrt(sig2(le01))).^2;   % An alternative way of "scaling" to guarantee num. stability

Sy = ones(m,1) * sig2;                        % Variance of all pixels (order as in Y)

for i = 1 : m             % Reshaping the variance Sy to size of X
    S(:,i:m:end) = reshape(Sy(i,:),size(X(:,i:m:end)));
end

s_n2 = Kn / sum(1./sig2);                     % Global variance sigma_n_bar_hat^2 in the paper
w = sqrt( s_n2 / (Kn * (m-q)) ) ./ S;         % Weights

