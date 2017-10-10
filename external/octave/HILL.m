function stego = HILL(cover_path, payload)
H=0; 
x = imread(cover_path);
cost=f_cal_cost(x);
stego=f_sim_embedding(x, cost, payload, H);
end


function cost = f_cal_cost(cover)
% Copyright (c) 2014 Shenzhen University,
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from Shenzhen University. 
% -------------------------------------------------------------------------
% Author: Ming Wang
% -------------------------------------------------------------------------
% Contact: Libin@szu.edu.cn
%          2120130422@email.szu.edu.cn      
% -------------------------------------------------------------------------
% Input:  cover ... cover image%        
% Output: cost ....... costs value of all pixels
% -------------------------------------------------------------------------
% [1] A New Cost Function for Spatial Image Steganography, 
% B.Li, M.Wang,J.Huang and X.Li, to be presented at IEEE International Conference on Image Processing, 2014.
% -------------------------------------------------------------------------
%Get filter
HF1=[-1,2,-1;2,-4,2;-1,2,-1];
H2 = fspecial('average',[3 3]);
%% Get cost
cover=double(cover);
sizeCover=size(cover);
padsize=max(size(HF1));
coverPadded = padarray(cover, [padsize padsize], 'symmetric');% add padding
R1 = conv2(coverPadded,HF1, 'same');%mirror-padded convolution
W1 = conv2(abs(R1),H2,'same');
if mod(size(HF1, 1), 2) == 0, W1= circshift(W1, [1, 0]); end;
if mod(size(HF1, 2), 2) == 0, W1 = circshift(W1, [0, 1]); end;
W1 = W1(((size(W1, 1)-sizeCover(1))/2)+1:end-((size(W1, 1)-sizeCover(1))/2), ((size(W1, 2)-sizeCover(2))/2)+1:end-((size(W1, 2)-sizeCover(2))/2));
rho=1./(W1+10^(-10));
HW =  fspecial('average',[15 15]);
cost = imfilter(rho, HW ,'symmetric','same');
end
 

function stegoB = f_sim_embedding(cover, costmat, payload)
cover = double(cover);
seed = 123;
wetCost = 10^10;
% compute embedding costs \rho
rhoA = costmat;
rhoA(rhoA > wetCost) = wetCost; % threshold on the costs
rhoA(isnan(rhoA)) = wetCost; % if all xi{} are zero threshold the cost 
rhoP1 = rhoA;
rhoM1 = rhoA;
rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value
stegoB = f_EmbeddingSimulator_seed(cover, rhoP1, rhoM1, payload*numel(cover)); 
end
          
function y = f_EmbeddingSimulator_seed(x, rhoP1, rhoM1, m)
%% --------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound). 
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
    n = numel(x);   
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    rand('state', sum(100*clock));
    randChange = rand(size(x));
    y = x;
    y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
    y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;
    
    function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;
        while m3 > message_length
            l3 = l3 * 2;
            pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            m3 = ternary_entropyf(pP1, pM1);
            iterations = iterations + 1;
            if (iterations > 10)
                lambda = l3;
                return;
            end
        end        
        
        l1 = 0; 
        m1 = double(n);        
        lambda = 0;
        iterations = 0;
        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload        
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<300)
            lambda = l1+(l3-l1)/2; 
            pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            m2 = ternary_entropyf(pP1, pM1);
    		if m2 < message_length
    			l3 = lambda;
    			m3 = m2;
            else
    			l1 = lambda;
    			m1 = m2;
            end
    		iterations = iterations + 1;
            
        end
%         disp(iterations);
    end
    
    function Ht = ternary_entropyf(pP1, pM1)
        p0 = 1-pP1-pM1;
        P = [p0(:); pP1(:); pM1(:)];
        H = -((P).*log2(P));
        H((P<eps) | (P > 1-eps)) = 0;
        Ht = sum(H);
    end
end
