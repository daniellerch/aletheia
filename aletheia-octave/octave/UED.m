function UED(cover, payload, stego)
% ternary

% Script modified by Rémi Cogranne, based upon the ones provided by Vojtech Holub (in 2014)
% use:
% toto = jpeg_read('./test.jpg');
% payload = 0.4;
% S_STRUCT = UED_ternary(toto, payload)
%
try
    C_STRUCT = jpeg_read(cover);
catch
    error('ERROR (problem with the cover image)');
end



wetConst = 10^13;

C_COEFFS = C_STRUCT.coef_arrays{1};

nzAC = nnz(C_COEFFS)-nnz(C_COEFFS(1:8:end,1:8:end));

%% Get costs
alphaIA = 1.3;
alphaIR = 1;

rhoP1 = zeros(size(C_COEFFS), 'double');
absCij = abs(C_COEFFS);

%% Intra blocks
% i+1, j
rhoP1(1:end-1, :) = rhoP1(1:end-1, :) + (absCij(1:end-1, :) + absCij(2:end, :) + alphaIA).^(-1);
% i-1, j
rhoP1(2:end, :) = rhoP1(2:end, :) + (absCij(2:end, :) + absCij(1:end-1, :) + alphaIA).^(-1);
% i, j+1
rhoP1(:, 1:end-1) = rhoP1(:, 1:end-1) + (absCij(:, 1:end-1) + absCij(:, 2:end) + alphaIA).^(-1);
% i, j-1
rhoP1(:, 2:end) = rhoP1(:, 2:end) + (absCij(:, 2:end) + absCij(:, 1:end-1) + alphaIA).^(-1);

%% Inter blocks
% i+8, j
rhoP1(1:end-8, :) = rhoP1(1:end-8, :) + (absCij(1:end-8, :) + absCij(9:end, :) + alphaIR).^(-1);
% i-8, j
rhoP1(9:end, :) = rhoP1(9:end, :) + (absCij(9:end, :) + absCij(1:end-8, :) + alphaIR).^(-1);
% i, j+8
rhoP1(:, 1:end-8) = rhoP1(:, 1:end-8) + (absCij(:, 1:end-8) + absCij(:, 9:end) + alphaIR).^(-1);
% i, j-8
rhoP1(:, 9:end) = rhoP1(:, 9:end) + (absCij(:, 9:end) + absCij(:, 1:end-8) + alphaIR).^(-1);

%% Omit embedding into DC and zero coefficients
rhoP1(1:8:end,1:8:end) = wetConst;
rhoP1(C_COEFFS==0) = wetConst;

rhoM1 = rhoP1;
rhoP1(C_COEFFS==-1) = wetConst;
rhoM1(C_COEFFS==1) = wetConst;
rhoP1(C_COEFFS>=1023) = wetConst;
rhoM1(C_COEFFS<=-1023) = wetConst;

%% Embedding simulation
[S_COEFFS] = EmbeddingSimulator(C_COEFFS, rhoP1, rhoM1, round(payload * nzAC) );

S_STRUCT = C_STRUCT;
S_STRUCT.coef_arrays{1} = S_COEFFS;
S_STRUCT.optimize_coding = 1;


try
    jpeg_write(S_STRUCT, stego);
catch
    error('ERROR (problem with saving the stego image)')
end



end


function [y] = EmbeddingSimulator(x, rhoP1, rhoM1, m)

    x = double(x);
    n = numel(x);
    
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    
    %RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
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
        
        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload        
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
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
    end
    
    function Ht = ternary_entropyf(pP1, pM1)
        pP1 = pP1(:);
        pM1 = pM1(:);
        Ht = -(pP1.*log2(pP1))-(pM1.*log2(pM1))-((1-pP1-pM1).*log2(1-pP1-pM1));
        Ht(isnan(Ht)) = 0;
        Ht = sum(Ht);
    end

end

