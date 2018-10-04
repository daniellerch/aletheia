function EBS_COLOR(cover, payload, stego)
 
% Script modified by R�mi Cogranne, based upon the ones provided by Vojtech Holub (in 2014)
% use:
% toto = jpeg_read('./test.jpg');
% payload = 0.4;
% S_STRUCT = EBS(toto, payload)
%


try
    C_STRUCT = jpeg_read(cover);
catch
    error('ERROR (problem with the cover image)');
end




wetConst = 10^13;

STEGO=C_STRUCT;
for index_color=1:3  
    if (index_color==1) 
        C_QUANT = C_STRUCT.quant_tables{index_color};
    else
        C_QUANT = C_STRUCT.quant_tables{2};
    end

    DCT_rounded = C_STRUCT.coef_arrays{index_color};
    %% Block Entropy Cost
    rho_ent = zeros(size(DCT_rounded));
    for row=1:size(DCT_rounded, 1)/8
        for col=1:size(DCT_rounded, 2)/8
            all_coeffs = DCT_rounded(row*8-7:row*8, col*8-7:col*8);
            all_coeffs(1, 1) = 0;  %remove DC
            nzAC_coeffs = all_coeffs(all_coeffs ~= 0);
            nzAC_unique_coeffs = unique(nzAC_coeffs);
            if numel(nzAC_unique_coeffs) > 1
                b = hist(nzAC_coeffs, nzAC_unique_coeffs);
                b = b(b~=0);
                p = b ./ sum(b);
                H_block = -sum(p.*log(p));
            else
                H_block = 0;
            end
            rho_ent(row*8-7:row*8, col*8-7:col*8) = 1/(H_block^2);
        end
    end
    
    qi = repmat(C_QUANT, size(DCT_rounded) ./ 8);
    
    rho_f =  (qi).^2;
    
    %% Final cost
    rho = rho_ent .* rho_f;
    
    rho = rho + 10^(-4);
    rho(rho > wetConst) = wetConst;
    rho(isnan(rho)) = wetConst;
    rho(DCT_rounded > 1022) = wetConst;
    rho(DCT_rounded < -1022) = wetConst;
    
    maxCostMat = false(size(rho));
    maxCostMat(1:8:end, 1:8:end) = true;
    maxCostMat(5:8:end, 1:8:end) = true;
    maxCostMat(1:8:end, 5:8:end) = true;
    maxCostMat(5:8:end, 5:8:end) = true;
    
    rho(maxCostMat) = wetConst;
    changeable = (rho< wetConst);
    rho=rho(changeable);

    
    %% Compute message lenght for each run
    nzAC = nnz(DCT_rounded)-nnz(DCT_rounded(1:8:end,1:8:end)); % number of nonzero AC DCT coefficients
    
    %% Embedding
    % permutes path
    DCT_rounded_changeable=DCT_rounded(changeable);
    perm = randperm(numel(DCT_rounded_changeable));
    
    [LSBs] = EmbeddingSimulator(DCT_rounded_changeable(perm), rho(perm)', round(payload*nzAC));
    
    LSBs(perm) = LSBs;                           % inverse permutation
    
    % Create stego coefficients
    temp = mod(DCT_rounded_changeable, 2);
    ToBeChanged = zeros(size(DCT_rounded_changeable));
    ToBeChanged(temp == LSBs) = DCT_rounded_changeable(temp == LSBs);
    ToBeChanged(temp ~= LSBs) = DCT_rounded_changeable(temp ~= LSBs) + ((rand(1)>0.5)-0.5)*2;
    
    
    S_COEFFS = DCT_rounded;
    S_COEFFS(changeable) = ToBeChanged;

    STEGO.coef_arrays{index_color} = S_COEFFS;

    try
        jpeg_write(STEGO, stego);
    catch
        error('ERROR (problem with saving the stego image)')
    end

end

end


function [LSBs] = EmbeddingSimulator(x, rho, m)
       
    rho = rho';
    x = double(x);
    n = numel(x);
    
    lambda = calc_lambda(rho, m, n);
    pChange = 1 - (double(1)./(1+exp(-lambda.*rho)));
    
    %RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
    rand('state', sum(100*clock));
    randChange = rand(size(x));
    flippedPixels = (randChange < pChange);
    LSBs = mod(x + flippedPixels, 2);
    
    function lambda = calc_lambda(rho, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;
        while m3 > message_length
            l3 = l3 * 2;
            p = double(1)./(1 + exp(-l3 .* rho));
            m3 = binary_entropyf(p);
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
            p = double(1)./(1+exp(-lambda.*rho));
            m2 = binary_entropyf(p);
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
    
    function Hb = binary_entropyf(p)
        p = p(:);
        Hb = (-p.*log2(p))-((1-p).*log2(1-p));
        Hb(isnan(Hb)) = 0;
        Hb = sum(Hb);
    end

end

