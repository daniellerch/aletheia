function [stego, distortion] = HUGO(cover, payload)
params.gamma = 1;
params.sigma = 1;


cover = double(imread(cover));
wetCost = 10^8;
responseP1 = [0; 0; -1; +1; 0; 0];

% create mirror padded cover image
padSize = 3;
coverPadded = padarray(cover, [padSize padSize], 'symmetric');

% create residuals
C_Rez_H = coverPadded(:, 1:end-1) - coverPadded(:, 2:end);
C_Rez_V = coverPadded(1:end-1, :) - coverPadded(2:end, :);
C_Rez_Diag = coverPadded(1:end-1, 1:end-1) - coverPadded(2:end, 2:end);
C_Rez_MDiag = coverPadded(1:end-1, 2:end) - coverPadded(2:end, 1:end-1);

stego = cover;                                  % initialize stego image
stegoPadded = coverPadded;
        
% create residuals
S_Rez_H = stegoPadded(:, 1:end-1) - stegoPadded(:, 2:end);
S_Rez_V = stegoPadded(1:end-1, :) - stegoPadded(2:end, :);
S_Rez_Diag = stegoPadded(1:end-1, 1:end-1) - stegoPadded(2:end, 2:end);
S_Rez_MDiag = stegoPadded(1:end-1, 2:end) - stegoPadded(2:end, 1:end-1);
        
rhoM1 = zeros(size(cover));                    % declare cost of -1 change           
rhoP1 = zeros(size(cover));                    % declare cost of +1 change        
        
%% Iterate over elements in the sublattice
for row=1:size(cover, 1)
    for col=1:size(cover, 2)    
        D_P1 = 0;
        D_M1 = 0;
            
        % Horizontal
        cover_sub = C_Rez_H(row+3, col:col+5)';
        stego_sub = S_Rez_H(row+3, col:col+5)';
            
        stego_sub_P1 = stego_sub + responseP1;
        stego_sub_M1 = stego_sub - responseP1;

        D_M1 = D_M1 + GetLocalDistortion(cover_sub, stego_sub_M1, params);
        D_P1 = D_P1 + GetLocalDistortion(cover_sub, stego_sub_P1, params);
            
        % Vertical
        cover_sub = C_Rez_V(row:row+5, col+3);
        stego_sub = S_Rez_V(row:row+5, col+3);
           
        stego_sub_P1 = stego_sub + responseP1;
        stego_sub_M1 = stego_sub - responseP1;

        D_M1 = D_M1 + GetLocalDistortion(cover_sub, stego_sub_M1, params);
        D_P1 = D_P1 + GetLocalDistortion(cover_sub, stego_sub_P1, params);            

        % Diagonal
        cover_sub = [C_Rez_Diag(row, col); C_Rez_Diag(row+1, col+1); C_Rez_Diag(row+2, col+2); C_Rez_Diag(row+3, col+3); C_Rez_Diag(row+4, col+4); C_Rez_Diag(row+5, col+5)];
        stego_sub = [S_Rez_Diag(row, col); S_Rez_Diag(row+1, col+1); S_Rez_Diag(row+2, col+2); S_Rez_Diag(row+3, col+3); S_Rez_Diag(row+4, col+4); S_Rez_Diag(row+5, col+5)];
            
        stego_sub_P1 = stego_sub + responseP1;
        stego_sub_M1 = stego_sub - responseP1;

        D_M1 = D_M1 + GetLocalDistortion(cover_sub, stego_sub_M1, params);
        D_P1 = D_P1 + GetLocalDistortion(cover_sub, stego_sub_P1, params);
            
        % Minor Diagonal
        cover_sub = [C_Rez_MDiag(row, col+5); C_Rez_MDiag(row+1, col+4); C_Rez_MDiag(row+2, col+3); C_Rez_MDiag(row+3, col+2); C_Rez_MDiag(row+4, col+1); C_Rez_MDiag(row+5, col)];
        stego_sub = [S_Rez_MDiag(row, col+5); S_Rez_MDiag(row+1, col+4); S_Rez_MDiag(row+2, col+3); S_Rez_MDiag(row+3, col+2); S_Rez_MDiag(row+4, col+1); S_Rez_MDiag(row+5, col)];

        stego_sub_P1 = stego_sub + responseP1;
        stego_sub_M1 = stego_sub - responseP1;

        D_M1 = D_M1 + GetLocalDistortion(cover_sub, stego_sub_M1, params);
        D_P1 = D_P1 + GetLocalDistortion(cover_sub, stego_sub_P1, params);
            
        rhoM1(row, col) = D_M1;
        rhoP1(row, col) = D_P1;            
    end
end      
        
% truncation of the costs
rhoM1(rhoM1>wetCost) = wetCost;
rhoP1(rhoP1>wetCost) = wetCost;
        
rhoP1(cover == 255) = wetCost;
rhoM1(cover == 0) = wetCost;
               
%% Embedding   
% embedding simulator - params.qarity \in {2,3}
stego = EmbeddingSimulator(cover, rhoP1, rhoM1, round(numel(cover)*payload), false);

% compute distortion
distM1 = rhoM1(stego-cover==-1);
distP1 = rhoP1(stego-cover==1);
distortion = sum(distM1) + sum(distP1);

end






% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound). 
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
function [y] = EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges)

    n = numel(x);   
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    if fixEmbeddingChanges == 1
        %RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187));
        rand('state', 139187);
    else
        %RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
        rand('state', sum(100*clock));
    end
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
            if (iterations > 100)
                lambda = l3;
                return;
            end
        end        
        
        l1 = 0; 
        m1 = double(n);        
        lambda = 0;
        
        alpha = double(message_length)/n;
        % limit search to 100 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload        
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<100)
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
        p0 = 1-pP1-pM1;
        P = [p0(:); pP1(:); pM1(:)];
        H = -((P).*log2(P));
        H((P<eps) | (P > 1-eps)) = 0;
        Ht = sum(H);
    end
end

function D = GetLocalDistortion(C_resVect, S_resVect, params)

    D = 0;
    % C_resVect and S_resVect must have size of 6x1   
    D = D + GetLocalPotential(C_resVect(1:3), S_resVect(1:3), params);
    D = D + GetLocalPotential(C_resVect(2:4), S_resVect(2:4), params);
    D = D + GetLocalPotential(C_resVect(3:5), S_resVect(3:5), params);
    D = D + GetLocalPotential(C_resVect(4:6), S_resVect(4:6), params);

end

function Vc = GetLocalPotential(c_res, s_res, params)

    c_w = (params.sigma + sqrt(sum(c_res.^2))).^(-params.gamma);
    s_w = (params.sigma + sqrt(sum(s_res.^2))).^(-params.gamma);
    Vc = (c_w + s_w);
end


