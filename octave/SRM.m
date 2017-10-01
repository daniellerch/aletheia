function f = SRM(IMAGE)
% -------------------------------------------------------------------------
% Copyright (c) 2011 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Contact: jan@kodovsky.com | fridrich@binghamton.edu | October 2011
%          http://dde.binghamton.edu/download/feature_extractors
% -------------------------------------------------------------------------
% Extracts all 106 submodels presented in [1] as part of a rich model for
% steganalysis of digital images. All features are calculated in the
% spatial domain and are stored in a structured variable 'f'. For more
% deatils about the individual submodels, please see the publication [1].
% Total dimensionality of all 106 submodels is 34671.
% -------------------------------------------------------------------------
% Input:  IMAGE ... path to the image (can be JPEG)
% Output: f ....... extracted SRM features in a structured format
% -------------------------------------------------------------------------
% [1] Rich Models for Steganalysis of Digital Images, J. Fridrich and J.
% Kodovsky, IEEE Transactions on Information Forensics and Security, 2011.
% Under review.
% -------------------------------------------------------------------------

X = double(imread(IMAGE));

f = post_processing(all1st(X,1),'f1',1);   % 1st order, q=1
f = post_processing(all1st(X,2),'f1',2,f); % 1st order, q=2
for q=[1 1.5 2], f = post_processing(all2nd(X,q*2),'f2',q,f); end    % 2nd order
for q=[1 1.5 2], f = post_processing(all3rd(X,q*3),'f3',q,f); end    % 3rd order
for q=[1 1.5 2], f = post_processing(all3x3(X,q*4),'f3x3',q,f); end  % 3x3
for q=[1 1.5 2], f = post_processing(all5x5(X,q*12),'f5x5',q,f); end % 5x5

function RESULT = post_processing(DATA,f,q,RESULT)

Ss = fieldnames(DATA);
for Sid = 1:length(Ss)
    VARNAME = [f '_' Ss{Sid} '_q' strrep(num2str(q),'.','')];
    eval(['RESULT.' VARNAME ' = reshape(single(DATA.' Ss{Sid} '),1,[]);' ])
end

% symmetrize
L = fieldnames(RESULT);
for i=1:length(L)
    name = L{i}; % feature name
    if name(1)=='s', continue; end
    [T,N,Q] = parse_feaname(name);
    if strcmp(T,''), continue; end
    % symmetrization
    if strcmp(N(1:3),'min') || strcmp(N(1:3),'max')
        % minmax symmetrization
        OUT = ['s' T(2:end) '_minmax' N(4:end) '_' Q];
        if isfield(RESULT,OUT), continue; end
        eval(['Fmin = RESULT.' strrep(name,'max','min') ';']);
        eval(['Fmax = RESULT.' strrep(name,'min','max') ';']);
        F = symfea([Fmin Fmax]',2,4,'mnmx')'; %#ok<*NASGU>
        eval(['RESULT.' OUT ' = single(F);' ]);
    elseif strcmp(N(1:4),'spam')
        % spam symmetrization
        OUT = ['s' T(2:end) '_' N '_' Q];
        if isfield(RESULT,OUT), continue; end
        eval(['Fold = RESULT.' name ';']);
        F = symm1(Fold',2,4)';
        eval(['RESULT.' OUT ' = single(F);' ]);
    end
end
% delete RESULT.f*
L = fieldnames(RESULT);
for i=1:length(L)
    name = L{i}; % feature name
    if name(1)=='f'
        RESULT = rmfield(RESULT,name);
    end
end
% merge spam features
L = fieldnames(RESULT);
for i=1:length(L)
    name = L{i}; % feature name
    [T,N,Q] = parse_feaname(name);
    if ~strcmp(N(1:4),'spam'), continue; end
    if strcmp(T,''), continue; end
    if strcmp(N(end),'v')||(strcmp(N,'spam11')&&strcmp(T,'s5x5'))
    elseif strcmp(N(end),'h')
        % h+v union
        OUT = [T '_' N 'v_' Q ];
        if isfield(RESULT,OUT), continue; end
        name2 = strrep(name,'h_','v_');
        eval(['Fh = RESULT.' name ';']);
        eval(['Fv = RESULT.' name2 ';']);
        eval(['RESULT.' OUT ' = [Fh Fv];']);
        RESULT = rmfield(RESULT,name);
        RESULT = rmfield(RESULT,name2);
    elseif strcmp(N,'spam11')
        % KBKV creation
        OUT = ['s35_' N '_' Q];
        if isfield(RESULT,OUT), continue; end
        name1 = strrep(name,'5x5','3x3');
        name2 = strrep(name,'3x3','5x5');
        if ~isfield(RESULT,name1), continue; end
        if ~isfield(RESULT,name2), continue; end
        eval(['F_KB = RESULT.' name1 ';']);
        eval(['F_KV = RESULT.' name2 ';']);
        eval(['RESULT.' OUT ' = [F_KB F_KV];']);
        RESULT = rmfield(RESULT,name1);
        RESULT = rmfield(RESULT,name2);
    end
end
function [T,N,Q] = parse_feaname(name)
[T,N,Q] = deal('');
P = strfind(name,'_'); if length(P)~=2, return; end
T = name(1:P(1)-1);
N = name(P(1)+1:P(2)-1);
Q = name(P(2)+1:end);
function g = all1st(X,q)
%
% X must be a matrix of doubles or singles (the image) and q is the 
% quantization step (any positive number).
%
% Recommended values of q are c, 1.5c, 2c, where c is the central
% coefficient in the differential (at X(I,J)).
%
% This function outputs co-occurrences of ALL 1st-order residuals
% listed in Figure 1 in our journal HUGO paper (version from June 14), 
% including the naming convention.
%
% List of outputted features:
%
% 1a) spam14h
% 1b) spam14v (orthogonal-spam)
% 1c) minmax22v
% 1d) minmax24
% 1e) minmax34v
% 1f) minmax41
% 1g) minmax34
% 1h) minmax48h
% 1i) minmax54
%
% Naming convention:
%
% name = {type}{f}{sigma}{scan}
% type \in {spam, minmax}
% f \in {1,2,3,4,5} number of filters that are "minmaxed"
% sigma \in {1,2,3,4,8} symmetry index
% scan \in {h,v,\emptyset} scan of the cooc matrix (empty = sum of both 
% h and v scans).
%
% All odd residuals are implemented the same way simply by
% narrowing the range for I and J and replacing the residuals --
% -- they should "stick out" (trcet) in the same direction as 
% the 1st order ones. For example, for the 3rd order:
%
% RU = -X(I-2,J+2)+3*X(I-1,J+1)-3*X(I,J)+X(I+1,J-1); ... etc.
%
% Note1: The term X(I,J) should always have the "-" sign.
% Note2: This script does not include s, so, cout, cin versions (weak).
% This function calls Cooc.m and Quant.m

[M N] = size(X); [I,J,T,order] = deal(2:M-1,2:N-1,2,4);
% Variable names are self-explanatory (R = right, U = up, L = left, D = down)
[R,L,U,D]  = deal(X(I,J+1)-X(I,J),X(I,J-1)-X(I,J),X(I-1,J)-X(I,J),X(I+1,J)-X(I,J));
[Rq,Lq,Uq,Dq] = deal(Quant(R,q,T),Quant(L,q,T),Quant(U,q,T),Quant(D,q,T));
[RU,LU,RD,LD] = deal(X(I-1,J+1)-X(I,J),X(I-1,J-1)-X(I,J),X(I+1,J+1)-X(I,J),X(I+1,J-1)-X(I,J));
[RUq,RDq,LUq,LDq] = deal(Quant(RU,q,T),Quant(RD,q,T),Quant(LU,q,T),Quant(LD,q,T));
% minmax22h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical.
[RLq_min,UDq_min,RLq_max,UDq_max] = deal(min(Rq,Lq),min(Uq,Dq),max(Rq,Lq),max(Uq,Dq));
g.min22h = reshape(Cooc(RLq_min,order,'hor',T) + Cooc(UDq_min,order,'ver',T),[],1);
g.max22h = reshape(Cooc(RLq_max,order,'hor',T) + Cooc(UDq_max,order,'ver',T),[],1);
% minmax34h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical
[Uq_min,Rq_min,Dq_min,Lq_min] = deal(min(min(Lq,Uq),Rq),min(min(Uq,Rq),Dq),min(min(Rq,Dq),Lq),min(min(Dq,Lq),Uq));
[Uq_max,Rq_max,Dq_max,Lq_max] = deal(max(max(Lq,Uq),Rq),max(max(Uq,Rq),Dq),max(max(Rq,Dq),Lq),max(max(Dq,Lq),Uq));
g.min34h = reshape(Cooc([Uq_min;Dq_min],order,'hor',T) + Cooc([Lq_min Rq_min],order,'ver',T),[],1);
g.max34h = reshape(Cooc([Uq_max;Dq_max],order,'hor',T) + Cooc([Rq_max Lq_max],order,'ver',T),[],1);
% spam14h/v -- to be symmetrized as spam, directional, hv-nonsymmetrical
g.spam14h = reshape(Cooc(Rq,order,'hor',T) + Cooc(Uq,order,'ver',T),[],1);
g.spam14v = reshape(Cooc(Rq,order,'ver',T) + Cooc(Uq,order,'hor',T),[],1);
% minmax22v -- to be symmetrized as mnmx, directional, hv-nonsymmetrical. Good with higher-order residuals! Note: 22h is bad (too much neighborhood overlap).
g.min22v = reshape(Cooc(RLq_min,order,'ver',T) + Cooc(UDq_min,order,'hor',T),[],1);
g.max22v = reshape(Cooc(RLq_max,order,'ver',T) + Cooc(UDq_max,order,'hor',T),[],1);
% minmax24 -- to be symmetrized as mnmx, directional, hv-symmetrical. Darn good, too.
[RUq_min,RDq_min,LUq_min,LDq_min] = deal(min(Rq,Uq),min(Rq,Dq),min(Lq,Uq),min(Lq,Dq));
[RUq_max,RDq_max,LUq_max,LDq_max] = deal(max(Rq,Uq),max(Rq,Dq),max(Lq,Uq),max(Lq,Dq));
g.min24 = reshape(Cooc([RUq_min;RDq_min;LUq_min;LDq_min],order,'hor',T) + Cooc([RUq_min RDq_min LUq_min LDq_min],order,'ver',T),[],1);
g.max24 = reshape(Cooc([RUq_max;RDq_max;LUq_max;LDq_max],order,'hor',T) + Cooc([RUq_max RDq_max LUq_max LDq_max],order,'ver',T),[],1);
% minmax34v -- v works well, h does not, to be symmetrized as mnmx, directional, hv-nonsymmetrical
g.min34v = reshape(Cooc([Uq_min Dq_min],order,'ver',T) + Cooc([Rq_min;Lq_min],order,'hor',T),[],1);
g.max34v = reshape(Cooc([Uq_max Dq_max],order,'ver',T) + Cooc([Rq_max;Lq_max],order,'hor',T),[],1);
% minmax41 -- to be symmetrized as mnmx, non-directional, hv-symmetrical
[R_min,R_max] = deal(min(RLq_min,UDq_min),max(RLq_max,UDq_max));
g.min41 = reshape(Cooc(R_min,order,'hor',T) + Cooc(R_min,order,'ver',T),[],1);
g.max41 = reshape(Cooc(R_max,order,'hor',T) + Cooc(R_max,order,'ver',T),[],1);
% minmax34 -- good, to be symmetrized as mnmx, directional, hv-symmetrical
[RUq_min,RDq_min,LUq_min,LDq_min] = deal(min(RUq_min,RUq),min(RDq_min,RDq),min(LUq_min,LUq),min(LDq_min,LDq));
[RUq_max,RDq_max,LUq_max,LDq_max] = deal(max(RUq_max,RUq),max(RDq_max,RDq),max(LUq_max,LUq),max(LDq_max,LDq));
g.min34 = reshape(Cooc([RUq_min;RDq_min;LUq_min;LDq_min],order,'hor',T) + Cooc([RUq_min RDq_min LUq_min LDq_min],order,'ver',T),[],1);
g.max34 = reshape(Cooc([RUq_max;RDq_max;LUq_max;LDq_max],order,'hor',T) + Cooc([RUq_max RDq_max LUq_max LDq_max],order,'ver',T),[],1);
% minmax48h -- h better than v, to be symmetrized as mnmx, directional, hv-nonsymmetrical. 48v is almost as good as 48h; for 3rd-order but weaker for 1st-order. Here, I am outputting both but Figure 1 in our paper lists only 48h.
[RUq_min2,RDq_min2,LDq_min2,LUq_min2] = deal(min(RUq_min,LUq),min(RDq_min,RUq),min(LDq_min,RDq),min(LUq_min,LDq));
[RUq_min3,RDq_min3,LDq_min3,LUq_min3] = deal(min(RUq_min,RDq),min(RDq_min,LDq),min(LDq_min,LUq),min(LUq_min,RUq));
g.min48h = reshape(Cooc([RUq_min2;LDq_min2;RDq_min3;LUq_min3],order,'hor',T) + Cooc([RDq_min2 LUq_min2 RUq_min3 LDq_min3],order,'ver',T),[],1);
g.min48v = reshape(Cooc([RDq_min2;LUq_min2;RUq_min3;LDq_min3],order,'hor',T) + Cooc([RUq_min2 LDq_min2 RDq_min3 LUq_min3],order,'ver',T),[],1);
[RUq_max2,RDq_max2,LDq_max2,LUq_max2] = deal(max(RUq_max,LUq),max(RDq_max,RUq),max(LDq_max,RDq),max(LUq_max,LDq));
[RUq_max3,RDq_max3,LDq_max3,LUq_max3] = deal(max(RUq_max,RDq),max(RDq_max,LDq),max(LDq_max,LUq),max(LUq_max,RUq));
g.max48h = reshape(Cooc([RUq_max2;LDq_max2;RDq_max3;LUq_max3],order,'hor',T) + Cooc([RDq_max2 LUq_max2 RUq_max3 LDq_max3],order,'ver',T),[],1);
g.max48v = reshape(Cooc([RDq_max2;LUq_max2;RUq_max3;LDq_max3],order,'hor',T) + Cooc([RUq_max2 LDq_max2 RDq_max3 LUq_max3],order,'ver',T),[],1);
% minmax54 -- to be symmetrized as mnmx, directional, hv-symmetrical
[RUq_min4,RDq_min4,LDq_min4,LUq_min4] = deal(min(RUq_min2,RDq),min(RDq_min2,LDq),min(LDq_min2,LUq),min(LUq_min2,RUq));
[RUq_min5,RDq_min5,LDq_min5,LUq_min5] = deal(min(RUq_min3,LUq),min(RDq_min3,RUq),min(LDq_min3,RDq),min(LUq_min3,LDq));
g.min54 = reshape(Cooc([RUq_min4;LDq_min4;RDq_min5;LUq_min5],order,'hor',T) + Cooc([RDq_min4 LUq_min4 RUq_min5 LDq_min5],order,'ver',T),[],1);
[RUq_max4,RDq_max4,LDq_max4,LUq_max4] = deal(max(RUq_max2,RDq),max(RDq_max2,LDq),max(LDq_max2,LUq),max(LUq_max2,RUq));
[RUq_max5,RDq_max5,LDq_max5,LUq_max5] = deal(max(RUq_max3,LUq),max(RDq_max3,RUq),max(LDq_max3,RDq),max(LUq_max3,LDq));
g.max54 = reshape(Cooc([RUq_max4;LDq_max4;RDq_max5;LUq_max5],order,'hor',T) + Cooc([RDq_max4 LUq_max4 RUq_max5 LDq_max5],order,'ver',T),[],1);
function g = all2nd(X,q)
%
% X must be a matrix of doubles or singles (the image) and q is the 
% quantization step (any positive number).
%
% Recommended values of q are c, 1.5c, 2c, where c is the central
% coefficient in the differential (at X(I,J)).
%
% This function outputs co-occurrences of ALL 2nd-order residuals
% listed in Figure 1 in our journal HUGO paper (version from June 14), 
% including the naming convention.
%
% List of outputted features:
%
% 1a) spam12h
% 1b) spam12v (orthogonal-spam)
% 1c) minmax21
% 1d) minmax41
% 1e) minmax24h (24v is also outputted but not listed in Figure 1)
% 1f) minmax32
%
% Naming convention:
%
% name = {type}{f}{sigma}{scan}
% type \in {spam, minmax}
% f \in {1,2,3,4,5} number of filters that are "minmaxed"
% sigma \in {1,2,3,4,8} symmetry index
% scan \in {h,v,\emptyset} scan of the cooc matrix (empty = sum of both 
% h and v scans).
%
% All even residuals are implemented the same way simply by
% narrowing the range for I and J and replacing the residuals.
%
% Note1: The term X(I,J) should always have the "-" sign.
% Note2: This script does not include s, so, cout, cin versions (weak).
%
% This function calls Residual.m, Cooc.m, and Quant.m

[T,order] = deal(2,4);
% 2nd-order residuals are implemented using Residual.m
[Dh,Dv,Dd,Dm] = deal(Residual(X,2,'hor'),Residual(X,2,'ver'),Residual(X,2,'diag'),Residual(X,2,'mdiag'));
[Yh,Yv,Yd,Ym] = deal(Quant(Dh,q,T),Quant(Dv,q,T),Quant(Dd,q,T),Quant(Dm,q,T));
% spam12h/v
g.spam12h = reshape(Cooc(Yh,order,'hor',T) + Cooc(Yv,order,'ver',T),[],1);
g.spam12v = reshape(Cooc(Yh,order,'ver',T) + Cooc(Yv,order,'hor',T),[],1);
% minmax21
[Dmin,Dmax] = deal(min(Yh,Yv),max(Yh,Yv));
g.min21 = reshape(Cooc(Dmin,order,'hor',T) + Cooc(Dmin,order,'ver',T),[],1);
g.max21 = reshape(Cooc(Dmax,order,'hor',T) + Cooc(Dmax,order,'ver',T),[],1);
% minmax41   
[Dmin2,Dmax2] = deal(min(Dmin,min(Yd,Ym)),max(Dmax,max(Yd,Ym)));
g.min41 = reshape(Cooc(Dmin2,order,'hor',T) + Cooc(Dmin2,order,'ver',T),[],1);
g.max41 = reshape(Cooc(Dmax2,order,'hor',T) + Cooc(Dmax2,order,'ver',T),[],1);
% minmax32 -- good, directional, hv-symmetrical, to be symmetrized as mnmx
[RUq_min,RDq_min] = deal(min(Dmin,Ym),min(Dmin,Yd));
[RUq_max,RDq_max] = deal(max(Dmax,Ym),max(Dmax,Yd));
g.min32 = reshape(Cooc([RUq_min;RDq_min],order,'hor',T) + Cooc([RUq_min RDq_min],order,'ver',T),[],1);
g.max32 = reshape(Cooc([RUq_max;RDq_max],order,'hor',T) + Cooc([RUq_max RDq_max],order,'ver',T),[],1);
% minmax24h,v -- both "not bad," h slightly better, directional, hv-nonsymmetrical, to be symmetrized as mnmx
[RUq_min2,RDq_min2,RUq_min3,LUq_min3] = deal(min(Ym,Yh),min(Yd,Yh),min(Ym,Yv),min(Yd,Yv));
g.min24h = reshape(Cooc([RUq_min2;RDq_min2],order,'hor',T)+Cooc([RUq_min3 LUq_min3],order,'ver',T),[],1);
g.min24v = reshape(Cooc([RUq_min2 RDq_min2],order,'ver',T)+Cooc([RUq_min3;LUq_min3],order,'hor',T),[],1);
[RUq_max2,RDq_max2,RUq_max3,LUq_max3] = deal(max(Ym,Yh),max(Yd,Yh),max(Ym,Yv),max(Yd,Yv));
g.max24h = reshape(Cooc([RUq_max2;RDq_max2],order,'hor',T)+Cooc([RUq_max3 LUq_max3],order,'ver',T),[],1);
g.max24v = reshape(Cooc([RUq_max2 RDq_max2],order,'ver',T)+Cooc([RUq_max3;LUq_max3],order,'hor',T),[],1);
function g = all3rd(X,q)
%
% X must be a matrix of doubles or singles (the image) and q is the 
% quantization step (any positive number).
%
% Recommended values of q are c, 1.5c, 2c, where c is the central
% coefficient in the differential (at X(I,J)).
%
% This function outputs co-occurrences of ALL 3rd-order residuals
% listed in Figure 1 in our journal HUGO paper (version from June 14), 
% including the naming convention.
%
% List of outputted features:
%
% 1a) spam14h
% 1b) spam14v (orthogonal-spam)
% 1c) minmax22v
% 1d) minmax24
% 1e) minmax34v
% 1f) minmax41
% 1g) minmax34
% 1h) minmax48h
% 1i) minmax54
%
% Naming convention:
%
% name = {type}{f}{sigma}{scan}
% type \in {spam, minmax}
% f \in {1,2,3,4,5} number of filters that are "minmaxed"
% sigma \in {1,2,3,4,8} symmetry index
% scan \in {h,v,\emptyset} scan of the cooc matrix (empty = sum of both 
% h and v scans).
%
% All odd residuals are implemented the same way simply by
% narrowing the range for I and J and replacing the residuals --
% -- they should "stick out" (trcet) in the same direction as 
% the 1st order ones. For example, for the 3rd order:
%
% RU = -X(I-2,J+2)+3*X(I-1,J+1)-3*X(I,J)+X(I+1,J-1); ... etc.
%
% Note1: The term X(I,J) should always have the "-" sign.
% Note2: This script does not include s, so, cout, cin versions (weak).

[M N] = size(X); [I,J,T,order] = deal(3:M-2,3:N-2,2,4);
[R,L,U,D] = deal(-X(I,J+2)+3*X(I,J+1)-3*X(I,J)+X(I,J-1),-X(I,J-2)+3*X(I,J-1)-3*X(I,J)+X(I,J+1),-X(I-2,J)+3*X(I-1,J)-3*X(I,J)+X(I+1,J),-X(I+2,J)+3*X(I+1,J)-3*X(I,J)+X(I-1,J));
[Rq,Lq,Uq,Dq] = deal(Quant(R,q,T),Quant(L,q,T),Quant(U,q,T),Quant(D,q,T));
[RU,LU,RD,LD] = deal(-X(I-2,J+2)+3*X(I-1,J+1)-3*X(I,J)+X(I+1,J-1),-X(I-2,J-2)+3*X(I-1,J-1)-3*X(I,J)+X(I+1,J+1),-X(I+2,J+2)+3*X(I+1,J+1)-3*X(I,J)+X(I-1,J-1),-X(I+2,J-2)+3*X(I+1,J-1)-3*X(I,J)+X(I-1,J+1));
[RUq,RDq,LUq,LDq] = deal(Quant(RU,q,T),Quant(RD,q,T),Quant(LU,q,T),Quant(LD,q,T));
% minmax22h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical
[RLq_min,UDq_min] = deal(min(Rq,Lq),min(Uq,Dq));
[RLq_max,UDq_max] = deal(max(Rq,Lq),max(Uq,Dq));
g.min22h = reshape(Cooc(RLq_min,order,'hor',T) + Cooc(UDq_min,order,'ver',T),[],1);
g.max22h = reshape(Cooc(RLq_max,order,'hor',T) + Cooc(UDq_max,order,'ver',T),[],1);
% minmax34h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical
[Uq_min,Rq_min,Dq_min,Lq_min] = deal(min(RLq_min,Uq),min(UDq_min,Rq),min(RLq_min,Dq),min(UDq_min,Lq));
[Uq_max,Rq_max,Dq_max,Lq_max] = deal(max(RLq_max,Uq),max(UDq_max,Rq),max(RLq_max,Dq),max(UDq_max,Lq));
g.min34h = reshape(Cooc([Uq_min;Dq_min],order,'hor',T)+Cooc([Rq_min Lq_min],order,'ver',T),[],1);
g.max34h = reshape(Cooc([Uq_max;Dq_max],order,'hor',T)+Cooc([Rq_max Lq_max],order,'ver',T),[],1);
% spam14h,v -- to be symmetrized as spam, directional, hv-nonsymmetrical
g.spam14h = reshape(Cooc(Rq,order,'hor',T) + Cooc(Uq,order,'ver',T),[],1);
g.spam14v = reshape(Cooc(Rq,order,'ver',T) + Cooc(Uq,order,'hor',T),[],1);
% minmax22v -- to be symmetrized as mnmx, directional, hv-nonsymmetrical. Good with higher-order residuals! Note: 22h is bad (too much neighborhood overlap).
g.min22v = reshape(Cooc(RLq_min,order,'ver',T) + Cooc(UDq_min,order,'hor',T),[],1);
g.max22v = reshape(Cooc(RLq_max,order,'ver',T) + Cooc(UDq_max,order,'hor',T),[],1);
% minmax24 -- to be symmetrized as mnmx, directional, hv-symmetrical Note: Darn good, too.
[RUq_min,RDq_min,LUq_min,LDq_min] = deal(min(Rq,Uq),min(Rq,Dq),min(Lq,Uq),min(Lq,Dq));
[RUq_max,RDq_max,LUq_max,LDq_max] = deal(max(Rq,Uq),max(Rq,Dq),max(Lq,Uq),max(Lq,Dq));
g.min24 = reshape(Cooc([RUq_min;RDq_min;LUq_min;LDq_min],order,'hor',T) + Cooc([RUq_min RDq_min LUq_min LDq_min],order,'ver',T),[],1);
g.max24 = reshape(Cooc([RUq_max;RDq_max;LUq_max;LDq_max],order,'hor',T) + Cooc([RUq_max RDq_max LUq_max LDq_max],order,'ver',T),[],1);
% minmax34v -- v works well, h does not, to be symmetrized as mnmx, directional, hv-nonsymmetrical
g.min34v = reshape(Cooc([Uq_min Dq_min],order,'ver',T) + Cooc([Rq_min;Lq_min],order,'hor',T),[],1);
g.max34v = reshape(Cooc([Uq_max Dq_max],order,'ver',T) + Cooc([Rq_max;Lq_max],order,'hor',T),[],1);
% minmax41 -- unknown performance as of 6/14/11, to be symmetrized as mnmx, non-directional, hv-symmetrical
[R_min,R_max] = deal(min(RUq_min,LDq_min),max(RUq_max,LDq_max));
g.min41 = reshape(Cooc(R_min,order,'hor',T) + Cooc(R_min,order,'ver',T),[],1);
g.max41 = reshape(Cooc(R_max,order,'hor',T) + Cooc(R_max,order,'ver',T),[],1);
% minmax34 -- good, to be symmetrized as mnmx, directional, hv-symmetrical
[RUq_min2,RDq_min2,LUq_min2,LDq_min2] = deal(min(RUq_min,RUq),min(RDq_min,RDq),min(LUq_min,LUq),min(LDq_min,LDq));
[RUq_max2,RDq_max2,LUq_max2,LDq_max2] = deal(max(RUq_max,RUq),max(RDq_max,RDq),max(LUq_max,LUq),max(LDq_max,LDq));
g.min34 = reshape(Cooc([RUq_min2;RDq_min2;LUq_min2;LDq_min2],order,'hor',T) + Cooc([RUq_min2 RDq_min2 LUq_min2 LDq_min2],order,'ver',T),[],1);
g.max34 = reshape(Cooc([RUq_max2;RDq_max2;LUq_max2;LDq_max2],order,'hor',T) + Cooc([RUq_max2 RDq_max2 LUq_max2 LDq_max2],order,'ver',T),[],1);
% minmax48h -- h better than v, to be symmetrized as mnmx, directional, hv-nonsymmetrical. 48v is almost as good as 48h for 3rd-order but weaker for 1st-order. Here, I am outputting both but Figure 1 in our paper lists only 48h.
[RUq_min3,RDq_min3,LDq_min3,LUq_min3] = deal(min(RUq_min2,LUq),min(RDq_min2,RUq),min(LDq_min2,RDq),min(LUq_min2,LDq));
[RUq_min4,RDq_min4,LDq_min4,LUq_min4] = deal(min(RUq_min2,RDq),min(RDq_min2,LDq),min(LDq_min2,LUq),min(LUq_min2,RUq));
g.min48h = reshape(Cooc([RUq_min3;LDq_min3;RDq_min4;LUq_min4],order,'hor',T)+Cooc([RDq_min3 LUq_min3 RUq_min4 LDq_min4],order,'ver',T),[],1);
g.min48v = reshape(Cooc([RUq_min3 LDq_min3 RDq_min4 LUq_min4],order,'ver',T)+Cooc([RDq_min3;LUq_min3;RUq_min4;LDq_min4],order,'hor',T),[],1);
[RUq_max3,RDq_max3,LDq_max3,LUq_max3] = deal(max(RUq_max2,LUq),max(RDq_max2,RUq),max(LDq_max2,RDq),max(LUq_max2,LDq));
[RUq_max4,RDq_max4,LDq_max4,LUq_max4] = deal(max(RUq_max2,RDq),max(RDq_max2,LDq),max(LDq_max2,LUq),max(LUq_max2,RUq));
g.max48h = reshape(Cooc([RUq_max3;LDq_max3;RDq_max4;LUq_max4],order,'hor',T)+Cooc([RDq_max3 LUq_max3 RUq_max4 LDq_max4],order,'ver',T),[],1);
g.max48v = reshape(Cooc([RUq_max3 LDq_max3 RDq_max4 LUq_max4],order,'ver',T)+Cooc([RDq_max3;LUq_max3;RUq_max4;LDq_max4],order,'hor',T),[],1);
% minmax54 -- to be symmetrized as mnmx, directional, hv-symmetrical
[RUq_min5,RDq_min5,LDq_min5,LUq_min5] = deal(min(RUq_min3,RDq),min(RDq_min3,LDq),min(LDq_min3,LUq),min(LUq_min3,RUq));
[RUq_max5,RDq_max5,LDq_max5,LUq_max5] = deal(max(RUq_max3,RDq),max(RDq_max3,LDq),max(LDq_max3,LUq),max(LUq_max3,RUq));
g.min54 = reshape(Cooc([RUq_min5;LDq_min5;RDq_min5;LUq_min5],order,'hor',T) + Cooc([RDq_min5 LUq_min5 RUq_min5 LDq_min5],order,'ver',T),[],1);
g.max54 = reshape(Cooc([RUq_max5;LDq_max5;RDq_max5;LUq_max5],order,'hor',T) + Cooc([RDq_max5 LUq_max5 RUq_max5 LDq_max5],order,'ver',T),[],1);
function g = all3x3(X,q)
% This function outputs co-occurrences of ALL residuals based on the
% KB kernel and its "halves" (EDGE residuals) as listed in Figure 1
% in our journal HUGO paper (version from June 14), including the naming
% convention.
[T,order] = deal(2,4);
% spam11 (old name KB residual), good, non-directional, hv-symmetrical, to be symmetrized as spam
D = Residual(X,2,'KB'); Y = Quant(D,q,T);
g.spam11 = reshape(Cooc(Y,order,'hor',T) + Cooc(Y,order,'ver',T),[],1);
% EDGE residuals
D = Residual(X,2,'edge-h');Du = D(:,1:size(D,2)/2);Db = D(:,size(D,2)/2+1:end);
D = Residual(X,2,'edge-v');Dl = D(:,1:size(D,2)/2);Dr = D(:,size(D,2)/2+1:end);
[Yu,Yb,Yl,Yr] = deal(Quant(Du,q,T),Quant(Db,q,T),Quant(Dl,q,T),Quant(Dr,q,T));
% spam14h,v  not bad, directional, hv-nonsym, to be symmetrized as spam
g.spam14v = reshape(Cooc([Yu Yb],order,'ver',T) + Cooc([Yl;Yr],order,'hor',T),[],1);
g.spam14h = reshape(Cooc([Yu;Yb],order,'hor',T) + Cooc([Yl Yr],order,'ver',T),[],1);
% minmax24 -- EXCELLENT, directional, hv-sym, to be symmetrized as mnmx
[Dmin1,Dmin2,Dmin3,Dmin4] = deal(min(Yu,Yl),min(Yb,Yr),min(Yu,Yr),min(Yb,Yl));
g.min24 = reshape(Cooc([Dmin1 Dmin2 Dmin3 Dmin4],order,'ver',T) + Cooc([Dmin1;Dmin2;Dmin3;Dmin4],order,'hor',T),[],1);
[Dmax1,Dmax2,Dmax3,Dmax4] = deal(max(Yu,Yl),max(Yb,Yr),max(Yu,Yr),max(Yb,Yl));
g.max24 = reshape(Cooc([Dmax1 Dmax2 Dmax3 Dmax4],order,'ver',T) + Cooc([Dmax1;Dmax2;Dmax3;Dmax4],order,'hor',T),[],1);
% minmax22 - hv-nonsymmetrical
% min22h -- good, to be symmetrized as mnmx, directional, hv-nonsymmetrical
% min22v -- EXCELLENT - to be symmetrized as mnmx, directional,
[UEq_min,REq_min] = deal(min(Yu,Yb),min(Yr,Yl));
g.min22h = reshape(Cooc(UEq_min,order,'hor',T) + Cooc(REq_min,order,'ver',T),[],1);
g.min22v = reshape(Cooc(UEq_min,order,'ver',T) + Cooc(REq_min,order,'hor',T),[],1);
[UEq_max,REq_max] = deal(max(Yu,Yb),max(Yr,Yl));
g.max22h = reshape(Cooc(UEq_max,order,'hor',T) + Cooc(REq_max,order,'ver',T),[],1);
g.max22v = reshape(Cooc(UEq_max,order,'ver',T) + Cooc(REq_max,order,'hor',T),[],1);
% minmax41 -- good, non-directional, hv-sym, to be symmetrized as mnmx
[Dmin5,Dmax5] = deal(min(Dmin1,Dmin2),max(Dmax1,Dmax2));
g.min41 = reshape(Cooc(Dmin5,order,'ver',T) + Cooc(Dmin5,order,'hor',T),[],1);
g.max41 = reshape(Cooc(Dmax5,order,'ver',T) + Cooc(Dmax5,order,'hor',T),[],1);
function g = all5x5(X,q)
% This function outputs co-occurrences of ALL residuals based on the
% KV kernel and its "halves" (EDGE residuals) as listed in Figure 1
% in our journal HUGO paper (version from June 14), including the naming
% convention.
[M N] = size(X); [I,J,T,order] = deal(3:M-2,3:N-2,2,4);
% spam11 (old name KV residual), good, non-directional, hv-symmetrical, to be symmetrized as spam
D = Residual(X,3,'KV'); Y = Quant(D,q,T);
g.spam11 = reshape(Cooc(Y,order,'hor',T) + Cooc(Y,order,'ver',T),[],1);
% EDGE residuals    
Du = 8*X(I,J-1)+8*X(I-1,J)+8*X(I,J+1)-6*X(I-1,J-1)-6*X(I-1,J+1)-2*X(I,J-2)-2*X(I,J+2)-2*X(I-2,J)+2*X(I-1,J-2)+2*X(I-2,J-1)+2*X(I-2,J+1)+2*X(I-1,J+2)-X(I-2,J-2)-X(I-2,J+2)-12*X(I,J);
Dr = 8*X(I-1,J)+8*X(I,J+1)+8*X(I+1,J)-6*X(I-1,J+1)-6*X(I+1,J+1)-2*X(I-2,J)-2*X(I+2,J)-2*X(I,J+2)+2*X(I-2,J+1)+2*X(I-1,J+2)+2*X(I+1,J+2)+2*X(I+2,J+1)-X(I-2,J+2)-X(I+2,J+2)-12*X(I,J);
Db = 8*X(I,J+1)+8*X(I+1,J)+8*X(I,J-1)-6*X(I+1,J+1)-6*X(I+1,J-1)-2*X(I,J-2)-2*X(I,J+2)-2*X(I+2,J)+2*X(I+1,J+2)+2*X(I+2,J+1)+2*X(I+2,J-1)+2*X(I+1,J-2)-X(I+2,J+2)-X(I+2,J-2)-12*X(I,J);
Dl = 8*X(I+1,J)+8*X(I,J-1)+8*X(I-1,J)-6*X(I+1,J-1)-6*X(I-1,J-1)-2*X(I-2,J)-2*X(I+2,J)-2*X(I,J-2)+2*X(I+2,J-1)+2*X(I+1,J-2)+2*X(I-1,J-2)+2*X(I-2,J-1)-X(I+2,J-2)-X(I-2,J-2)-12*X(I,J);
[Yu,Yb,Yl,Yr] = deal(Quant(Du,q,T),Quant(Db,q,T),Quant(Dl,q,T),Quant(Dr,q,T));
% spam14v  not bad, directional, hv-nonsym, to be symmetrized as spam
g.spam14v = reshape(Cooc([Yu Yb],order,'ver',T) + Cooc([Yl;Yr],order,'hor',T),[],1);
g.spam14h = reshape(Cooc([Yu;Yb],order,'hor',T) + Cooc([Yl Yr],order,'ver',T),[],1);
% minmax24 -- EXCELLENT, directional, hv-sym, to be symmetrized as mnmx
[Dmin1,Dmin2,Dmin3,Dmin4] = deal(min(Yu,Yl),min(Yb,Yr),min(Yu,Yr),min(Yb,Yl));
g.min24 = reshape(Cooc([Dmin1 Dmin2 Dmin3 Dmin4],order,'ver',T) + Cooc([Dmin1;Dmin2;Dmin3;Dmin4],order,'hor',T),[],1);
[Dmax1,Dmax2,Dmax3,Dmax4] = deal(max(Yu,Yl),max(Yb,Yr),max(Yu,Yr),max(Yb,Yl));
g.max24 = reshape(Cooc([Dmax1 Dmax2 Dmax3 Dmax4],order,'ver',T) + Cooc([Dmax1;Dmax2;Dmax3;Dmax4],order,'hor',T),[],1);
% minmax22 - hv-nonsymmetrical
% min22h -- good, to be symmetrized as mnmx, directional, hv-nonsymmetrical
% min22v -- EXCELLENT - to be symmetrized as mnmx, directional,
[UEq_min,REq_min] = deal(min(Yu,Yb),min(Yr,Yl));
g.min22h = reshape(Cooc(UEq_min,order,'hor',T) + Cooc(REq_min,order,'ver',T),[],1);
g.min22v = reshape(Cooc(UEq_min,order,'ver',T) + Cooc(REq_min,order,'hor',T),[],1);
[UEq_max,REq_max] = deal(max(Yu,Yb),max(Yr,Yl));
g.max22h = reshape(Cooc(UEq_max,order,'hor',T) + Cooc(REq_max,order,'ver',T),[],1);
g.max22v = reshape(Cooc(UEq_max,order,'ver',T) + Cooc(REq_max,order,'hor',T),[],1);
% minmax41 -- good, non-directional, hv-sym, to be symmetrized as mnmx
[Dmin5,Dmax5] = deal(min(Dmin1,Dmin2),max(Dmax1,Dmax2));
g.min41 = reshape(Cooc(Dmin5,order,'ver',T) + Cooc(Dmin5,order,'hor',T),[],1);
g.max41 = reshape(Cooc(Dmax5,order,'ver',T) + Cooc(Dmax5,order,'hor',T),[],1);
function f = Cooc(D,order,type,T)
% Co-occurrence operator to be appied to a 2D array of residuals D \in [-T,T]
% T     ... threshold
% order ... cooc order \in {1,2,3,4,5}
% type  ... cooc type \in {hor,ver,diag,mdiag,square,square-ori,hvdm}
% f     ... an array of size (2T+1)^order

B = 2*T+1;
if max(abs(D(:))) > T, fprintf('*** ERROR in Cooc.m: Residual out of range ***\n'), end

switch order
    case 1
        f = hist(D(:),-T:T);
    case 2
        f = zeros(B,B);
        if strcmp(type,'hor'),   L = D(:,1:end-1); R = D(:,2:end);end
        if strcmp(type,'ver'),   L = D(1:end-1,:); R = D(2:end,:);end
        if strcmp(type,'diag'),  L = D(1:end-1,1:end-1); R = D(2:end,2:end);end
        if strcmp(type,'mdiag'), L = D(1:end-1,2:end); R = D(2:end,1:end-1);end
        for i = -T : T
            R2 = R(L(:)==i);
            for j = -T : T
                f(i+T+1,j+T+1) = sum(R2(:)==j);
            end
        end
    case 3
        f = zeros(B,B,B);
        if strcmp(type,'hor'),   L = D(:,1:end-2); C = D(:,2:end-1); R = D(:,3:end);end
        if strcmp(type,'ver'),   L = D(1:end-2,:); C = D(2:end-1,:); R = D(3:end,:);end
        if strcmp(type,'diag'),  L = D(1:end-2,1:end-2); C = D(2:end-1,2:end-1); R = D(3:end,3:end);end
        if strcmp(type,'mdiag'), L = D(1:end-2,3:end); C = D(2:end-1,2:end-1); R = D(3:end,1:end-2);end
        for i = -T : T
            C2 = C(L(:)==i);
            R2 = R(L(:)==i);
            for j = -T : T
                R3 = R2(C2(:)==j);
                for k = -T : T
                    f(i+T+1,j+T+1,k+T+1) = sum(R3(:)==k);
                end
            end
        end
    case 4
        f = zeros(B,B,B,B);
        if strcmp(type,'hor'),    L = D(:,1:end-3); C = D(:,2:end-2); E = D(:,3:end-1); R = D(:,4:end);end
        if strcmp(type,'ver'),    L = D(1:end-3,:); C = D(2:end-2,:); E = D(3:end-1,:); R = D(4:end,:);end
        if strcmp(type,'diag'),   L = D(1:end-3,1:end-3); C = D(2:end-2,2:end-2); E = D(3:end-1,3:end-1); R = D(4:end,4:end);end
        if strcmp(type,'mdiag'),  L = D(4:end,1:end-3); C = D(3:end-1,2:end-2); E = D(2:end-2,3:end-1); R = D(1:end-3,4:end);end
        if strcmp(type,'square'), L = D(2:end,1:end-1); C = D(2:end,2:end); E = D(1:end-1,2:end); R = D(1:end-1,1:end-1);end
        if strcmp(type,'square-ori'), [M, N] = size(D); Dh = D(:,1:M); Dv = D(:,M+1:2*M);
                                  L = Dh(2:end,1:end-1); C = Dv(2:end,2:end); E = Dh(1:end-1,2:end); R = Dv(1:end-1,1:end-1);end
        if strcmp(type,'hvdm'),   [M, N] = size(D); L = D(:,1:M); C = D(:,M+1:2*M); E = D(:,2*M+1:3*M); R = D(:,3*M+1:4*M);end
        if strcmp(type,'s-in'),   [M, N] = size(D); Dh = D(:,1:M); Dv = D(:,M+1:2*M); Dh1 = D(:,2*M+1:3*M); Dv1 = D(:,3*M+1:4*M);
                                  L = Dh(2:end,1:end-1); C = Dh1(2:end,2:end); E = Dh1(1:end-1,2:end); R = Dh(1:end-1,1:end-1);end
        if strcmp(type,'s-out'),  [M, N] = size(D); Dh = D(:,1:M); Dv = D(:,M+1:2*M); Dh1 = D(:,2*M+1:3*M); Dv1 = D(:,3*M+1:4*M);
                                  L = Dh1(2:end,1:end-1); C = Dh(2:end,2:end); E = Dh(1:end-1,2:end); R = Dh1(1:end-1,1:end-1);end
        if strcmp(type,'ori-in'), [M, N] = size(D); Dh = D(:,1:M); Dv = D(:,M+1:2*M); Dh1 = D(:,2*M+1:3*M); Dv1 = D(:,3*M+1:4*M);
                                  L = Dh(2:end,1:end-1); C = Dv1(2:end,2:end); E = Dh1(1:end-1,2:end); R = Dv(1:end-1,1:end-1);end
        if strcmp(type,'ori-out'),[M, N] = size(D); Dh = D(:,1:M); Dv = D(:,M+1:2*M); Dh1 = D(:,2*M+1:3*M); Dv1 = D(:,3*M+1:4*M);
                                  L = Dh1(2:end,1:end-1); C = Dv(2:end,2:end); E = Dh(1:end-1,2:end); R = Dv1(1:end-1,1:end-1);end
        for i = -T : T
            ind = (L(:)==i); C2 = C(ind); E2 = E(ind); R2 = R(ind);
            for j = -T : T
                ind = (C2(:)==j); E3 = E2(ind); R3 = R2(ind);
                for k = -T : T
                    R4 = R3(E3(:)==k);
                    for l = -T : T
                        f(i+T+1,j+T+1,k+T+1,l+T+1) = sum(R4(:)==l);
                    end
                end
            end
        end
    case 5
        f = zeros(B,B,B,B,B);
        if strcmp(type,'hor'),L = D(:,1:end-4); C = D(:,2:end-3); E = D(:,3:end-2); F = D(:,4:end-1); R = D(:,5:end);end
        if strcmp(type,'ver'),L = D(1:end-4,:); C = D(2:end-3,:); E = D(3:end-2,:); F = D(4:end-1,:); R = D(5:end,:);end
        
        for i = -T : T
            ind = (L(:)==i); C2 = C(ind); E2 = E(ind); F2 = F(ind); R2 = R(ind);
            for j = -T : T
                ind = (C2(:)==j); E3 = E2(ind); F3 = F2(ind); R3 = R2(ind);
                for k = -T : T
                    ind = (E3(:)==k); F4 = F3(ind); R4 = R3(ind);
                    for l = -T : T
                        R5 = R4(F4(:)==l);
                        for m = -T : T
                            f(i+T+1,j+T+1,k+T+1,l+T+1,m+T+1) = sum(R5(:)==m);
                        end
                    end
                end
            end
        end
end

% normalization
f = double(f);
fsum = sum(f(:));
if fsum>0, f = f/fsum; end

function Y = Quant(X,q,T)
% Quantization routine
% X ... variable to be quantized/truncated
% T ... threshold
% q ... quantization step (with type = 'scalar') or a vector of increasing
% non-negative integers outlining the quantization process.
% Y ... quantized/truncated variable
% Example 0: when q is a positive scalar, Y = trunc(round(X/q),T)
% Example 1: q = [0 1 2 3] quantizes 0 to 0, 1 to 1, 2 to 2, [3,Inf) to 3,
% (-Inf,-3] to -3, -2 to -2, -1 to -1. It is equivalent to Quant(.,3,1).
% Example 2: q = [0 2 4 5] quantizes 0 to 0, {1,2} to 1, {3,4} to 2,
% [5,Inf) to 3, and similarly with the negatives.
% Example 3: q = [1 2] quantizes {-1,0,1} to 0, [2,Inf) to 1, (-Inf,-2] to -1.
% Example 4: q = [1 3 7 15 16] quantizes {-1,0,1} to 0, {2,3} to 1, {4,5,6,7}
% to 2, {8,9,10,11,12,13,14,15} to 3, [16,Inf) to 4, and similarly the
% negatives.

if numel(q) == 1
    if q > 0, Y = trunc(round(X/q),T);
    else fprintf('*** ERROR: Attempt to quantize with non-positive step. ***\n'),end
else
    q = round(q);   % Making sure the vector q is made of integers
    if min(q(2:end)-q(1:end-1)) <= 0
        fprintf('*** ERROR: quantization vector not strictly increasing. ***\n')
    end
    if min(q) < 0, fprintf('*** ERROR: Attempt to quantize with negative step. ***\n'),end
    
    T = q(end);   % The last value determines the truncation threshold
    v = zeros(1,2*T+1);   % value-substitution vector
    Y = trunc(X,T)+T+1;   % Truncated X and shifted to positive values
    if q(1) == 0
        v(T+1) = 0; z = 1; ind = T+2;
        for i = 2 : numel(q)
            v(ind:ind+q(i)-q(i-1)-1) = z;
            ind = ind+q(i)-q(i-1);
            z = z+1;
        end
        v(1:T) = -v(end:-1:T+2);
    else
        v(T+1-q(1):T+1+q(1)) = 0; z = 1; ind = T+2+q(1);
        for i = 2 : numel(q)
            v(ind:ind+q(i)-q(i-1)-1) = z;
            ind = ind+q(i)-q(i-1);
            z = z+1;
        end
        v(1:T-q(1)) = -v(end:-1:T+2+q(1));
    end
    Y = v(Y);   % The actual quantization :)
end
function Z = trunc(X,T)
% Truncation to [-T,T]
Z = X;
Z(Z > T)  =  T;
Z(Z < -T) = -T;
function fsym = symfea(f,T,order,type)
% Marginalization by sign and directional symmetry for a feature vector
% stored as one of our 2*(2T+1)^order-dimensional feature vectors. This
% routine should be used for features possiessing sign and directional
% symmetry, such as spam-like features or 3x3 features. It should NOT be
% used for features from MINMAX residuals. Use the alternative
% symfea_minmax for this purpose.
% The feature f is assumed to be a 2dim x database_size matrix of features
% stored as columns (e.g., hor+ver, diag+minor_diag), with dim =
% 2(2T+1)^order.

[dim,N] = size(f);
B = 2*T+1;
c = B^order;
ERR = 1;

if strcmp(type,'spam')
    if dim == 2*c
        switch order    % Reduced dimensionality for a B^order dimensional feature vector
            case 1, red = T + 1;
            case 2, red = (T + 1)^2;
            case 3, red = 1 + 3*T + 4*T^2 + 2*T^3;
            case 4, red = B^2 + 4*T^2*(T + 1)^2;
            case 5, red = 1/4*(B^2 + 1)*(B^3 + 1);
        end
        fsym = zeros(2*red,N);

        for i = 1 : N
            switch order
                case 1, cube = f(1:c,i);
                case 2, cube = reshape(f(1:c,i),[B B]);
                case 3, cube = reshape(f(1:c,i),[B B B]);
                case 4, cube = reshape(f(1:c,i),[B B B B]);
                case 5, cube = reshape(f(1:c,i),[B B B B B]);
            end
            % [size(symm_dir(cube,T,order)) red]
            fsym(1:red,i) = symm(cube,T,order);
            switch order
                case 1, cube = f(c+1:2*c,i);
                case 2, cube = reshape(f(c+1:2*c,i),[B B]);
                case 3, cube = reshape(f(c+1:2*c,i),[B B B]);
                case 4, cube = reshape(f(c+1:2*c,i),[B B B B]);
                case 5, cube = reshape(f(c+1:2*c,i),[B B B B B]);
            end
            fsym(red+1:2*red,i) = symm(cube,T,order);
        end
    else
        fsym = [];
        fprintf('*** ERROR: feature dimension is not 2x(2T+1)^order. ***\n')
    end
    ERR = 0;
end

if strcmp(type,'mnmx')
    if dim == 2*c
        switch order
            case 3, red = B^3 - T*B^2;          % Dim of the marginalized set is (2T+1)^3-T*(2T+1)^2
            case 4, red = B^4 - 2*T*(T+1)*B^2;  % Dim of the marginalized set is (2T+1)^4-2T*(T+1)*(2T+1)^2
        end
        fsym = zeros(red, N);
        for i = 1 : N
            switch order
                case 1, cube_min = f(1:c,i); cube_max = f(c+1:2*c,i);
                case 2, cube_min = reshape(f(1:c,i),[B B]); cube_max = reshape(f(c+1:2*c,i),[B B]); f_signsym = cube_min + cube_max(end:-1:1,end:-1:1);
                case 3, cube_min = reshape(f(1:c,i),[B B B]); cube_max = reshape(f(c+1:2*c,i),[B B B]);  f_signsym = cube_min + cube_max(end:-1:1,end:-1:1,end:-1:1);
                case 4, cube_min = reshape(f(1:c,i),[B B B B]); cube_max = reshape(f(c+1:2*c,i),[B B B B]);  f_signsym = cube_min + cube_max(end:-1:1,end:-1:1,end:-1:1,end:-1:1);
                case 5, cube_min = reshape(f(1:c,i),[B B B B B]); cube_max = reshape(f(c+1:2*c,i),[B B B B B]);  f_signsym = cube_min + cube_max(end:-1:1,end:-1:1,end:-1:1,end:-1:1,end:-1:1);
            end
            % f_signsym = cube_min + cube_max(end:-1:1,end:-1:1,end:-1:1);
            fsym(:,i) = symm_dir(f_signsym,T,order);
        end
    end
    ERR = 0;
end

if ERR == 1, fprintf('*** ERROR: Feature dimension and T, order incompatible. ***\n'), end
function As = symm_dir(A,T,order)
% Symmetry marginalization routine. The purpose is to reduce the feature
% dimensionality and make the features more populated.
% A is an array of features of size (2*T+1)^order, otherwise error is outputted.
%
% Directional marginalization pertains to the fact that the 
% differences d1, d2, d3, ... in a natural (both cover and stego) image
% are as likely to occur as ..., d3, d2, d1.

% Basically, we merge all pairs of bins (i,j,k, ...) and (..., k,j,i) 
% as long as they are two different bins. Thus, instead of dim =
% (2T+1)^order, we decrease the dim by 1/2*{# of non-symmetric bins}.
% For order = 3, the reduced dim is (2T+1)^order - T(2T+1)^(order-1),
% for order = 4, it is (2T+1)^4 - 2T(T+1)(2T+1)^2.

B = 2*T+1;
done = zeros(size(A));
switch order
    case 3, red = B^3 - T*B^2;          % Dim of the marginalized set is (2T+1)^3-T*(2T+1)^2
    case 4, red = B^4 - 2*T*(T+1)*B^2;  % Dim of the marginalized set is (2T+1)^4-2T*(T+1)*(2T+1)^2
    case 5, red = B^5 - 2*T*(T+1)*B^3;
end
As = zeros(red, 1);
m = 1;
        
switch order
    case 3
        for i = -T : T
            for j = -T : T
                for k = -T : T
                    if k ~= i   % Asymmetric bin
                        if done(i+T+1,j+T+1,k+T+1) == 0
                            As(m) = A(i+T+1,j+T+1,k+T+1) + A(k+T+1,j+T+1,i+T+1);   % Two mirror-bins are merged
                            done(i+T+1,j+T+1,k+T+1) = 1;
                            done(k+T+1,j+T+1,i+T+1) = 1;
                            m = m + 1;
                        end
                    else        % Symmetric bin is just copied
                        As(m) = A(i+T+1,j+T+1,k+T+1);
                        done(i+T+1,j+T+1,k+T+1) = 1;
                        m = m + 1;
                    end
                end
            end
        end
    case 4
        for i = -T : T
            for j = -T : T
                for k = -T : T
                    for n = -T : T
                        if (i ~= n) || (j ~= k)   % Asymmetric bin
                            if done(i+T+1,j+T+1,k+T+1,n+T+1) == 0
                                As(m) = A(i+T+1,j+T+1,k+T+1,n+T+1) + A(n+T+1,k+T+1,j+T+1,i+T+1);  % Two mirror-bins are merged
                                done(i+T+1,j+T+1,k+T+1,n+T+1) = 1;
                                done(n+T+1,k+T+1,j+T+1,i+T+1) = 1;
                                m = m + 1;
                            end
                        else                      % Symmetric bin is just copied
                            As(m) = A(i+T+1,j+T+1,k+T+1,n+T+1);
                            done(i+T+1,j+T+1,k+T+1,n+T+1) = 1;
                            m = m + 1;
                        end
                    end
                end
            end
        end
     case 5
        for i = -T : T
            for j = -T : T
                for k = -T : T
                    for l = -T : T
                        for n = -T : T
                            if (i ~= n) || (j ~= l)   % Asymmetric bin
                                if done(i+T+1,j+T+1,k+T+1,l+T+1,n+T+1) == 0
                                    As(m) = A(i+T+1,j+T+1,k+T+1,l+T+1,n+T+1) + A(n+T+1,l+T+1,k+T+1,j+T+1,i+T+1);  % Two mirror-bins are merged
                                    done(i+T+1,j+T+1,k+T+1,l+T+1,n+T+1) = 1;
                                    done(n+T+1,l+T+1,k+T+1,j+T+1,i+T+1) = 1;
                                    m = m + 1;
                                end
                            else                      % Symmetric bin is just copied
                                As(m) = A(i+T+1,j+T+1,k+T+1,l+T+1,n+T+1);
                                done(i+T+1,j+T+1,k+T+1,l+T+1,n+T+1) = 1;
                                m = m + 1;
                            end
                        end
                    end
                end
            end
        end
    otherwise
        fprintf('*** ERROR: Order not equal to 3 or 4 or 5! ***\n')
end
function fsym = symm1(f,T,order)
% Marginalization by sign and directional symmetry for a feature vector
% stored as a (2T+1)^order-dimensional array. The input feature f is 
% assumed to be a dim x database_size matrix of features stored as columns.

[dim,N] = size(f);
B = 2*T+1;
c = B^order;
ERR = 1;

if dim == c
    ERR = 0;
    switch order    % Reduced dimensionality for a c-dimensional feature vector
        case 1, red = T + 1;
        case 2, red = (T + 1)^2;
        case 3, red = 1 + 3*T + 4*T^2 + 2*T^3;
        case 4, red = B^2 + 4*T^2*(T + 1)^2;
        case 5, red = 1/4*(B^2 + 1)*(B^3 + 1);
    end
    fsym = zeros(red,N);

    for i = 1 : N
        switch order
            case 1, cube = f(:,i);
            case 2, cube = reshape(f(:,i),[B B]);
            case 3, cube = reshape(f(:,i),[B B B]);
            case 4, cube = reshape(f(:,i),[B B B B]);
            case 5, cube = reshape(f(:,i),[B B B B B]);
        end
        % [size(symm_dir(cube,T,order)) red]
        fsym(:,i) = symm(cube,T,order);
    end
end

if ERR == 1, fprintf('*** ERROR in symm1: Feature dimension and T, order incompatible. ***\n'), end
function As = symm(A,T,order)
% Symmetry marginalization routine. The purpose is to reduce the feature
% dimensionality and make the features more populated. It can be applied to
% 1D -- 5D co-occurrence matrices (order \in {1,2,3,4,5}) with sign and 
% directional symmetries (explained below). 
% A must be an array of (2*T+1)^order, otherwise error is outputted.
%
% Marginalization by symmetry pertains to the fact that, fundamentally,
% the differences between consecutive pixels in a natural image (both cover
% and stego) d1, d2, d3, ..., have the same probability of occurrence as the
% triple -d1, -d2, -d3, ...
%
% Directional marginalization pertains to the fact that the 
% differences d1, d2, d3, ... in a natural (cover and stego) image are as
% likely to occur as ..., d3, d2, d1.

ERR = 1;  % Flag denoting when size of A is incompatible with the input parameters T and order
m = 2;
B = 2*T + 1;

switch order
    case 1  % First-order coocs are only symmetrized
        if numel(A) == 2*T+1
           As(1) = A(T+1);  % The only non-marginalized bin is the origin 0
           As(2:T+1) = A(1:T) + A(T+2:end);
           As = As(:);
           ERR = 0;
        end
    case 2
        if numel(A) == (2*T+1)^2
            As = zeros((T+1)^2, 1);
            As(1) = A(T+1,T+1); % The only non-marginalized bin is the origin (0,0)
            for i = -T : T
                for j = -T : T
                    if (done(i+T+1,j+T+1) == 0) && (abs(i)+abs(j) ~= 0)
                        As(m) = A(i+T+1,j+T+1) + A(T+1-i,T+1-j);
                        done(i+T+1,j+T+1) = 1;
                        done(T+1-i,T+1-j) = 1;
                        if (i ~= j) && (done(j+T+1,i+T+1) == 0)
                            As(m) = As(m) + A(j+T+1,i+T+1) + A(T+1-j,T+1-i);
                            done(j+T+1,i+T+1) = 1;
                            done(T+1-j,T+1-i) = 1;
                        end
                        m = m + 1;
                    end
                end
            end
            ERR = 0;
        end
    case 3
        if numel(A) == B^3
            done = zeros(size(A));
            As = zeros(1+3*T+4*T^2+2*T^3, 1);
            As(1) = A(T+1,T+1,T+1); % The only non-marginalized bin is the origin (0,0,0)
            for i = -T : T
                for j = -T : T
                    for k = -T : T
                        if (done(i+T+1,j+T+1,k+T+1) == 0) && (abs(i)+abs(j)+abs(k) ~= 0)
                            As(m) = A(i+T+1,j+T+1,k+T+1) + A(T+1-i,T+1-j,T+1-k);
                            done(i+T+1,j+T+1,k+T+1) = 1;
                            done(T+1-i,T+1-j,T+1-k) = 1;
                            if (i ~= k) && (done(k+T+1,j+T+1,i+T+1) == 0)
                                As(m) = As(m) + A(k+T+1,j+T+1,i+T+1) + A(T+1-k,T+1-j,T+1-i);
                                done(k+T+1,j+T+1,i+T+1) = 1;
                                done(T+1-k,T+1-j,T+1-i) = 1;
                            end
                            m = m + 1;
                        end
                    end
                end
            end
            ERR = 0;
        end
    case 4
        if numel(A) == (2*T+1)^4
            done = zeros(size(A));
            As = zeros(B^2 + 4*T^2*(T+1)^2, 1);
            As(1) = A(T+1,T+1,T+1,T+1); % The only non-marginalized bin is the origin (0,0,0,0)
            for i = -T : T
                for j = -T : T
                    for k = -T : T
                        for n = -T : T
                            if (done(i+T+1,j+T+1,k+T+1,n+T+1) == 0) && (abs(i)+abs(j)+abs(k)+abs(n)~=0)
                                As(m) = A(i+T+1,j+T+1,k+T+1,n+T+1) + A(T+1-i,T+1-j,T+1-k,T+1-n);
                                done(i+T+1,j+T+1,k+T+1,n+T+1) = 1;
                                done(T+1-i,T+1-j,T+1-k,T+1-n) = 1;
                                if ((i ~= n) || (j ~= k)) && (done(n+T+1,k+T+1,j+T+1,i+T+1) == 0)
                                    As(m) = As(m) + A(n+T+1,k+T+1,j+T+1,i+T+1) + A(T+1-n,T+1-k,T+1-j,T+1-i);
                                    done(n+T+1,k+T+1,j+T+1,i+T+1) = 1;
                                    done(T+1-n,T+1-k,T+1-j,T+1-i) = 1;
                                end
                                m = m + 1;
                            end
                        end
                    end
                end
            end
            ERR = 0;
        end
    case 5
        if numel(A) == (2*T+1)^5
            done = zeros(size(A));
            As = zeros(1/4*(B^2 + 1)*(B^3 + 1), 1);
            As(1) = A(T+1,T+1,T+1,T+1,T+1); % The only non-marginalized bin is the origin (0,0,0,0,0)
            for i = -T : T
                for j = -T : T
                    for k = -T : T
                        for l = -T : T
                            for n = -T : T
                                if (done(i+T+1,j+T+1,k+T+1,l+T+1,n+T+1) == 0) && (abs(i)+abs(j)+abs(k)+abs(l)+abs(n)~=0)
                                    As(m) = A(i+T+1,j+T+1,k+T+1,l+T+1,n+T+1) + A(T+1-i,T+1-j,T+1-k,T+1-l,T+1-n);
                                    done(i+T+1,j+T+1,k+T+1,l+T+1,n+T+1) = 1;
                                    done(T+1-i,T+1-j,T+1-k,T+1-l,T+1-n) = 1;
                                    if ((i ~= n) || (j ~= l)) && (done(n+T+1,l+T+1,k+T+1,j+T+1,i+T+1) == 0)
                                        As(m) = As(m) + A(n+T+1,l+T+1,k+T+1,j+T+1,i+T+1) + A(T+1-n,T+1-l,T+1-k,T+1-j,T+1-i);
                                        done(n+T+1,l+T+1,k+T+1,j+T+1,i+T+1) = 1;
                                        done(T+1-n,T+1-l,T+1-k,T+1-j,T+1-i) = 1;
                                    end
                                    m = m + 1;
                                end
                            end
                        end
                    end
                end
            end
            ERR = 0;
        end
    otherwise
        As = [];
        fprintf('  Order of cooc is not in {1,2,3,4,5}.\n')
end

if ERR == 1
    As = [];
    fprintf('*** ERROR in symm: The number of elements in the array is not (2T+1)^order. ***\n')
end

 As = As(:);
function D = Residual(X,order,type)
% Computes the noise residual of a given type and order from MxN image X.
% residual order \in {1,2,3,4,5,6}
% type \in {hor,ver,diag,mdiag,KB,edge-h,edge-v,edge-d,edge-m}
% The resulting residual is an (M-b)x(N-b) array of the specified order,
% where b = ceil(order/2). This cropping is little more than it needs to 
% be to make sure all the residuals are easily "synchronized".
% !!!!!!!!!!!!! Use order = 2 with KB and all edge residuals !!!!!!!!!!!!!

[M N] = size(X);
I = 1+ceil(order/2) : M-ceil(order/2);
J = 1+ceil(order/2) : N-ceil(order/2);

switch type
    case 'hor'
        switch order
            case 1, D = - X(I,J) + X(I,J+1);
            case 2, D = X(I,J-1) - 2*X(I,J) + X(I,J+1);
            case 3, D = X(I,J-1) - 3*X(I,J) + 3*X(I,J+1) - X(I,J+2);
            case 4, D = -X(I,J-2) + 4*X(I,J-1) - 6*X(I,J) + 4*X(I,J+1) - X(I,J+2);
            case 5, D = -X(I,J-2) + 5*X(I,J-1) - 10*X(I,J) + 10*X(I,J+1) - 5*X(I,J+2) + X(I,J+3);
            case 6, D = X(I,J-3) - 6*X(I,J-2) + 15*X(I,J-1) - 20*X(I,J) + 15*X(I,J+1) - 6*X(I,J+2) + X(I,J+3);
        end
    case 'ver'
        switch order
            case 1, D = - X(I,J) + X(I+1,J);
            case 2, D = X(I-1,J) - 2*X(I,J) + X(I+1,J);
            case 3, D = X(I-1,J) - 3*X(I,J) + 3*X(I+1,J) - X(I+2,J);
            case 4, D = -X(I-2,J) + 4*X(I-1,J) - 6*X(I,J) + 4*X(I+1,J) - X(I+2,J);
            case 5, D = -X(I-2,J) + 5*X(I-1,J) - 10*X(I,J) + 10*X(I+1,J) - 5*X(I+2,J) + X(I+3,J);
            case 6, D = X(I-3,J) - 6*X(I-2,J) + 15*X(I-1,J) - 20*X(I,J) + 15*X(I+1,J) - 6*X(I+2,J) + X(I+3,J);
        end
    case 'diag'
        switch order
            case 1, D = - X(I,J) + X(I+1,J+1);
            case 2, D = X(I-1,J-1) - 2*X(I,J) + X(I+1,J+1);
            case 3, D = X(I-1,J-1) - 3*X(I,J) + 3*X(I+1,J+1) - X(I+2,J+2);
            case 4, D = -X(I-2,J-2) + 4*X(I-1,J-1) - 6*X(I,J) + 4*X(I+1,J+1) - X(I+2,J+2);
            case 5, D = -X(I-2,J-2) + 5*X(I-1,J-1) - 10*X(I,J) + 10*X(I+1,J+1) - 5*X(I+2,J+2) + X(I+3,J+3);
            case 6, D = X(I-3,J-3) - 6*X(I-2,J-2) + 15*X(I-1,J-1) - 20*X(I,J) + 15*X(I+1,J+1) - 6*X(I+2,J+2) + X(I+3,J+3);
        end
    case 'mdiag'
        switch order
            case 1, D = - X(I,J) + X(I-1,J+1);
            case 2, D = X(I-1,J+1) - 2*X(I,J) + X(I+1,J-1);
            case 3, D = X(I-1,J+1) - 3*X(I,J) + 3*X(I+1,J-1) - X(I+2,J-2);
            case 4, D = -X(I-2,J+2) + 4*X(I-1,J+1) - 6*X(I,J) + 4*X(I+1,J-1) - X(I+2,J-2);
            case 5, D = -X(I-2,J+2) + 5*X(I-1,J+1) - 10*X(I,J) + 10*X(I+1,J-1) - 5*X(I+2,J-2) + X(I+3,J-3);
            case 6, D = X(I-3,J+3) - 6*X(I-2,J+2) + 15*X(I-1,J+1) - 20*X(I,J) + 15*X(I+1,J-1) - 6*X(I+2,J-2) + X(I+3,J-3);
        end
    case 'KB'
        D = -X(I-1,J-1) + 2*X(I-1,J) - X(I-1,J+1) + 2*X(I,J-1) - 4*X(I,J) + 2*X(I,J+1) - X(I+1,J-1) + 2*X(I+1,J) - X(I+1,J+1);
    case 'edge-h'
        Du = 2*X(I-1,J) + 2*X(I,J-1) + 2*X(I,J+1) - X(I-1,J-1) - X(I-1,J+1) - 4*X(I,J);   %   -1  2 -1
        Db = 2*X(I+1,J) + 2*X(I,J-1) + 2*X(I,J+1) - X(I+1,J-1) - X(I+1,J+1) - 4*X(I,J);   %    2  C  2    +  flipped vertically
        D = [Du,Db];
    case 'edge-v'
        Dl = 2*X(I,J-1) + 2*X(I-1,J) + 2*X(I+1,J) - X(I-1,J-1) - X(I+1,J-1) - 4*X(I,J);   %   -1  2
        Dr = 2*X(I,J+1) + 2*X(I-1,J) + 2*X(I+1,J) - X(I-1,J+1) - X(I+1,J+1) - 4*X(I,J);   %    2  C       +  flipped horizontally
        D = [Dl,Dr];                                                                      %   -1  2
    case 'edge-m'
        Dlu = 2*X(I,J-1) + 2*X(I-1,J) - X(I-1,J-1) - X(I+1,J-1) - X(I-1,J+1) - X(I,J); %      -1  2 -1
        Drb = 2*X(I,J+1) + 2*X(I+1,J) - X(I+1,J+1) - X(I+1,J-1) - X(I-1,J+1) - X(I,J); %       2  C       +  flipped mdiag
        D = [Dlu,Drb];                                                                 %      -1
    case 'edge-d'
        Dru = 2*X(I-1,J) + 2*X(I,J+1) - X(I-1,J+1) - X(I-1,J-1) - X(I+1,J+1) - X(I,J); %      -1  2 -1
        Dlb = 2*X(I,J-1) + 2*X(I+1,J) - X(I+1,J-1) - X(I+1,J+1) - X(I-1,J-1) - X(I,J); %          C  2    +  flipped diag
        D = [Dru,Dlb];                                                                 %            -1
    case 'KV'
        D = 8*X(I-1,J) + 8*X(I+1,J) + 8*X(I,J-1) + 8*X(I,J+1);
        D = D - 6*X(I-1,J+1) - 6*X(I-1,J-1) - 6*X(I+1,J-1) - 6*X(I+1,J+1);
        D = D - 2*X(I-2,J) - 2*X(I+2,J) - 2*X(I,J+2) - 2*X(I,J-2);
        D = D + 2*X(I-1,J-2) + 2*X(I-2,J-1) + 2*X(I-2,J+1) + 2*X(I-1,J+2) + 2*X(I+1,J+2) + 2*X(I+2,J+1) + 2*X(I+2,J-1) + 2*X(I+1,J-2);
        D = D - X(I-2,J-2) - X(I-2,J+2) - X(I+2,J-2) - X(I+2,J+2) - 12*X(I,J);
end
