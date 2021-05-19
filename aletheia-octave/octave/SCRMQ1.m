function f = SCRMQ1(IMAGE)

part = 'all';
Tc = 3;

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
% SCRMQ1 is an extension of SRMQ1 for color images [2]. Three-dimensional 
% RGB color cooccurrences are built for all residual submodels (CRMQ1) and 
% appended to SRMQ1 to form SCRMQ1.
% -------------------------------------------------------------------------
% Extracts all 106 submodels presented in [1] as part of a rich model for
% steganalysis of digital images. All features are calculated in the
% spatial domain and are stored in a structured variable 'f'. For more
% deatils about the individual submodels, please see the publication [1].
% Total dimensionality of all 106 submodels is 34671.
% -------------------------------------------------------------------------
% Input:  IMAGE ... path to the image (can be decompressed JPEG, .ppm,..)
%         Tc ...... Truncation for quantized residuals that enter CRMQ1
%                   Truncation in SRMQ1 is fixed to T=2
%         part .... 'color' or anything other than 'all' -output only CRMQ1
%                   otherwise output all SCRMQ1 features
% Output: f ....... extracted SCRMQ1 features in a structured format,
%                   CRMQ1 consists of features with 'c' at the end of their
%                   names, making the extension of SRMQ1.
% EXAMPLES:
%  X = imread('image1.ppm');
%  SCRMQ1Features = SCRMQ1(double(X));          % Tc=3 is default
%  CRMQ1Features  = SCRMQ1(double(X),2,'color');
% -------------------------------------------------------------------------
% [1] Rich Models for Steganalysis of Digital Images, J. Fridrich and J.
% Kodovsky, IEEE Transactions on Information Forensics and Security, 2011.
% Under review.
% [2] Rich Model for Steganalysis of Color Images, M. Goljan, J. Fridrich,
% and Rémi Cogranne, Proc. IEEE WIFS, Atlanta, GA, December 3–5, 2014
% -------------------------------------------------------------------------
if ischar(IMAGE)
    X = double(imread(IMAGE));
else
    X = double(IMAGE);
end

f = [];
% if nargin<3,
%     part = 'all';
%elseif isempty(part),
%    part = 'all';
%end
%if nargin<2,
%    % SCRMQ1 for color image, T=3 fixed
%    ,Tc=3,      % truncation for between color cooccurrences
%end
f = post_processing(all1stc(X,1,Tc),'f1',1,f,Tc);   % 1st order, q=1
% f = post_processing(all1stc(X,2,Tc),'f1',2,f,Tc); % 1st order, q=2
for q=[1], f = post_processing(all2ndc(X,q*2,Tc),'f2',q,f,Tc); end    % 2nd order
for q=[1], f = post_processing(all3rdc(X,q*3,Tc),'f3',q,f,Tc); end    % 3rd order
for q=[1], f = post_processing(all3x3c(X,q*4,Tc),'f3x3',q,f,Tc); end  % 3x3
for q=[1], f = post_processing(all5x5c(X,q*12,Tc),'f5x5',q,f,Tc); end % 5x5

if strcmp(part,'all')       % do all submodels
% SRM or merged SRM for color image, T=2 fixed
f = post_processing(all1st(X,1),'f1',1,f);   % 1st order, q=1
% f = post_processing(all1st(X,2),'f1',2,f); % 1st order, q=2
for q=[1], f = post_processing(all2nd(X,q*2),'f2',q,f); end    % 2nd order
for q=[1], f = post_processing(all3rd(X,q*3),'f3',q,f); end    % 3rd order
for q=[1], f = post_processing(all3x3(X,q*4),'f3x3',q,f); end  % 3x3
for q=[1], f = post_processing(all5x5(X,q*12),'f5x5',q,f); end % 5x5
end                         % do all submodels

%**** FUNCTION ****%
function RESULT = post_processing(DATA,f,q,RESULT,Tc)

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
        if strcmp(N(end),'c'),       	% color minmax symmetrization, T=3
%             F = symfea([Fmin Fmax]',3,3,'mnmxc')'; % w/o rgb<->bgr symm
            F = symfea([Fmin Fmax]',Tc,3,'mnmx')';   % including rgb<->bgr symm
        else
            F = symfea([Fmin Fmax]',2,4,'mnmx')';   %#ok<*NASGU>
        end
        eval(['RESULT.' OUT ' = single(F);' ]);
    elseif strcmp(N(1:4),'spam')
        % spam symmetrization
        OUT = ['s' T(2:end) '_' N '_' Q];
        if isfield(RESULT,OUT), continue; end
        eval(['Fold = RESULT.' name ';']);
        if strcmp(N(end),'c'),     	   % color sign symmetrization, T=3
%             F = symm1sign(Fold',3,3)'; % w/o rgb<->bgr symm
            F = symm1(Fold',Tc,3)';    % including rgb<->bgr symm
        else
            F = symm1(Fold',2,4)';
        end
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
    elseif strcmp(N,'spam11')||strcmp(N,'spam11c')
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

%**** FUNCTION ****%
function [T,N,Q] = parse_feaname(name)
[T,N,Q] = deal('');
P = strfind(name,'_'); if length(P)~=2, return; end
T = name(1:P(1)-1);
N = name(P(1)+1:P(2)-1);
Q = name(P(2)+1:end);

%**** FUNCTION ****%
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
% This function calls Cooc1.m and Quant.m

[M N three] = size(X); [I,J,T,order] = deal(2:M-1,2:N-1,2,4);
% The following way is possible but would require recomputing of residuals in function form_3Dcooc_between_colors
% if three==3, 
%     g1 = all1st(X(:,:,1),q);
%     g2 = all1st(X(:,:,2),q);
%     g3 = all1st(X(:,:,3),q);
%     g = sumcells(g1,g2,g3);
%     form_3Dcooc_between_colors
%     return
% end

% Variable names are self-explanatory (R = right, U = up, L = left, D = down)
for j=1:three
 [R,L,U,D]  = deal(X(I,J+1,j)-X(I,J,j),X(I,J-1,j)-X(I,J,j),X(I-1,J,j)-X(I,J,j),X(I+1,J,j)-X(I,J,j)); 
 [Rq{j},Lq{j},Uq{j},Dq{j}] = deal(Quant(R,q,T),Quant(L,q,T),Quant(U,q,T),Quant(D,q,T)); 
 [RU{j},LU{j},RD{j},LD{j}] = deal(X(I-1,J+1,j)-X(I,J,j),X(I-1,J-1,j)-X(I,J,j),X(I+1,J+1,j)-X(I,J,j),X(I+1,J-1,j)-X(I,J,j));
 [RU{j},LU{j},RD{j},LD{j}] = deal(Quant(RU{j},q,T),Quant(LU{j},q,T),Quant(RD{j},q,T),Quant(LD{j},q,T));
end,       clear R L U D X
% minmax22h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical.
for j=1:three
 [RL_min{j},UD_min{j},RL_max{j},UD_max{j}] = deal(min(Rq{j},Lq{j}),min(Uq{j},Dq{j}),max(Rq{j},Lq{j}),max(Uq{j},Dq{j}));
 g.min22h{j} = Cooc1(RL_min{j},order,'hor',T) + Cooc1(UD_min{j},order,'ver',T);
 g.max22h{j} = Cooc1(RL_max{j},order,'hor',T) + Cooc1(UD_max{j},order,'ver',T);
end
g.min22h = sumcells(g.min22h);
g.max22h = sumcells(g.max22h);

% minmax34h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical
for j=1:three
 [Uq_min{j},Rq_min{j},Dq_min{j},Lq_min{j}] = deal(min(min(Lq{j},Uq{j}),Rq{j}),min(min(Uq{j},Rq{j}),Dq{j}),min(min(Rq{j},Dq{j}),Lq{j}),min(min(Dq{j},Lq{j}),Uq{j}));
 [Uq_max{j},Rq_max{j},Dq_max{j},Lq_max{j}] = deal(max(max(Lq{j},Uq{j}),Rq{j}),max(max(Uq{j},Rq{j}),Dq{j}),max(max(Rq{j},Dq{j}),Lq{j}),max(max(Dq{j},Lq{j}),Uq{j}));
 g.min34h{j} = Cooc1([Uq_min{j};Dq_min{j}],order,'hor',T) + Cooc1([Lq_min{j} Rq_min{j}],order,'ver',T);
 g.max34h{j} = Cooc1([Uq_max{j};Dq_max{j}],order,'hor',T) + Cooc1([Rq_max{j} Lq_max{j}],order,'ver',T);
end
g.min34h = sumcells(g.min34h);
g.max34h = sumcells(g.max34h);

% spam14h/v -- to be symmetrized as spam, directional, hv-nonsymmetrical
for j=1:three
 g.spam14h{j} = Cooc1(Rq{j},order,'hor',T) + Cooc1(Uq{j},order,'ver',T);
 g.spam14v{j} = Cooc1(Rq{j},order,'ver',T) + Cooc1(Uq{j},order,'hor',T);
end
g.spam14h = sumcells(g.spam14h);
g.spam14v = sumcells(g.spam14v);

% minmax22v -- to be symmetrized as mnmx, directional, hv-nonsymmetrical. Good with higher-order residuals! Note: 22h is bad (too much neighborhood overlap).
for j=1:three
 g.min22v{j} = Cooc1(RL_min{j},order,'ver',T) + Cooc1(UD_min{j},order,'hor',T);
 g.max22v{j} = Cooc1(RL_max{j},order,'ver',T) + Cooc1(UD_max{j},order,'hor',T);
end
g.min22v = sumcells(g.min22v);
g.max22v = sumcells(g.max22v);

% minmax24 -- to be symmetrized as mnmx, directional, hv-symmetrical. Darn good, too.
for j=1:three
 [RU_min{j},RD_min{j},LU_min{j},LD_min{j}] = deal(min(Rq{j},Uq{j}),min(Rq{j},Dq{j}),min(Lq{j},Uq{j}),min(Lq{j},Dq{j}));
 [RU_max{j},RD_max{j},LU_max{j},LD_max{j}] = deal(max(Rq{j},Uq{j}),max(Rq{j},Dq{j}),max(Lq{j},Uq{j}),max(Lq{j},Dq{j}));
 g.min24{j} = Cooc1([RU_min{j};RD_min{j};LU_min{j};LD_min{j}],order,'hor',T) + Cooc1([RU_min{j} RD_min{j} LU_min{j} LD_min{j}],order,'ver',T);
 g.max24{j} = Cooc1([RU_max{j};RD_max{j};LU_max{j};LD_max{j}],order,'hor',T) + Cooc1([RU_max{j} RD_max{j} LU_max{j} LD_max{j}],order,'ver',T);
end,    clear Rq Uq Dq Lq 
g.min24 = sumcells(g.min24);
g.max24 = sumcells(g.max24);

% minmax34v -- v works well, h does not, to be symmetrized as mnmx, directional, hv-nonsymmetrical
for j=1:three
 g.min34v{j} = Cooc1([Uq_min{j} Dq_min{j}],order,'ver',T) + Cooc1([Rq_min{j};Lq_min{j}],order,'hor',T);
 g.max34v{j} = Cooc1([Uq_max{j} Dq_max{j}],order,'ver',T) + Cooc1([Rq_max{j};Lq_max{j}],order,'hor',T);
end   
g.min34v = sumcells(g.min34v);
g.max34v = sumcells(g.max34v);
    clear Uq_min Dq_min Rq_min Lq_min 
    clear Uq_max Dq_max Rq_max Lq_max
 
% minmax41 -- to be symmetrized as mnmx, non-directional, hv-symmetrical
for j=1:three
 [R_min{j},R_max{j}] = deal(min(RL_min{j},UD_min{j}),max(RL_max{j},UD_max{j}));  
 g.min41{j} = Cooc1(R_min{j},order,'hor',T) + Cooc1(R_min{j},order,'ver',T);
 g.max41{j} = Cooc1(R_max{j},order,'hor',T) + Cooc1(R_max{j},order,'ver',T);
end,   	clear RL_min UD_min RL_max UD_max
g.min41 = sumcells(g.min41);
g.max41 = sumcells(g.max41);
    clear R_min R_max

% minmax34 -- good, to be symmetrized as mnmx, directional, hv-symmetrical
for j=1:three
 [RU_min{j},RD_min{j},LU_min{j},LD_min{j}] = deal(min(RU_min{j},RU{j}),min(RD_min{j},RD{j}),min(LU_min{j},LU{j}),min(LD_min{j},LD{j}));
 [RU_max{j},RD_max{j},LU_max{j},LD_max{j}] = deal(max(RU_max{j},RU{j}),max(RD_max{j},RD{j}),max(LU_max{j},LU{j}),max(LD_max{j},LD{j}));
 g.min34{j} = Cooc1([RU_min{j};RD_min{j};LU_min{j};LD_min{j}],order,'hor',T) + Cooc1([RU_min{j} RD_min{j} LU_min{j} LD_min{j}],order,'ver',T);
 g.max34{j} = Cooc1([RU_max{j};RD_max{j};LU_max{j};LD_max{j}],order,'hor',T) + Cooc1([RU_max{j} RD_max{j} LU_max{j} LD_max{j}],order,'ver',T);
end
g.min34 = sumcells(g.min34);
g.max34 = sumcells(g.max34);

 % minmax48h -- h better than v, to be symmetrized as mnmx, directional, hv-nonsymmetrical. 48v is almost as good as 48h; for 3rd-order but weaker for 1st-order. Here, I am outputting both but Figure 1 in our paper lists only 48h.
for j=1:three
 [RU_min2{j},RD_min2{j},LD_min2{j},LU_min2{j}] = deal(min(RU_min{j},LU{j}),min(RD_min{j},RU{j}),min(LD_min{j},RD{j}),min(LU_min{j},LD{j}));
 [RU_min3{j},RD_min3{j},LD_min3{j},LU_min3{j}] = deal(min(RU_min{j},RD{j}),min(RD_min{j},LD{j}),min(LD_min{j},LU{j}),min(LU_min{j},RU{j}));
 g.min48h{j} = Cooc1([RU_min2{j};LD_min2{j};RD_min3{j};LU_min3{j}],order,'hor',T) + Cooc1([RD_min2{j} LU_min2{j} RU_min3{j} LD_min3{j}],order,'ver',T);
 g.min48v{j} = Cooc1([RD_min2{j};LU_min2{j};RU_min3{j};LD_min3{j}],order,'hor',T) + Cooc1([RU_min2{j} LD_min2{j} RD_min3{j} LU_min3{j}],order,'ver',T);
 [RU_max2{j},RD_max2{j},LD_max2{j},LU_max2{j}] = deal(max(RU_max{j},LU{j}),max(RD_max{j},RU{j}),max(LD_max{j},RD{j}),max(LU_max{j},LD{j}));
 [RU_max3{j},RD_max3{j},LD_max3{j},LU_max3{j}] = deal(max(RU_max{j},RD{j}),max(RD_max{j},LD{j}),max(LD_max{j},LU{j}),max(LU_max{j},RU{j}));
 g.max48h{j} = Cooc1([RU_max2{j};LD_max2{j};RD_max3{j};LU_max3{j}],order,'hor',T) + Cooc1([RD_max2{j} LU_max2{j} RU_max3{j} LD_max3{j}],order,'ver',T);
 g.max48v{j} = Cooc1([RD_max2{j};LU_max2{j};RU_max3{j};LD_max3{j}],order,'hor',T) + Cooc1([RU_max2{j} LD_max2{j} RD_max3{j} LU_max3{j}],order,'ver',T);
end,    clear RU_min RD_min LU_min LD_min RU_max RD_max LU_max LD_max
g.min48h = sumcells(g.min48h);
g.min48v = sumcells(g.min48v);
g.max48h = sumcells(g.max48h);
g.max48v = sumcells(g.max48v);

 % minmax54 -- to be symmetrized as mnmx, directional, hv-symmetrical
for j=1:three
 [RU_min4{j},RD_min4{j},LD_min4{j},LU_min4{j}] = deal(min(RU_min2{j},RD{j}),min(RD_min2{j},LD{j}),min(LD_min2{j},LU{j}),min(LU_min2{j},RU{j}));
 [RU_min5{j},RD_min5{j},LD_min5{j},LU_min5{j}] = deal(min(RU_min3{j},LU{j}),min(RD_min3{j},RU{j}),min(LD_min3{j},RD{j}),min(LU_min3{j},LD{j}));
 g.min54{j} = Cooc1([RU_min4{j};LD_min4{j};RD_min5{j};LU_min5{j}],order,'hor',T) + Cooc1([RD_min4{j} LU_min4{j} RU_min5{j} LD_min5{j}],order,'ver',T);  
 [RU_max4{j},RD_max4{j},LD_max4{j},LU_max4{j}] = deal(max(RU_max2{j},RD{j}),max(RD_max2{j},LD{j}),max(LD_max2{j},LU{j}),max(LU_max2{j},RU{j}));
 [RU_max5{j},RD_max5{j},LD_max5{j},LU_max5{j}] = deal(max(RU_max3{j},LU{j}),max(RD_max3{j},RU{j}),max(LD_max3{j},RD{j}),max(LU_max3{j},LD{j}));
 g.max54{j} = Cooc1([RU_max4{j};LD_max4{j};RD_max5{j};LU_max5{j}],order,'hor',T) + Cooc1([RD_max4{j} LU_max4{j} RU_max5{j} LD_max5{j}],order,'ver',T);
end,    clear RU_min2 RD_min2 LD_min2 LU_min2 RU_min3 RD_min3 LD_min3 LU_min3 
        clear RU_max2 RD_max2 LD_max2 LU_max2 RU_max3 RD_max3 LD_max3 LU_max3 
g.min54 = sumcells(g.min54);
g.max54 = sumcells(g.max54);
 
%**** FUNCTION ****%
function g = all1stc(X,q,T)
%
% X must be a matrix of doubles or singles (the color image) and q is the 
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
% This function calls Cooc1.m and Quant.m

[M N three] = size(X); [I,J,order] = deal(2:M-1,2:N-1,4);
% The following way is possible but would require recomputing of residuals in function form_3Dcooc_between_colors
% if three==3, 
%     g1 = all1st(X(:,:,1),q);
%     g2 = all1st(X(:,:,2),q);
%     g3 = all1st(X(:,:,3),q);
%     g = sumcells(g1,g2,g3);
%     form_3Dcooc_between_colors
%     return
% end

% Variable names are self-explanatory (R = right, U = up, L = left, D = down)
for j=1:three
 [R,L,U,D]  = deal(X(I,J+1,j)-X(I,J,j),X(I,J-1,j)-X(I,J,j),X(I-1,J,j)-X(I,J,j),X(I+1,J,j)-X(I,J,j)); 
 [Rq{j},Lq{j},Uq{j},Dq{j}] = deal(Quant(R,q,T),Quant(L,q,T),Quant(U,q,T),Quant(D,q,T)); 
 [RU{j},LU{j},RD{j},LD{j}] = deal(X(I-1,J+1,j)-X(I,J,j),X(I-1,J-1,j)-X(I,J,j),X(I+1,J+1,j)-X(I,J,j),X(I+1,J-1,j)-X(I,J,j));
 [RU{j},LU{j},RD{j},LD{j}] = deal(Quant(RU{j},q,T),Quant(LU{j},q,T),Quant(RD{j},q,T),Quant(LD{j},q,T));
end,       clear R L U D X
% minmax22h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical.
for j=1:three
 [RL_min{j},UD_min{j},RL_max{j},UD_max{j}] = deal(min(Rq{j},Lq{j}),min(Uq{j},Dq{j}),max(Rq{j},Lq{j}),max(Uq{j},Dq{j}));
end
 g.min22c = Cooc1(RL_min,3,'col',T) + Cooc1(UD_min,3,'col',T);
 g.max22c = Cooc1(RL_max,3,'col',T) + Cooc1(UD_max,3,'col',T);

% spam14h/v -- to be symmetrized as spam, directional, hv-nonsymmetrical
 g.spam14c = Cooc1(Rq,3,'col',T) + Cooc1(Uq,3,'col',T);

% minmax24 -- to be symmetrized as mnmx, directional, hv-symmetrical. Darn good, too.
for j=1:three
 [RU_min{j},RD_min{j},LU_min{j},LD_min{j}] = deal(min(Rq{j},Uq{j}),min(Rq{j},Dq{j}),min(Lq{j},Uq{j}),min(Lq{j},Dq{j}));
 [RU_max{j},RD_max{j},LU_max{j},LD_max{j}] = deal(max(Rq{j},Uq{j}),max(Rq{j},Dq{j}),max(Lq{j},Uq{j}),max(Lq{j},Dq{j}));
end
 g.min24c = Cooc1(vercatcell(vercatcell(RU_min,RD_min),vercatcell(LU_min,LD_min)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RU_min,RD_min),horcatcell(LU_min,LD_min)),3,'col',T);
 g.max24c = Cooc1(vercatcell(vercatcell(RU_max,RD_max),vercatcell(LU_max,LD_max)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RU_max,RD_max),horcatcell(LU_max,LD_max)),3,'col',T);

% minmax34h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical
for j=1:three
 [Uq_min{j},Rq_min{j},Dq_min{j},Lq_min{j}] = deal(min(min(Lq{j},Uq{j}),Rq{j}),min(min(Uq{j},Rq{j}),Dq{j}),min(min(Rq{j},Dq{j}),Lq{j}),min(min(Dq{j},Lq{j}),Uq{j}));
 [Uq_max{j},Rq_max{j},Dq_max{j},Lq_max{j}] = deal(max(max(Lq{j},Uq{j}),Rq{j}),max(max(Uq{j},Rq{j}),Dq{j}),max(max(Rq{j},Dq{j}),Lq{j}),max(max(Dq{j},Lq{j}),Uq{j}));
end,    clear Rq Uq Dq Lq 
%  g.min34hc = Cooc1(vercatcell(Uq_min,Dq_min),3,'col',T) + Cooc1(horcatcell(Lq_min,Rq_min),3,'col',T);   % the same as min34vc,  becomes min34hvc
%  g.max34hc = Cooc1(vercatcell(Uq_max,Dq_max),3,'col',T) + Cooc1(horcatcell(Rq_max,Lq_max),3,'col',T);   % the same as max34vc,  becomes max34hvc

% minmax34v -- v works well, h does not, to be symmetrized as mnmx, directional, hv-nonsymmetrical
 g.min34hvc = Cooc1(horcatcell(Uq_min,Dq_min),3,'col',T) + Cooc1(vercatcell(Rq_min,Lq_min),3,'col',T);
    clear Uq_min Dq_min Rq_min Lq_min 
 g.max34hvc = Cooc1(horcatcell(Uq_max,Dq_max),3,'col',T) + Cooc1(vercatcell(Rq_max,Lq_max),3,'col',T);
    clear Uq_max Dq_max Rq_max Lq_max
 
% minmax41 -- to be symmetrized as mnmx, non-directional, hv-symmetrical
for j=1:three
 [R_min{j},R_max{j}] = deal(min(RL_min{j},UD_min{j}),max(RL_max{j},UD_max{j}));  
end,   	clear RL_min UD_min RL_max UD_max
 g.min41c = Cooc1(R_min,3,'col',T);      clear R_min
 g.max41c = Cooc1(R_max,3,'col',T);      clear R_max

% minmax34 -- good, to be symmetrized as mnmx, directional, hv-symmetrical
for j=1:three
 [RU_min{j},RD_min{j},LU_min{j},LD_min{j}] = deal(min(RU_min{j},RU{j}),min(RD_min{j},RD{j}),min(LU_min{j},LU{j}),min(LD_min{j},LD{j}));
 [RU_max{j},RD_max{j},LU_max{j},LD_max{j}] = deal(max(RU_max{j},RU{j}),max(RD_max{j},RD{j}),max(LU_max{j},LU{j}),max(LD_max{j},LD{j}));
end
 g.min34c = Cooc1(vercatcell(vercatcell(RU_min,RD_min),vercatcell(LU_min,LD_min)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RU_min,RD_min),horcatcell(LU_min,LD_min)),3,'col',T);
 g.max34c = Cooc1(vercatcell(vercatcell(RU_max,RD_max),vercatcell(LU_max,LD_max)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RU_max,RD_max),horcatcell(LU_max,LD_max)),3,'col',T);

 % minmax48h -- h better than v, to be symmetrized as mnmx, directional, hv-nonsymmetrical. 48v is almost as good as 48h; for 3rd-order but weaker for 1st-order. Here, I am outputting both but Figure 1 in our paper lists only 48h.
for j=1:three
 [RU_min2{j},RD_min2{j},LD_min2{j},LU_min2{j}] = deal(min(RU_min{j},LU{j}),min(RD_min{j},RU{j}),min(LD_min{j},RD{j}),min(LU_min{j},LD{j}));
 [RU_min3{j},RD_min3{j},LD_min3{j},LU_min3{j}] = deal(min(RU_min{j},RD{j}),min(RD_min{j},LD{j}),min(LD_min{j},LU{j}),min(LU_min{j},RU{j}));
 [RU_max2{j},RD_max2{j},LD_max2{j},LU_max2{j}] = deal(max(RU_max{j},LU{j}),max(RD_max{j},RU{j}),max(LD_max{j},RD{j}),max(LU_max{j},LD{j}));
 [RU_max3{j},RD_max3{j},LD_max3{j},LU_max3{j}] = deal(max(RU_max{j},RD{j}),max(RD_max{j},LD{j}),max(LD_max{j},LU{j}),max(LU_max{j},RU{j}));
end,    clear RU_min RD_min LU_min LD_min RU_max RD_max LU_max LD_max
 g.min48c = Cooc1(vercatcell(vercatcell(RU_min2,LD_min2),vercatcell(RD_min3,LU_min3)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RD_min2,LU_min2),horcatcell(RU_min3,LD_min3)),3,'col',T);
 g.max48c = Cooc1(vercatcell(vercatcell(RU_max2,LD_max2),vercatcell(RD_max3,LU_max3)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RD_max2,LU_max2),horcatcell(RU_max3,LD_max3)),3,'col',T);

 % minmax54 -- to be symmetrized as mnmx, directional, hv-symmetrical
for j=1:three
 [RU_min4{j},RD_min4{j},LD_min4{j},LU_min4{j}] = deal(min(RU_min2{j},RD{j}),min(RD_min2{j},LD{j}),min(LD_min2{j},LU{j}),min(LU_min2{j},RU{j}));
 [RU_min5{j},RD_min5{j},LD_min5{j},LU_min5{j}] = deal(min(RU_min3{j},LU{j}),min(RD_min3{j},RU{j}),min(LD_min3{j},RD{j}),min(LU_min3{j},LD{j}));
 [RU_max4{j},RD_max4{j},LD_max4{j},LU_max4{j}] = deal(max(RU_max2{j},RD{j}),max(RD_max2{j},LD{j}),max(LD_max2{j},LU{j}),max(LU_max2{j},RU{j}));
 [RU_max5{j},RD_max5{j},LD_max5{j},LU_max5{j}] = deal(max(RU_max3{j},LU{j}),max(RD_max3{j},RU{j}),max(LD_max3{j},RD{j}),max(LU_max3{j},LD{j}));
end,    clear RU_min2 RD_min2 LD_min2 LU_min2 RU_min3 RD_min3 LD_min3 LU_min3 
        clear RU_max2 RD_max2 LD_max2 LU_max2 RU_max3 RD_max3 LD_max3 LU_max3 
 g.min54c = Cooc1(vercatcell(vercatcell(RU_min4,LD_min4),vercatcell(RD_min5,LU_min5)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RD_min4,LU_min4),horcatcell(RU_min5,LD_min5)),3,'col',T);
 g.max54c = Cooc1(vercatcell(vercatcell(RU_max4,LD_max4),vercatcell(RD_max5,LU_max5)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RD_max4,LU_max4),horcatcell(RU_max5,LD_max5)),3,'col',T);
 
%**** FUNCTION ****%
function z = sumcells(cell_of_arrays)
if numel(cell_of_arrays)~=3, error('3 cells are required here'), end
if iscell(cell_of_arrays)
    z = cell_of_arrays{1};
    for j=2:numel(cell_of_arrays)
        if any(size(cell_of_arrays{j})~=size(z)), error('All cells to be merged must be same size.'), end
        z = z + cell_of_arrays{j};
    end
else   	% no merging needed
    z = cell_of_arrays;
end

%**** FUNCTION ****%
function C = vercatcell(A,B)
if iscell(A) & iscell(B)
    C{1} = [A{1};B{1}];
    for j=2:numel(A)
        C{j} = [A{j};B{j}];
    end
end

%**** FUNCTION ****%
function C = horcatcell(A,B)
if iscell(A) & iscell(B)
    C{1} = [A{1},B{1}];
    for j=2:numel(A)
        C{j} = [A{j},B{j}];
    end
end

%**** FUNCTION ****%
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
% This function calls Residual.m, Cooc1.m, and Quant.m

[M N three] = size(X); [T,order] = deal(2,4);
% 2nd-order residuals are implemented using Residual.m
for j=1:three
 [Dh{j},Dv{j},Dd{j},Dm{j}] = deal(Residual(X(:,:,j),2,'hor'),Residual(X(:,:,j),2,'ver'),Residual(X(:,:,j),2,'diag'),Residual(X(:,:,j),2,'mdiag'));
 [Yh{j},Yv{j},Yd{j},Ym{j}] = deal(Quant(Dh{j},q,T),Quant(Dv{j},q,T),Quant(Dd{j},q,T),Quant(Dm{j},q,T));
end,    clear Dh Dv Dd Dm X
% spam12h/v
for j=1:three
 g.spam12h{j} = Cooc1(Yh{j},order,'hor',T) + Cooc1(Yv{j},order,'ver',T);
 g.spam12v{j} = Cooc1(Yh{j},order,'ver',T) + Cooc1(Yv{j},order,'hor',T);
end
g.spam12h = sumcells(g.spam12h);
g.spam12v = sumcells(g.spam12v);

% minmax21
for j=1:three
 [Dmin{j},Dmax{j}] = deal(min(Yh{j},Yv{j}),max(Yh{j},Yv{j}));
 g.min21{j} = Cooc1(Dmin{j},order,'hor',T) + Cooc1(Dmin{j},order,'ver',T);
 g.max21{j} = Cooc1(Dmax{j},order,'hor',T) + Cooc1(Dmax{j},order,'ver',T);
end
g.min21 = sumcells(g.min21);
g.max21 = sumcells(g.max21);

% minmax41   
for j=1:three
 [Dmin2{j},Dmax2{j}] = deal(min(Dmin{j},min(Yd{j},Ym{j})),max(Dmax{j},max(Yd{j},Ym{j})));
 g.min41{j} = Cooc1(Dmin2{j},order,'hor',T) + Cooc1(Dmin2{j},order,'ver',T);
 g.max41{j} = Cooc1(Dmax2{j},order,'hor',T) + Cooc1(Dmax2{j},order,'ver',T);
end
g.min41 = sumcells(g.min41);
g.max41 = sumcells(g.max41);
   	clear Dmin2 Dmax2

 % minmax32 -- good, directional, hv-symmetrical, to be symmetrized as mnmx
for j=1:three
 [RU_min{j},RD_min{j}] = deal(min(Dmin{j},Ym{j}),min(Dmin{j},Yd{j})); 
 [RU_max{j},RD_max{j}] = deal(max(Dmax{j},Ym{j}),max(Dmax{j},Yd{j})); 
 g.min32{j} = Cooc1([RU_min{j};RD_min{j}],order,'hor',T) + Cooc1([RU_min{j} RD_min{j}],order,'ver',T);
 g.max32{j} = Cooc1([RU_max{j};RD_max{j}],order,'hor',T) + Cooc1([RU_max{j} RD_max{j}],order,'ver',T);
end,    clear Dmin Dmax
g.min32 = sumcells(g.min32);
g.max32 = sumcells(g.max32);

 % minmax24h,v -- both "not bad," h slightly better, directional, hv-nonsymmetrical, to be symmetrized as mnmx
for j=1:three
 [RU_min2{j},RD_min2{j},RU_min3{j},LU_min3{j}] = deal(min(Ym{j},Yh{j}),min(Yd{j},Yh{j}),min(Ym{j},Yv{j}),min(Yd{j},Yv{j}));
 g.min24h{j} = Cooc1([RU_min2{j};RD_min2{j}],order,'hor',T)+Cooc1([RU_min3{j} LU_min3{j}],order,'ver',T);
 g.min24v{j} = Cooc1([RU_min2{j} RD_min2{j}],order,'ver',T)+Cooc1([RU_min3{j};LU_min3{j}],order,'hor',T);
 [RU_max2{j},RD_max2{j},RU_max3{j},LU_max3{j}] = deal(max(Ym{j},Yh{j}),max(Yd{j},Yh{j}),max(Ym{j},Yv{j}),max(Yd{j},Yv{j}));
 g.max24h{j} = Cooc1([RU_max2{j};RD_max2{j}],order,'hor',T) + Cooc1([RU_max3{j} LU_max3{j}],order,'ver',T);
 g.max24v{j} = Cooc1([RU_max2{j} RD_max2{j}],order,'ver',T) + Cooc1([RU_max3{j};LU_max3{j}],order,'hor',T);
end
g.min24h = sumcells(g.min24h);
g.min24v = sumcells(g.min24v);
g.max24h = sumcells(g.max24h);
g.max24v = sumcells(g.max24v);

%**** FUNCTION ****%
function g = all2ndc(X,q,T)
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
% This function calls Residual.m, Cooc1.m, and Quant.m

[M N three] = size(X); order = 4;
% 2nd-order residuals are implemented using Residual.m
for j=1:three
 [Dh{j},Dv{j},Dd{j},Dm{j}] = deal(Residual(X(:,:,j),2,'hor'),Residual(X(:,:,j),2,'ver'),Residual(X(:,:,j),2,'diag'),Residual(X(:,:,j),2,'mdiag'));
 [Yh{j},Yv{j},Yd{j},Ym{j}] = deal(Quant(Dh{j},q,T),Quant(Dv{j},q,T),Quant(Dd{j},q,T),Quant(Dm{j},q,T));
end,    clear Dh Dv Dd Dm X
% spam12h/v
 g.spam12c = Cooc1(Yh,3,'col',T) + Cooc1(Yv,3,'col',T);

% minmax21
for j=1:three
 [Dmin{j},Dmax{j}] = deal(min(Yh{j},Yv{j}),max(Yh{j},Yv{j}));
end
 g.min21c = Cooc1(Dmin,3,'col',T);
 g.max21c = Cooc1(Dmax,3,'col',T);

% minmax41   
for j=1:three
 [Dmin2{j},Dmax2{j}] = deal(min(Dmin{j},min(Yd{j},Ym{j})),max(Dmax{j},max(Yd{j},Ym{j})));
end
 g.min41c = Cooc1(Dmin2,3,'col',T);      clear Dmin2
 g.max41c = Cooc1(Dmax2,3,'col',T);      clear Dmax2

 % minmax32 -- good, directional, hv-symmetrical, to be symmetrized as mnmx
for j=1:three
 [RU_min{j},RD_min{j}] = deal(min(Dmin{j},Ym{j}),min(Dmin{j},Yd{j})); 
 [RU_max{j},RD_max{j}] = deal(max(Dmax{j},Ym{j}),max(Dmax{j},Yd{j})); 
end,    clear Dmin Dmax
 g.min32c = Cooc1(vercatcell(RU_min,RD_min),3,'col',T);
 g.max32c = Cooc1(vercatcell(RU_max,RD_max),3,'col',T);

 % minmax24h,v -- both "not bad," h slightly better, directional, hv-nonsymmetrical, to be symmetrized as mnmx
for j=1:three
 [RU_min2{j},RD_min2{j},RU_min3{j},LU_min3{j}] = deal(min(Ym{j},Yh{j}),min(Yd{j},Yh{j}),min(Ym{j},Yv{j}),min(Yd{j},Yv{j}));
 [RU_max2{j},RD_max2{j},RU_max3{j},LU_max3{j}] = deal(max(Ym{j},Yh{j}),max(Yd{j},Yh{j}),max(Ym{j},Yv{j}),max(Yd{j},Yv{j}));
end
 g.min24c = Cooc1(vercatcell(RU_min2,RD_min2),3,'col',T) + Cooc1(horcatcell(RU_min3,LU_min3),3,'col',T);
 g.max24c = Cooc1(vercatcell(RU_max2,RD_max2),3,'col',T) + Cooc1(horcatcell(RU_max3,LU_max3),3,'col',T);

%**** FUNCTION ****%
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

[M N three] = size(X);  [I,J,T,order] = deal(3:M-2,3:N-2,2,4);
for j=1:three
 [R{j},L{j},U{j},D{j}] = deal(-X(I,J+2,j)+3*X(I,J+1,j)-3*X(I,J,j)+X(I,J-1,j),-X(I,J-2,j)+3*X(I,J-1,j)-3*X(I,J,j)+...
    X(I,J+1,j),-X(I-2,J,j)+3*X(I-1,J,j)-3*X(I,J,j)+X(I+1,J,j),-X(I+2,J,j)+3*X(I+1,J,j)-3*X(I,J,j)+X(I-1,J,j));
 [Rq{j},Lq{j},Uq{j},Dq{j}] = deal(Quant(R{j},q,T),Quant(L{j},q,T),Quant(U{j},q,T),Quant(D{j},q,T));          
 [RU{j},LU{j},RD{j},LD{j}] = deal(-X(I-2,J+2,j)+3*X(I-1,J+1,j)-3*X(I,J,j)+X(I+1,J-1,j),-X(I-2,J-2,j)+3*X(I-1,J-1,j)-3*X(I,J,j)+...
    X(I+1,J+1,j),-X(I+2,J+2,j)+3*X(I+1,J+1,j)-3*X(I,J,j)+X(I-1,J-1,j),-X(I+2,J-2,j)+3*X(I+1,J-1,j)-3*X(I,J,j)+X(I-1,J+1,j));
 [RU{j},LU{j},RD{j},LD{j}] = deal(Quant(RU{j},q,T),Quant(LU{j},q,T),Quant(RD{j},q,T),Quant(LD{j},q,T));
end,    clear R L U D X
% minmax22h,v -- to be symmetrized as mnmx, directional, hv-nonsymmetrical. Good with higher-order residuals! Note: 22h is bad (too much neighborhood overlap).
for j=1:three
 [RL_min{j},UD_min{j}] = deal(min(Rq{j},Lq{j}),min(Uq{j},Dq{j}));
 [RL_max{j},UD_max{j}] = deal(max(Rq{j},Lq{j}),max(Uq{j},Dq{j}));
 g.min22h{j} = Cooc1(RL_min{j},order,'hor',T) + Cooc1(UD_min{j},order,'ver',T);
 g.max22h{j} = Cooc1(RL_max{j},order,'hor',T) + Cooc1(UD_max{j},order,'ver',T);
 g.min22v{j} = Cooc1(RL_min{j},order,'ver',T) + Cooc1(UD_min{j},order,'hor',T);
 g.max22v{j} = Cooc1(RL_max{j},order,'ver',T) + Cooc1(UD_max{j},order,'hor',T);
end
g.min22h = sumcells(g.min22h);
g.max22h = sumcells(g.max22h);
g.min22v = sumcells(g.min22v);
g.max22v = sumcells(g.max22v);

% spam14h,v -- to be symmetrized as spam, directional, hv-nonsymmetrical
for j=1:three
 g.spam14h{j} = Cooc1(Rq{j},order,'hor',T) + Cooc1(Uq{j},order,'ver',T);
 g.spam14v{j} = Cooc1(Rq{j},order,'ver',T) + Cooc1(Uq{j},order,'hor',T);
end
g.spam14h = sumcells(g.spam14h);
g.spam14v = sumcells(g.spam14v);

% minmax24 -- to be symmetrized as mnmx, directional, hv-symmetrical Note: Darn good, too.
for j=1:three
 [RU_min{j},RD_min{j},LU_min{j},LD_min{j}] = deal(min(Rq{j},Uq{j}),min(Rq{j},Dq{j}),min(Lq{j},Uq{j}),min(Lq{j},Dq{j}));
 [RU_max{j},RD_max{j},LU_max{j},LD_max{j}] = deal(max(Rq{j},Uq{j}),max(Rq{j},Dq{j}),max(Lq{j},Uq{j}),max(Lq{j},Dq{j}));
 g.min24{j} = Cooc1([RU_min{j};RD_min{j};LU_min{j};LD_min{j}],order,'hor',T) + Cooc1([RU_min{j} RD_min{j} LU_min{j} LD_min{j}],order,'ver',T);
 g.max24{j} = Cooc1([RU_max{j};RD_max{j};LU_max{j};LD_max{j}],order,'hor',T) + Cooc1([RU_max{j} RD_max{j} LU_max{j} LD_max{j}],order,'ver',T);
end
g.min24 = sumcells(g.min24);
g.max24 = sumcells(g.max24);

% minmax34h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical
% minmax34v -- v works well, h does not, to be symmetrized as mnmx, directional, hv-nonsymmetrical
for j=1:three
 [Uq_min{j},Rq_min{j},Dq_min{j},Lq_min{j}] = deal(min(RL_min{j},Uq{j}),min(UD_min{j},Rq{j}),min(RL_min{j},Dq{j}),min(UD_min{j},Lq{j}));
 [Uq_max{j},Rq_max{j},Dq_max{j},Lq_max{j}] = deal(max(RL_max{j},Uq{j}),max(UD_max{j},Rq{j}),max(RL_max{j},Dq{j}),max(UD_max{j},Lq{j}));
 g.min34h{j} = Cooc1([Uq_min{j};Dq_min{j}],order,'hor',T) + Cooc1([Rq_min{j} Lq_min{j}],order,'ver',T);
 g.max34h{j} = Cooc1([Uq_max{j};Dq_max{j}],order,'hor',T) + Cooc1([Rq_max{j} Lq_max{j}],order,'ver',T);
 g.min34v{j} = Cooc1([Uq_min{j} Dq_min{j}],order,'ver',T) + Cooc1([Rq_min{j};Lq_min{j}],order,'hor',T);
 g.max34v{j} = Cooc1([Uq_max{j} Dq_max{j}],order,'ver',T) + Cooc1([Rq_max{j};Lq_max{j}],order,'hor',T); 
end,    clear Rq Lq Uq Dq 
g.min34h = sumcells(g.min34h);
g.max34h = sumcells(g.max34h);
g.min34v = sumcells(g.min34v);
g.max34v = sumcells(g.max34v);

% minmax41 -- unknown performance as of 6/14/11, to be symmetrized as mnmx, non-directional, hv-symmetrical
for j=1:three
 [R_min{j},R_max{j}] = deal(min(RU_min{j},LD_min{j}),max(RU_max{j},LD_max{j}));
 g.min41{j} = Cooc1(R_min{j},order,'hor',T) + Cooc1(R_min{j},order,'ver',T);
 g.max41{j} = Cooc1(R_max{j},order,'hor',T) + Cooc1(R_max{j},order,'ver',T);
end
g.min41 = sumcells(g.min41);
g.max41 = sumcells(g.max41);

% minmax34 -- good, to be symmetrized as mnmx, directional, hv-symmetrical
for j=1:three
 [RU_min2{j},RD_min2{j},LU_min2{j},LD_min2{j}] = deal(min(RU_min{j},RU{j}),min(RD_min{j},RD{j}),min(LU_min{j},LU{j}),min(LD_min{j},LD{j}));
 [RU_max2{j},RD_max2{j},LU_max2{j},LD_max2{j}] = deal(max(RU_max{j},RU{j}),max(RD_max{j},RD{j}),max(LU_max{j},LU{j}),max(LD_max{j},LD{j}));
 g.min34{j} = Cooc1([RU_min2{j};RD_min2{j};LU_min2{j};LD_min2{j}],order,'hor',T) + Cooc1([RU_min2{j} RD_min2{j} LU_min2{j} LD_min2{j}],order,'ver',T);
 g.max34{j} = Cooc1([RU_max2{j};RD_max2{j};LU_max2{j};LD_max2{j}],order,'hor',T) + Cooc1([RU_max2{j} RD_max2{j} LU_max2{j} LD_max2{j}],order,'ver',T);
end,    clear RU_min RD_min LU_min LD_min RU_max RD_max LU_max LD_max
g.min34 = sumcells(g.min34);
g.max34 = sumcells(g.max34);

% minmax48h -- h better than v, to be symmetrized as mnmx, directional, hv-nonsymmetrical. 48v is almost as good as 48h for 3rd-order but weaker for 1st-order. Here, I am outputting both but Figure 1 in our paper lists only 48h.
for j=1:three
 [RU_min3{j},RD_min3{j},LD_min3{j},LU_min3{j}] = deal(min(RU_min2{j},LU{j}),min(RD_min2{j},RU{j}),min(LD_min2{j},RD{j}),min(LU_min2{j},LD{j}));
 [RU_min4{j},RD_min4{j},LD_min4{j},LU_min4{j}] = deal(min(RU_min2{j},RD{j}),min(RD_min2{j},LD{j}),min(LD_min2{j},LU{j}),min(LU_min2{j},RU{j}));
 g.min48h{j} = Cooc1([RU_min3{j};LD_min3{j};RD_min4{j};LU_min4{j}],order,'hor',T) + Cooc1([RD_min3{j} LU_min3{j} RU_min4{j} LD_min4{j}],order,'ver',T);
 g.min48v{j} = Cooc1([RU_min3{j} LD_min3{j} RD_min4{j} LU_min4{j}],order,'ver',T) + Cooc1([RD_min3{j};LU_min3{j};RU_min4{j};LD_min4{j}],order,'hor',T);
end,  	clear RU_min2 RD_min2 LU_min2 LD_min2
g.min48h = sumcells(g.min48h);
g.min48v = sumcells(g.min48v);
    clear RU_min4 RD_min4 LD_min4 LU_min4

for j=1:three
 [RU_max3{j},RD_max3{j},LD_max3{j},LU_max3{j}] = deal(max(RU_max2{j},LU{j}),max(RD_max2{j},RU{j}),max(LD_max2{j},RD{j}),max(LU_max2{j},LD{j}));
 [RU_max4{j},RD_max4{j},LD_max4{j},LU_max4{j}] = deal(max(RU_max2{j},RD{j}),max(RD_max2{j},LD{j}),max(LD_max2{j},LU{j}),max(LU_max2{j},RU{j}));
 g.max48h{j} = Cooc1([RU_max3{j};LD_max3{j};RD_max4{j};LU_max4{j}],order,'hor',T) + Cooc1([RD_max3{j} LU_max3{j} RU_max4{j} LD_max4{j}],order,'ver',T);
 g.max48v{j} = Cooc1([RU_max3{j} LD_max3{j} RD_max4{j} LU_max4{j}],order,'ver',T) + Cooc1([RD_max3{j};LU_max3{j};RU_max4{j};LD_max4{j}],order,'hor',T);
end,    clear RU_max2 RD_max2 LU_max2 LD_max2
g.max48h = sumcells(g.max48h);
g.max48v = sumcells(g.max48v);
    clear RU_max4 RD_max4 LD_max4 LU_max4 
    
% minmax54 -- to be symmetrized as mnmx, directional, hv-symmetrical
for j=1:three
 [RU_min5{j},RD_min5{j},LD_min5{j},LU_min5{j}] = deal(min(RU_min3{j},RD{j}),min(RD_min3{j},LD{j}),min(LD_min3{j},LU{j}),min(LU_min3{j},RU{j}));
 [RU_max5{j},RD_max5{j},LD_max5{j},LU_max5{j}] = deal(max(RU_max3{j},RD{j}),max(RD_max3{j},LD{j}),max(LD_max3{j},LU{j}),max(LU_max3{j},RU{j}));
 g.min54{j} = Cooc1([RU_min5{j};LD_min5{j};RD_min5{j};LU_min5{j}],order,'hor',T) + Cooc1([RD_min5{j} LU_min5{j} RU_min5{j} LD_min5{j}],order,'ver',T);
 g.max54{j} = Cooc1([RU_max5{j};LD_max5{j};RD_max5{j};LU_max5{j}],order,'hor',T) + Cooc1([RD_max5{j} LU_max5{j} RU_max5{j} LD_max5{j}],order,'ver',T);
end
g.min54 = sumcells(g.min54);
g.max54 = sumcells(g.max54);

%**** FUNCTION ****%
function g = all3rdc(X,q,T)
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

[M N three] = size(X);  [I,J,order] = deal(3:M-2,3:N-2,4);
for j=1:three
 [R,L,U,D] = deal(-X(I,J+2,j)+3*X(I,J+1,j)-3*X(I,J,j)+X(I,J-1,j),-X(I,J-2,j)+3*X(I,J-1,j)-3*X(I,J,j)+...
    X(I,J+1,j),-X(I-2,J,j)+3*X(I-1,J,j)-3*X(I,J,j)+X(I+1,J,j),-X(I+2,J,j)+3*X(I+1,J,j)-3*X(I,J,j)+X(I-1,J,j));
 [Rq{j},Lq{j},Uq{j},Dq{j}] = deal(Quant(R,q,T),Quant(L,q,T),Quant(U,q,T),Quant(D,q,T));          
 [RU{j},LU{j},RD{j},LD{j}] = deal(-X(I-2,J+2,j)+3*X(I-1,J+1,j)-3*X(I,J,j)+X(I+1,J-1,j),-X(I-2,J-2,j)+3*X(I-1,J-1,j)-3*X(I,J,j)+...
    X(I+1,J+1,j),-X(I+2,J+2,j)+3*X(I+1,J+1,j)-3*X(I,J,j)+X(I-1,J-1,j),-X(I+2,J-2,j)+3*X(I+1,J-1,j)-3*X(I,J,j)+X(I-1,J+1,j));
 [RU{j},LU{j},RD{j},LD{j}] = deal(Quant(RU{j},q,T),Quant(LU{j},q,T),Quant(RD{j},q,T),Quant(LD{j},q,T));
end,    clear R L U D X
% minmax22h,v -- to be symmetrized as mnmx, directional, hv-nonsymmetrical. Good with higher-order residuals! Note: 22h is bad (too much neighborhood overlap).
for j=1:three
 [RL_min{j},UD_min{j}] = deal(min(Rq{j},Lq{j}),min(Uq{j},Dq{j}));
 [RL_max{j},UD_max{j}] = deal(max(Rq{j},Lq{j}),max(Uq{j},Dq{j}));
end
 g.min22c = Cooc1(RL_min,3,'col',T) + Cooc1(UD_min,3,'col',T);
 g.max22c = Cooc1(RL_max,3,'col',T) + Cooc1(UD_max,3,'col',T);

% spam14h,v -- to be symmetrized as spam, directional, hv-nonsymmetrical
 g.spam14c = Cooc1(Rq,3,'col',T) + Cooc1(Uq,3,'col',T);

% minmax24 -- to be symmetrized as mnmx, directional, hv-symmetrical Note: Darn good, too.
for j=1:three
 [RU_min{j},RD_min{j},LU_min{j},LD_min{j}] = deal(min(Rq{j},Uq{j}),min(Rq{j},Dq{j}),min(Lq{j},Uq{j}),min(Lq{j},Dq{j}));
 [RU_max{j},RD_max{j},LU_max{j},LD_max{j}] = deal(max(Rq{j},Uq{j}),max(Rq{j},Dq{j}),max(Lq{j},Uq{j}),max(Lq{j},Dq{j}));
end
 g.min24c = Cooc1(vercatcell(vercatcell(RU_min,RD_min),vercatcell(LU_min,LD_min)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RU_min,RD_min),horcatcell(LU_min,LD_min)),3,'col',T);
 g.max24c = Cooc1(vercatcell(vercatcell(RU_max,RD_max),vercatcell(LU_max,LD_max)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RU_max,RD_max),horcatcell(LU_max,LD_max)),3,'col',T);

% minmax34h -- to be symmetrized as mnmx, directional, hv-nonsymmetrical
% minmax34v -- v works well, h does not, to be symmetrized as mnmx, directional, hv-nonsymmetrical
for j=1:three
 [Uq_min{j},Rq_min{j},Dq_min{j},Lq_min{j}] = deal(min(RL_min{j},Uq{j}),min(UD_min{j},Rq{j}),min(RL_min{j},Dq{j}),min(UD_min{j},Lq{j}));
 [Uq_max{j},Rq_max{j},Dq_max{j},Lq_max{j}] = deal(max(RL_max{j},Uq{j}),max(UD_max{j},Rq{j}),max(RL_max{j},Dq{j}),max(UD_max{j},Lq{j}));
end,    clear Rq Lq Uq Dq 
 g.min34hvc = Cooc1(vercatcell(Uq_min,Dq_min),3,'col',T) + Cooc1(horcatcell(Rq_min,Lq_min),3,'col',T);    clear Uq_min Rq_min Dq_min Lq_min
 g.max34hvc = Cooc1(vercatcell(Uq_max,Dq_max),3,'col',T) + Cooc1(horcatcell(Rq_max,Lq_max),3,'col',T);    clear Uq_max Rq_max Dq_max Lq_max

% minmax41 -- unknown performance as of 6/14/11, to be symmetrized as mnmx, non-directional, hv-symmetrical
for j=1:three
 [R_min{j},R_max{j}] = deal(min(RU_min{j},LD_min{j}),max(RU_max{j},LD_max{j}));
end
 g.min41c = Cooc1(R_min,3,'col',T);
 g.max41c = Cooc1(R_max,3,'col',T);

% minmax34 -- good, to be symmetrized as mnmx, directional, hv-symmetrical
for j=1:three
 [RU_min2{j},RD_min2{j},LU_min2{j},LD_min2{j}] = deal(min(RU_min{j},RU{j}),min(RD_min{j},RD{j}),min(LU_min{j},LU{j}),min(LD_min{j},LD{j}));
 [RU_max2{j},RD_max2{j},LU_max2{j},LD_max2{j}] = deal(max(RU_max{j},RU{j}),max(RD_max{j},RD{j}),max(LU_max{j},LU{j}),max(LD_max{j},LD{j}));
end,    clear RU_min RD_min LU_min LD_min RU_max RD_max LU_max LD_max
 g.min34c = Cooc1(vercatcell(vercatcell(RU_min2,RD_min2),vercatcell(LU_min2,LD_min2)),3,'col',T);
 g.max34c = Cooc1(vercatcell(vercatcell(RU_max2,RD_max2),vercatcell(LU_max2,LD_max2)),3,'col',T);

% minmax48h -- h better than v, to be symmetrized as mnmx, directional, hv-nonsymmetrical. 48v is almost as good as 48h for 3rd-order but weaker for 1st-order. Here, I am outputting both but Figure 1 in our paper lists only 48h.
for j=1:three
 [RU_min3{j},RD_min3{j},LD_min3{j},LU_min3{j}] = deal(min(RU_min2{j},LU{j}),min(RD_min2{j},RU{j}),min(LD_min2{j},RD{j}),min(LU_min2{j},LD{j}));
 [RU_min4{j},RD_min4{j},LD_min4{j},LU_min4{j}] = deal(min(RU_min2{j},RD{j}),min(RD_min2{j},LD{j}),min(LD_min2{j},LU{j}),min(LU_min2{j},RU{j}));
end,  	clear RU_min2 RD_min2 LU_min2 LD_min2
 g.min48c = Cooc1(vercatcell(vercatcell(RU_min3,LD_min3),vercatcell(RD_min4,LU_min4)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RD_min3,LU_min3),horcatcell(RU_min4,LD_min4)),3,'col',T);
    clear RU_min4 RD_min4 LD_min4 LU_min4

for j=1:three
 [RU_max3{j},RD_max3{j},LD_max3{j},LU_max3{j}] = deal(max(RU_max2{j},LU{j}),max(RD_max2{j},RU{j}),max(LD_max2{j},RD{j}),max(LU_max2{j},LD{j}));
 [RU_max4{j},RD_max4{j},LD_max4{j},LU_max4{j}] = deal(max(RU_max2{j},RD{j}),max(RD_max2{j},LD{j}),max(LD_max2{j},LU{j}),max(LU_max2{j},RU{j}));
end,    clear RU_max2 RD_max2 LU_max2 LD_max2
 g.max48c = Cooc1(vercatcell(vercatcell(RU_max3,LD_max3),vercatcell(RD_max4,LU_max4)),3,'col',T) +...
     Cooc1(horcatcell(horcatcell(RD_max3,LU_max3),horcatcell(RU_max4,LD_max4)),3,'col',T);
    clear RU_max4 RD_max4 LD_max4 LU_max4 
    
% minmax54 -- to be symmetrized as mnmx, directional, hv-symmetrical
for j=1:three
 [RU_min5{j},RD_min5{j},LD_min5{j},LU_min5{j}] = deal(min(RU_min3{j},RD{j}),min(RD_min3{j},LD{j}),min(LD_min3{j},LU{j}),min(LU_min3{j},RU{j}));
 [RU_max5{j},RD_max5{j},LD_max5{j},LU_max5{j}] = deal(max(RU_max3{j},RD{j}),max(RD_max3{j},LD{j}),max(LD_max3{j},LU{j}),max(LU_max3{j},RU{j}));
end
 g.min54c = Cooc1(vercatcell(vercatcell(RU_min5,LD_min5),vercatcell(RD_min5,LU_min5)),3,'col',T);
 g.max54c = Cooc1(vercatcell(vercatcell(RU_max5,LD_max5),vercatcell(RD_max5,LU_max5)),3,'col',T);

%**** FUNCTION ****%
function g = all3x3(X,q)
% This function outputs co-occurrences of ALL residuals based on the
% KB kernel and its "halves" (EDGE residuals) as listed in Figure 1
% in our journal HUGO paper (version from June 14), including the naming
% convention.

[M N three] = size(X);   [T,order] = deal(2,4);
% spam11 (old name KB residual), good, non-directional, hv-symmetrical, to be symmetrized as spam
for j=1:three
 D{j} = Residual(X(:,:,j),2,'KB'); Y{j} = Quant(D{j},q,T);
 g.spam11{j} = Cooc1(Y{j},order,'hor',T) + Cooc1(Y{j},order,'ver',T);
end
g.spam11 = sumcells(g.spam11);

% EDGE residuals
for j=1:three
 D{j} = Residual(X(:,:,j),2,'edge-h'); Du{j} = D{j}(:,1:size(D{j},2)/2); Db{j} = D{j}(:,size(D{j},2)/2+1:end);
 D{j} = Residual(X(:,:,j),2,'edge-v'); Dl{j} = D{j}(:,1:size(D{j},2)/2); Dr{j} = D{j}(:,size(D{j},2)/2+1:end);
 [Yu{j},Yb{j},Yl{j},Yr{j}] = deal(Quant(Du{j},q,T),Quant(Db{j},q,T),Quant(Dl{j},q,T),Quant(Dr{j},q,T));
end,     clear D

% spam14h,v  not bad, directional, hv-nonsym, to be symmetrized as spam
for j=1:three
 g.spam14v{j} = Cooc1([Yu{j} Yb{j}],order,'ver',T) + Cooc1([Yl{j};Yr{j}],order,'hor',T);
 g.spam14h{j} = Cooc1([Yu{j};Yb{j}],order,'hor',T) + Cooc1([Yl{j} Yr{j}],order,'ver',T);
end
g.spam14v = sumcells(g.spam14v);
g.spam14h = sumcells(g.spam14h);

% minmax24 -- EXCELLENT, directional, hv-sym, to be symmetrized as mnmx
for j=1:three
 [Dmin1{j},Dmin2{j},Dmin3{j},Dmin4{j}] = deal(min(Yu{j},Yl{j}),min(Yb{j},Yr{j}),min(Yu{j},Yr{j}),min(Yb{j},Yl{j}));
 g.min24{j} = Cooc1([Dmin1{j} Dmin2{j} Dmin3{j} Dmin4{j}],order,'ver',T) + Cooc1([Dmin1{j};Dmin2{j};Dmin3{j};Dmin4{j}],order,'hor',T);
 [Dmax1{j},Dmax2{j},Dmax3{j},Dmax4{j}] = deal(max(Yu{j},Yl{j}),max(Yb{j},Yr{j}),max(Yu{j},Yr{j}),max(Yb{j},Yl{j}));
 g.max24{j} = Cooc1([Dmax1{j} Dmax2{j} Dmax3{j} Dmax4{j}],order,'ver',T) + Cooc1([Dmax1{j};Dmax2{j};Dmax3{j};Dmax4{j}],order,'hor',T);
end 
g.min24 = sumcells(g.min24);
g.max24 = sumcells(g.max24);
    clear Dmin3 Dmin4 Dmax3 Dmax4
    
% minmax22 - hv-nonsymmetrical
% min22h -- good, to be symmetrized as mnmx, directional, hv-nonsymmetrical
% min22v -- EXCELLENT - to be symmetrized as mnmx, directional,
for j=1:three
 [UEq_min{j},REq_min{j}] = deal(min(Yu{j},Yb{j}),min(Yr{j},Yl{j}));
 g.min22h{j} = Cooc1(UEq_min{j},order,'hor',T) + Cooc1(REq_min{j},order,'ver',T);
 g.min22v{j} = Cooc1(UEq_min{j},order,'ver',T) + Cooc1(REq_min{j},order,'hor',T);
 [UEq_max{j},REq_max{j}] = deal(max(Yu{j},Yb{j}),max(Yr{j},Yl{j}));
 g.max22h{j} = Cooc1(UEq_max{j},order,'hor',T) + Cooc1(REq_max{j},order,'ver',T);
 g.max22v{j} = Cooc1(UEq_max{j},order,'ver',T) + Cooc1(REq_max{j},order,'hor',T);
end
g.min22h = sumcells(g.min22h);
g.min22v = sumcells(g.min22v);
g.max22h = sumcells(g.max22h);
g.max22v = sumcells(g.max22v);

% minmax41 -- good, non-directional, hv-sym, to be symmetrized as mnmx
for j=1:three
 [Dmin5{j},Dmax5{j}] = deal(min(Dmin1{j},Dmin2{j}),max(Dmax1{j},Dmax2{j}));
 g.min41{j} = Cooc1(Dmin5{j},order,'ver',T) + Cooc1(Dmin5{j},order,'hor',T);
 g.max41{j} = Cooc1(Dmax5{j},order,'ver',T) + Cooc1(Dmax5{j},order,'hor',T);
end,     clear Dmin1 Dmin2 Dmax1 Dmax2 
g.min41 = sumcells(g.min41);
g.max41 = sumcells(g.max41);

%**** FUNCTION ****%
function g = all3x3c(X,q,T)
% This function outputs co-occurrences of ALL residuals based on the
% KB kernel and its "halves" (EDGE residuals) as listed in Figure 1
% in our journal HUGO paper (version from June 14), including the naming
% convention.

[M N three] = size(X);   order = 4;
% spam11 (old name KB residual), good, non-directional, hv-symmetrical, to be symmetrized as spam
for j=1:three
 D{j} = Residual(X(:,:,j),2,'KB'); Y{j} = Quant(D{j},q,T);
end
 g.spam11c = Cooc1(Y,3,'col',T);

% EDGE residuals
for j=1:three
 D{j} = Residual(X(:,:,j),2,'edge-h'); Du{j} = D{j}(:,1:size(D{j},2)/2); Db{j} = D{j}(:,size(D{j},2)/2+1:end);
 D{j} = Residual(X(:,:,j),2,'edge-v'); Dl{j} = D{j}(:,1:size(D{j},2)/2); Dr{j} = D{j}(:,size(D{j},2)/2+1:end);
 [Yu{j},Yb{j},Yl{j},Yr{j}] = deal(Quant(Du{j},q,T),Quant(Db{j},q,T),Quant(Dl{j},q,T),Quant(Dr{j},q,T));
end,     clear D

% spam14h,v  not bad, directional, hv-nonsym, to be symmetrized as spam
 g.spam14c = Cooc1(vercatcell(Yu,Yb),3,'col',T) + Cooc1(horcatcell(Yl,Yr),3,'col',T);

% minmax24 -- EXCELLENT, directional, hv-sym, to be symmetrized as mnmx
for j=1:three
 [Dmin1{j},Dmin2{j},Dmin3{j},Dmin4{j}] = deal(min(Yu{j},Yl{j}),min(Yb{j},Yr{j}),min(Yu{j},Yr{j}),min(Yb{j},Yl{j}));
 [Dmax1{j},Dmax2{j},Dmax3{j},Dmax4{j}] = deal(max(Yu{j},Yl{j}),max(Yb{j},Yr{j}),max(Yu{j},Yr{j}),max(Yb{j},Yl{j}));
end 
 g.min24c = Cooc1(vercatcell(vercatcell(Dmin1,Dmin2),vercatcell(Dmin3,Dmin4)),3,'col',T);
 g.max24c = Cooc1(vercatcell(vercatcell(Dmax1,Dmax2),vercatcell(Dmax3,Dmax4)),3,'col',T);
    clear Dmin3 Dmin4 Dmax3 Dmax4
    
% minmax22 - hv-nonsymmetrical
% min22h -- good, to be symmetrized as mnmx, directional, hv-nonsymmetrical
% min22v -- EXCELLENT - to be symmetrized as mnmx, directional,
for j=1:three
 [UEq_min{j},REq_min{j}] = deal(min(Yu{j},Yb{j}),min(Yr{j},Yl{j}));
 [UEq_max{j},REq_max{j}] = deal(max(Yu{j},Yb{j}),max(Yr{j},Yl{j}));
end
 g.min22c = Cooc1(UEq_min,3,'col',T) + Cooc1(REq_min,3,'col',T);
 g.max22c = Cooc1(UEq_max,3,'col',T) + Cooc1(REq_max,3,'col',T);

% minmax41 -- good, non-directional, hv-sym, to be symmetrized as mnmx
for j=1:three
 [Dmin5{j},Dmax5{j}] = deal(min(Dmin1{j},Dmin2{j}),max(Dmax1{j},Dmax2{j}));
end,     clear Dmin1 Dmin2 Dmax1 Dmax2 
 g.min41c = Cooc1(Dmin5,3,'col',T);
 g.max41c = Cooc1(Dmax5,3,'col',T);

%**** FUNCTION ****%
function g = all5x5(X,q)
% This function outputs co-occurrences of ALL residuals based on the
% KV kernel and its "halves" (EDGE residuals) as listed in Figure 1
% in our journal HUGO paper (version from June 14), including the naming
% convention.
[M N three] = size(X);  [I,J,T,order] = deal(3:M-2,3:N-2,2,4);
% spam11 (old name KV residual), good, non-directional, hv-symmetrical, to be symmetrized as spam
for j=1:three
 D{j} = Residual(X(:,:,j),3,'KV'); Y{j} = Quant(D{j},q,T);
 g.spam11{j} = Cooc1(Y{j},order,'hor',T) + Cooc1(Y{j},order,'ver',T);
end
g.spam11 = sumcells(g.spam11);

% EDGE residuals    
for j=1:three
 Du = 8*X(I,J-1,j)+8*X(I-1,J,j)+8*X(I,J+1,j)-6*X(I-1,J-1,j)-6*X(I-1,J+1,j)-2*X(I,J-2,j)-2*X(I,J+2,j)-2*X(I-2,J,j)+2*X(I-1,J-2,j)+2*X(I-2,J-1,j)+2*X(I-2,J+1,j)+2*X(I-1,J+2,j)-X(I-2,J-2,j)-X(I-2,J+2,j)-12*X(I,J,j);
 Dr = 8*X(I-1,J,j)+8*X(I,J+1,j)+8*X(I+1,J,j)-6*X(I-1,J+1,j)-6*X(I+1,J+1,j)-2*X(I-2,J,j)-2*X(I+2,J,j)-2*X(I,J+2,j)+2*X(I-2,J+1,j)+2*X(I-1,J+2,j)+2*X(I+1,J+2,j)+2*X(I+2,J+1,j)-X(I-2,J+2,j)-X(I+2,J+2,j)-12*X(I,J,j);
 Db = 8*X(I,J+1,j)+8*X(I+1,J,j)+8*X(I,J-1,j)-6*X(I+1,J+1,j)-6*X(I+1,J-1,j)-2*X(I,J-2,j)-2*X(I,J+2,j)-2*X(I+2,J,j)+2*X(I+1,J+2,j)+2*X(I+2,J+1,j)+2*X(I+2,J-1,j)+2*X(I+1,J-2,j)-X(I+2,J+2,j)-X(I+2,J-2,j)-12*X(I,J,j);
 Dl = 8*X(I+1,J,j)+8*X(I,J-1,j)+8*X(I-1,J,j)-6*X(I+1,J-1,j)-6*X(I-1,J-1,j)-2*X(I-2,J,j)-2*X(I+2,J,j)-2*X(I,J-2,j)+2*X(I+2,J-1,j)+2*X(I+1,J-2,j)+2*X(I-1,J-2,j)+2*X(I-2,J-1,j)-X(I+2,J-2,j)-X(I-2,J-2,j)-12*X(I,J,j);
 [Yu{j},Yb{j},Yl{j},Yr{j}] = deal(Quant(Du,q,T),Quant(Db,q,T),Quant(Dl,q,T),Quant(Dr,q,T));
end,    clear Du Db Dl Dr

% spam14v  not bad, directional, hv-nonsym, to be symmetrized as spam
for j=1:three
 g.spam14v{j} = Cooc1([Yu{j} Yb{j}],order,'ver',T) + Cooc1([Yl{j};Yr{j}],order,'hor',T);
 g.spam14h{j} = Cooc1([Yu{j};Yb{j}],order,'hor',T) + Cooc1([Yl{j} Yr{j}],order,'ver',T);
end
g.spam14v = sumcells(g.spam14v);
g.spam14h = sumcells(g.spam14h);

% minmax24 -- EXCELLENT, directional, hv-sym, to be symmetrized as mnmx
for j=1:three
 [Dmin1{j},Dmin2{j},Dmin3{j},Dmin4{j}] = deal(min(Yu{j},Yl{j}),min(Yb{j},Yr{j}),min(Yu{j},Yr{j}),min(Yb{j},Yl{j}));
 g.min24{j} = Cooc1([Dmin1{j} Dmin2{j} Dmin3{j} Dmin4{j}],order,'ver',T) + Cooc1([Dmin1{j};Dmin2{j};Dmin3{j};Dmin4{j}],order,'hor',T);
 [Dmax1{j},Dmax2{j},Dmax3{j},Dmax4{j}] = deal(max(Yu{j},Yl{j}),max(Yb{j},Yr{j}),max(Yu{j},Yr{j}),max(Yb{j},Yl{j}));
 g.max24{j} = Cooc1([Dmax1{j} Dmax2{j} Dmax3{j} Dmax4{j}],order,'ver',T) + Cooc1([Dmax1{j};Dmax2{j};Dmax3{j};Dmax4{j}],order,'hor',T);
end
g.min24 = sumcells(g.min24);
g.max24 = sumcells(g.max24);
 
% minmax22 - hv-nonsymmetrical
% min22h -- good, to be symmetrized as mnmx, directional, hv-nonsymmetrical
% min22v -- EXCELLENT - to be symmetrized as mnmx, directional,
for j=1:three
 [UEq_min{j},REq_min{j}] = deal(min(Yu{j},Yb{j}),min(Yr{j},Yl{j}));
 g.min22h{j} = Cooc1(UEq_min{j},order,'hor',T) + Cooc1(REq_min{j},order,'ver',T);
 g.min22v{j} = Cooc1(UEq_min{j},order,'ver',T) + Cooc1(REq_min{j},order,'hor',T);
 [UEq_max{j},REq_max{j}] = deal(max(Yu{j},Yb{j}),max(Yr{j},Yl{j}));
 g.max22h{j} = Cooc1(UEq_max{j},order,'hor',T) + Cooc1(REq_max{j},order,'ver',T);
 g.max22v{j} = Cooc1(UEq_max{j},order,'ver',T) + Cooc1(REq_max{j},order,'hor',T);
end
g.min22h = sumcells(g.min22h);
g.min22v = sumcells(g.min22v);
g.max22h = sumcells(g.max22h);
g.max22v = sumcells(g.max22v);

% minmax41 -- good, non-directional, hv-sym, to be symmetrized as mnmx
for j=1:three
 [Dmin5{j},Dmax5{j}] = deal(min(Dmin1{j},Dmin2{j}),max(Dmax1{j},Dmax2{j}));
 g.min41{j} = Cooc1(Dmin5{j},order,'ver',T) + Cooc1(Dmin5{j},order,'hor',T);
 g.max41{j} = Cooc1(Dmax5{j},order,'ver',T) + Cooc1(Dmax5{j},order,'hor',T);
end
g.min41 = sumcells(g.min41);
g.max41 = sumcells(g.max41);

%**** FUNCTION ****%
function g = all5x5c(X,q,T)
% This function outputs co-occurrences of ALL residuals based on the
% KV kernel and its "halves" (EDGE residuals) as listed in Figure 1
% in our journal HUGO paper (version from June 14), including the naming
% convention.
[M N three] = size(X);  [I,J,order] = deal(3:M-2,3:N-2,4);
% spam11 (old name KV residual), good, non-directional, hv-symmetrical, to be symmetrized as spam
for j=1:three
 D{j} = Residual(X(:,:,j),3,'KV'); Y{j} = Quant(D{j},q,T);
end
 g.spam11c = Cooc1(Y,3,'col',T);

% EDGE residuals    
for j=1:three
 Du = 8*X(I,J-1,j)+8*X(I-1,J,j)+8*X(I,J+1,j)-6*X(I-1,J-1,j)-6*X(I-1,J+1,j)-2*X(I,J-2,j)-2*X(I,J+2,j)-2*X(I-2,J,j)+2*X(I-1,J-2,j)+2*X(I-2,J-1,j)+2*X(I-2,J+1,j)+2*X(I-1,J+2,j)-X(I-2,J-2,j)-X(I-2,J+2,j)-12*X(I,J,j);
 Dr = 8*X(I-1,J,j)+8*X(I,J+1,j)+8*X(I+1,J,j)-6*X(I-1,J+1,j)-6*X(I+1,J+1,j)-2*X(I-2,J,j)-2*X(I+2,J,j)-2*X(I,J+2,j)+2*X(I-2,J+1,j)+2*X(I-1,J+2,j)+2*X(I+1,J+2,j)+2*X(I+2,J+1,j)-X(I-2,J+2,j)-X(I+2,J+2,j)-12*X(I,J,j);
 Db = 8*X(I,J+1,j)+8*X(I+1,J,j)+8*X(I,J-1,j)-6*X(I+1,J+1,j)-6*X(I+1,J-1,j)-2*X(I,J-2,j)-2*X(I,J+2,j)-2*X(I+2,J,j)+2*X(I+1,J+2,j)+2*X(I+2,J+1,j)+2*X(I+2,J-1,j)+2*X(I+1,J-2,j)-X(I+2,J+2,j)-X(I+2,J-2,j)-12*X(I,J,j);
 Dl = 8*X(I+1,J,j)+8*X(I,J-1,j)+8*X(I-1,J,j)-6*X(I+1,J-1,j)-6*X(I-1,J-1,j)-2*X(I-2,J,j)-2*X(I+2,J,j)-2*X(I,J-2,j)+2*X(I+2,J-1,j)+2*X(I+1,J-2,j)+2*X(I-1,J-2,j)+2*X(I-2,J-1,j)-X(I+2,J-2,j)-X(I-2,J-2,j)-12*X(I,J,j);
 [Yu{j},Yb{j},Yl{j},Yr{j}] = deal(Quant(Du,q,T),Quant(Db,q,T),Quant(Dl,q,T),Quant(Dr,q,T));
end,    clear Du Db Dl Dr

% spam14v  not bad, directional, hv-nonsym, to be symmetrized as spam
 g.spam14c = Cooc1(horcatcell(Yu,Yb),3,'col',T) + Cooc1(vercatcell(Yl,Yr),3,'col',T);

% minmax24 -- EXCELLENT, directional, hv-sym, to be symmetrized as mnmx
for j=1:three
 [Dmin1{j},Dmin2{j},Dmin3{j},Dmin4{j}] = deal(min(Yu{j},Yl{j}),min(Yb{j},Yr{j}),min(Yu{j},Yr{j}),min(Yb{j},Yl{j}));
 [Dmax1{j},Dmax2{j},Dmax3{j},Dmax4{j}] = deal(max(Yu{j},Yl{j}),max(Yb{j},Yr{j}),max(Yu{j},Yr{j}),max(Yb{j},Yl{j}));
end
 g.min24c = Cooc1(horcatcell(horcatcell(Dmin1,Dmin2),horcatcell(Dmin3,Dmin4)),3,'col',T);      clear Dmin3 Dmin4 
 g.max24c = Cooc1(horcatcell(horcatcell(Dmax1,Dmax2),horcatcell(Dmax3,Dmax4)),3,'col',T);      clear Dmax3 Dmax4
 
% minmax22 - hv-nonsymmetrical
% min22h -- good, to be symmetrized as mnmx, directional, hv-nonsymmetrical
% min22v -- EXCELLENT - to be symmetrized as mnmx, directional,
for j=1:three
 [UEq_min{j},REq_min{j}] = deal(min(Yu{j},Yb{j}),min(Yr{j},Yl{j}));
 [UEq_max{j},REq_max{j}] = deal(max(Yu{j},Yb{j}),max(Yr{j},Yl{j}));
end
 g.min22c = Cooc1(UEq_min,3,'col',T) + Cooc1(REq_min,3,'col',T);        clear UEq_min REq_min
 g.max22c = Cooc1(UEq_max,3,'col',T) + Cooc1(REq_max,3,'col',T);        clear UEq_max REq_max

% minmax41 -- good, non-directional, hv-sym, to be symmetrized as mnmx
for j=1:three
 [Dmin5{j},Dmax5{j}] = deal(min(Dmin1{j},Dmin2{j}),max(Dmax1{j},Dmax2{j}));
end
 g.min41c = Cooc1(Dmin5,3,'col',T);
 g.max41c = Cooc1(Dmax5,3,'col',T);

%**** FUNCTION ****%
function f = Cooc1(D,order,type,T)
% Co-occurrence operator to be appied to a 2D array of residuals D \in [-T,T]
% T     ... threshold
% order ... cooc order \in {1,2,3,4,5}
% type  ... cooc type \in {hor,ver,diag,mdiag,square,square-ori,hvdm}
% f     ... an array of length (2T+1)^order  (1D array!)

if iscell(D),
  for j=1:numel(D)
    if max(abs(D{j}(:))) > T, fprintf('*** ERROR in Cooc1.m: Residual out of range ***\n'), end
  end  
else
    if max(abs(D(:))) > T, fprintf('*** ERROR in Cooc1.m: Residual out of range ***\n'), end
end
B = 2*T+1;  z = (B^order-1)/2;  range = (-z:z);
switch order
    case 1
        f = histc(D(:),-T:T);
    case 2
        f = zeros(B,B);
        if strcmp(type,'hor'),   L = D(:,1:end-1); R = D(:,2:end);end
        if strcmp(type,'ver'),   L = D(1:end-1,:); R = D(2:end,:);end
        if strcmp(type,'diag'),  L = D(1:end-1,1:end-1); R = D(2:end,2:end);end
        if strcmp(type,'mdiag'), L = D(1:end-1,2:end); R = D(2:end,1:end-1);end
        A = L(:)+B*R(:);   clear L R
        f = histc(A,range); 
    case 3
        f = zeros(B,B,B);
        if strcmp(type,'hor'),   L = D(:,1:end-2); C = D(:,2:end-1); R = D(:,3:end);end
        if strcmp(type,'ver'),   L = D(1:end-2,:); C = D(2:end-1,:); R = D(3:end,:);end
        if strcmp(type,'diag'),  L = D(1:end-2,1:end-2); C = D(2:end-1,2:end-1); R = D(3:end,3:end);end
        if strcmp(type,'mdiag'), L = D(1:end-2,3:end); C = D(2:end-1,2:end-1); R = D(3:end,1:end-2);end
        if strcmp(type,'col'),   L = D{1}; C = D{2}; R = D{3};  end       
        A = L(:)+B*C(:)+B^2*R(:);    clear L C R
        f = histc(A,range); 
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
        A = L(:)+B*C(:)+B^2*E(:)+B^3*R(:);    clear L C E R
        f = histc(A,range); 
    case 5
        f = zeros(B,B,B,B,B);
        if strcmp(type,'hor'),L = D(:,1:end-4); C = D(:,2:end-3); E = D(:,3:end-2); F = D(:,4:end-1); R = D(:,5:end);end
        if strcmp(type,'ver'),L = D(1:end-4,:); C = D(2:end-3,:); E = D(3:end-2,:); F = D(4:end-1,:); R = D(5:end,:);end
        A = L(:)+B*C(:)+B^2*E(:)+B^3*F(:)+B^4*R(:);    clear L C E F R
        f = histc(A,range); 
end

% normalization
f = double(f);
fsum = sum(f(:));
if fsum>0, f = f/fsum; end

%**** FUNCTION ****%
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

%**** FUNCTION ****%
function Z = trunc(X,T)
% Truncation to [-T,T]
Z = X;
Z(Z > T)  =  T;
Z(Z < -T) = -T;

%**** FUNCTION ****%
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

if strcmp(type,'mnmxc')
    if dim == 2*c
        switch order
            case 3, red = c;        
            case 4, red = c;
        end
        fsym = zeros(red, N);
        for i = 1 : N
            switch order
                case 1, cube_min = f(1:c,i); cube_max = f(c+1:2*c,i);
                case 2, cube_min = reshape(f(1:c,i),[B B]); cube_max = reshape(f(c+1:2*c,i),[B B]);  f_mmsym = cube_min + cube_max(end:-1:1,end:-1:1);
                case 3, cube_min = reshape(f(1:c,i),[B B B]); cube_max = reshape(f(c+1:2*c,i),[B B B]);  f_mmsym = cube_min + cube_max(end:-1:1,end:-1:1,end:-1:1);
                case 4, cube_min = reshape(f(1:c,i),[B B B B]); cube_max = reshape(f(c+1:2*c,i),[B B B B]);  f_mmsym = cube_min + cube_max(end:-1:1,end:-1:1,end:-1:1,end:-1:1);
                case 5, cube_min = reshape(f(1:c,i),[B B B B B]); cube_max = reshape(f(c+1:2*c,i),[B B B B B]);  f_mmsym = cube_min + cube_max(end:-1:1,end:-1:1,end:-1:1,end:-1:1,end:-1:1);
            end
            fsym(:,i) = f_mmsym(:);
        end
    end
    ERR = 0;
end
if ERR == 1, fprintf('*** ERROR: Feature dimension and T, order incompatible. ***\n'), end

%**** FUNCTION ****%
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

%**** FUNCTION ****%
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

%**** FUNCTION ****%
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

%**** FUNCTION ****%
function fsym = symm1sign(f,T,order)
% Marginalization by sign symmetry for a feature vector
% stored as a (2T+1)^order-dimensional array. The input feature f is 
% assumed to be a dim x database_size matrix of features stored as columns.

[dim,N] = size(f);
B = 2*T+1;
c = B^order;
ERR = 1;

if dim == c
    ERR = 0;
    red = (B^order+1)/2;    % Reduced dimensionality for a c-dimensional feature vector
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
        fsym(:,i) = symmsign(cube,T,order);
    end
end

if ERR == 1, fprintf('*** ERROR in symm1: Feature dimension and T, order incompatible. ***\n'), end

%**** FUNCTION ****%
function As = symmsign(A,T,order)
% Symmetry marginalization routine. The purpose is to reduce the feature
% dimensionality and make the features more populated. It can be applied to
% 1D -- 5D co-occurrence matrices (order \in {1,2,3,4,5}) with sign 
% symmetries (explained below). 
% A must be an array of (2*T+1)^order, otherwise error is outputted.
%
% Marginalization by symmetry pertains to the fact that, fundamentally,
% the differences between consecutive pixels in a natural image (both cover
% and stego) d1, d2, d3, ..., have the same probability of occurrence as the
% triple -d1, -d2, -d3, ...

ERR = 1;  % Flag denoting when size of A is incompatible with the input parameters T and order
m = 2;
B = 2*T+1;
red = (B^order+1)/2;        % Reduced dimensionality for a c-dimensional feature vector

switch order
    case 1  % First-order coocs
        if numel(A) == 2*T+1
           As(1) = A(T+1);  % The only non-marginalized bin is the origin 0
           As(2:T+1) = A(1:T) + A(T+2:end);
           As = As(:);
           ERR = 0;
        end
    case 2
        if numel(A) == (2*T+1)^2
            As = zeros(red, 1);
            As(1) = A(T+1,T+1); % The only non-marginalized bin is the origin (0,0)
            for i = -T : T
                for j = -T : T
                    if (done(i+T+1,j+T+1) == 0) && (abs(i)+abs(j) ~= 0)
                        As(m) = A(i+T+1,j+T+1) + A(T+1-i,T+1-j);
                        done(i+T+1,j+T+1) = 1;
                        done(T+1-i,T+1-j) = 1;
                        m = m + 1;
                    end
                end
            end
            ERR = 0;
        end
    case 3
        if numel(A) == B^3
            done = zeros(size(A));
            As = zeros(red, 1);
            As(1) = A(T+1,T+1,T+1); % The only non-marginalized bin is the origin (0,0,0)
            for i = -T : T
                for j = -T : T
                    for k = -T : T
                        if (done(i+T+1,j+T+1,k+T+1) == 0) && (abs(i)+abs(j)+abs(k) ~= 0)
                            As(m) = A(i+T+1,j+T+1,k+T+1) + A(T+1-i,T+1-j,T+1-k);
                            done(i+T+1,j+T+1,k+T+1) = 1;
                            done(T+1-i,T+1-j,T+1-k) = 1;
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
            As = zeros(red, 1);
            As(1) = A(T+1,T+1,T+1,T+1); % The only non-marginalized bin is the origin (0,0,0,0)
            for i = -T : T
                for j = -T : T
                    for k = -T : T
                        for n = -T : T
                            if (done(i+T+1,j+T+1,k+T+1,n+T+1) == 0) && (abs(i)+abs(j)+abs(k)+abs(n)~=0)
                                As(m) = A(i+T+1,j+T+1,k+T+1,n+T+1) + A(T+1-i,T+1-j,T+1-k,T+1-n);
                                done(i+T+1,j+T+1,k+T+1,n+T+1) = 1;
                                done(T+1-i,T+1-j,T+1-k,T+1-n) = 1;
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
            As = zeros(red, 1);
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

%**** FUNCTION ****%
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
