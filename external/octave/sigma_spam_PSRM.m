function f = sigma_spam_PSRM(IMAGE, q, Prob)
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
% Contact: jan@kodovsky.com | fridrich@binghamton.edu | November 2011
%          http://dde.binghamton.edu/download/feature_extractors
% -------------------------------------------------------------------------
% Extracts 39 submodels presented in [1] -- only q=1c. All features are
% calculated in the spatial domain and are stored in a structured variable
% 'f'. For more deatils about the individual submodels, please see the
% publication [1]. Total dimensionality of all 39 submodels is 12,753.
% -------------------------------------------------------------------------
% Input:  IMAGE ... path to the image (can be JPEG)
% Output: f ...... extracted PSRM features in a structured format
% -------------------------------------------------------------------------

settings.qBins = 3;
settings.projCount = 55;
settings.seedIndex = 1;
settings.fixedSize = 8;
settings.binSize = q;

X = double(IMAGE);

f = post_processing(all1st(X,1),'f1');      % 1st order
f = post_processing(all2nd(X,1),'f2',f);    % 2nd order
f = post_processing(all3rd(X,1),'f3',f);    % 3rd order
f = post_processing(all3x3(X,1),'f3x3', f);  % 3x3
f = post_processing(all5x5(X,1),'f5x5', f); % 5x5

function RESULT = post_processing(DATA,f,RESULT)

Ss = fieldnames(DATA);
for Sid = 1:length(Ss)
    VARNAME = [f '_' Ss{Sid}];
    eval(['RESULT.' VARNAME ' = reshape(single(DATA.' Ss{Sid} '),1,[]);' ])
end

% symmetrize
L = fieldnames(RESULT);
for i=1:length(L)
    name = L{i}; % feature name
    if name(1)=='s', continue; end
    [T,N] = parse_feaname(name);
    if strcmp(T,''), continue; end
    % symmetrization
    if strcmp(N(1:3),'min') || strcmp(N(1:3),'max')
        % minmax symmetrization
        OUT = ['s' T(2:end) '_minmax' N(4:end)];
        if isfield(RESULT,OUT), continue; end
        Fmin = []; Fmax = [];
        eval(['Fmin = RESULT.' strrep(name,'max','min') ';']);
        eval(['Fmax = RESULT.' strrep(name,'min','max') ';']);
        F = mergeMinMax(Fmin, Fmax);
        eval(['RESULT.' OUT ' = single(F);' ]);
    elseif strcmp(N(1:4),'spam')
        % spam symmetrization
        OUT = ['s' T(2:end) '_' N];
        if isfield(RESULT,OUT), continue; end
        eval(['RESULT.' OUT ' = single(RESULT.' name ');' ]);
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
    [T,N] = parse_feaname(name);
    if ~strcmp(N(1:4),'spam'), continue; end
    if strcmp(T,''), continue; end
    if strcmp(N(end),'v')||(strcmp(N,'spam11')&&strcmp(T,'s5x5'))
    elseif strcmp(N(end),'h')
        % h+v union
        OUT = [T '_' N 'v'];
        if isfield(RESULT,OUT), continue; end
        name2 = strrep(name,'h','v');
        Fh = []; Fv = [];
        eval(['Fh = RESULT.' name ';']);
        eval(['Fv = RESULT.' name2 ';']);
        eval(['RESULT.' OUT ' = [Fh Fv];']);
        RESULT = rmfield(RESULT,name);
        RESULT = rmfield(RESULT,name2);
    elseif strcmp(N,'spam11')
        % KBKV creation
        OUT = ['s35_' N];
        if isfield(RESULT,OUT), continue; end
        name1 = strrep(name,'5x5','3x3');
        name2 = strrep(name,'3x3','5x5');
        if ~isfield(RESULT,name1), continue; end
        if ~isfield(RESULT,name2), continue; end
        F_KB = []; F_KV = [];
        eval(['F_KB = RESULT.' name1 ';']);
        eval(['F_KV = RESULT.' name2 ';']);
        eval(['RESULT.' OUT ' = [F_KB F_KV];']);
        RESULT = rmfield(RESULT,name1);
        RESULT = rmfield(RESULT,name2);
    end
end

end

function [T,N] = parse_feaname(name)
[T,N] = deal('');
S = strfind(name,'_'); if length(S)~=1, return; end
T = name(1:S-1);
N = name(S+1:end);

end

function g = all1st(X,q)
R = [0 0 0; 0 -1 1; 0 0 0];
U = [0 1 0; 0 -1 0; 0 0 0];
g.spam14h = reshape(ProjHistSpam(X,R,Prob,'hor',q) + ProjHistSpam(X,U,Prob,'ver',q),[],1);settings.seedIndex=settings.seedIndex+1;
g.spam14v = reshape(ProjHistSpam(X,R,Prob,'ver',q) + ProjHistSpam(X,U,Prob,'hor',q),[],1);settings.seedIndex=settings.seedIndex+1;
end

function g = all2nd(X,q)
Dh = [0 0 0; 1 -2 1; 0 0 0]/2;
Dv = [0 1 0; 0 -2 0; 0 1 0]/2;
g.spam12h = reshape(ProjHistSpam(X,Dh,Prob,'hor',q) + ProjHistSpam(X,Dv,Prob,'ver',q),[],1);settings.seedIndex=settings.seedIndex+1;
g.spam12v = reshape(ProjHistSpam(X,Dh,Prob,'ver',q) + ProjHistSpam(X,Dv,Prob,'hor',q),[],1);settings.seedIndex=settings.seedIndex+1;
end

function g = all3rd(X,q)
R = [0 0 0 0 0; 0 0 0 0 0; 0 1 -3 3 -1; 0 0 0 0 0; 0 0 0 0 0]/3;
U = [0 0 -1 0 0; 0 0 3 0 0; 0 0 -3 0 0; 0 0 1 0 0; 0 0 0 0 0]/3;
g.spam14h = reshape(ProjHistSpam(X,R,Prob,'hor',q) + ProjHistSpam(X,U,Prob,'ver',q),[],1);settings.seedIndex=settings.seedIndex+1;
g.spam14v = reshape(ProjHistSpam(X,R,Prob,'ver',q) + ProjHistSpam(X,U,Prob,'hor',q),[],1);settings.seedIndex=settings.seedIndex+1;
end

function g = all3x3(X,q)
F = [-1 2 -1; 2 -4 2; -1 2 -1]/4;
g.spam11 = reshape(ProjHistSpam(X,F,Prob,'hor',q) + ProjHistSpam(X,F,Prob,'ver',q),[],1);settings.seedIndex=settings.seedIndex+1;

[Du, Db, Dl, Dr] = deal(F);
Du(3,:) = 0;
Db(1,:) = 0;
Dl(:,3) = 0;
Dr(:,1) = 0;
g.spam14v = reshape(ProjHistSpam(X,Du,Prob,'ver',q) + ProjHistSpam(X,Db,Prob,'ver',q) + ProjHistSpam(X,Dl,Prob,'hor',q) + ProjHistSpam(X,Dr,Prob,'hor',q),[],1);settings.seedIndex=settings.seedIndex+1;
g.spam14h = reshape(ProjHistSpam(X,Du,Prob,'hor',q) + ProjHistSpam(X,Db,Prob,'hor',q) + ProjHistSpam(X,Dl,Prob,'ver',q) + ProjHistSpam(X,Dr,Prob,'ver',q),[],1);settings.seedIndex=settings.seedIndex+1;
end

function g = all5x5(X,q)
F = [-1 2 -2 2 -1; 2 -6 8 -6 2; -2 8 -12 8 -2; 2 -6 8 -6 2; -1 2 -2 2 -1]/12;
g.spam11 = reshape(ProjHistSpam(X,F,Prob,'hor',q) + ProjHistSpam(X,F,Prob,'ver',q),[],1);settings.seedIndex=settings.seedIndex+1;

[Du, Db, Dl, Dr] = deal(F);
Du(4:5,:) = 0;
Db(1:2,:) = 0;
Dl(:,4:5) = 0;
Dr(:,1:2) = 0;
g.spam14v = reshape(ProjHistSpam(X,Du,Prob,'ver',q) + ProjHistSpam(X,Db,Prob,'ver',q) + ProjHistSpam(X,Dl,Prob,'hor',q) + ProjHistSpam(X,Dr,Prob,'hor',q),[],1);settings.seedIndex=settings.seedIndex+1;
g.spam14h = reshape(ProjHistSpam(X,Du,Prob,'hor',q) + ProjHistSpam(X,Db,Prob,'hor',q) + ProjHistSpam(X,Dl,Prob,'ver',q) + ProjHistSpam(X,Dr,Prob,'ver',q),[],1);settings.seedIndex=settings.seedIndex+1;
end

function h = ProjHistSpam(X,F,Prob, type, centerVal)
    %RandStream.setGlobalStream(RandStream('mt19937ar','Seed',settings.seedIndex));
    rand('state', settings.seedIndex);
    h = zeros(settings.qBins * settings.projCount, 1);
    for projIndex = 1:settings.projCount
            Psize = randi(settings.fixedSize, 2, 1);
            
            P = randn(Psize(1), Psize(2)); 
            n = sqrt(sum(P(:).^2));
            P = P ./ n;
            
            if strcmp(type, 'ver'), P = P'; end;
            binEdges = 0:settings.qBins;
            binEdges = binEdges * settings.binSize * centerVal;
            
            proj = conv2( X, conv2( F, P ), 'valid' );
            sigma = sqrt( conv2( Prob, conv2( F, P ).^2, 'valid' ) );
            
            h_neigh = prob_hist(abs(proj(:)), sigma, binEdges); 
            
            if size(P, 2) > 1
                proj = conv2( X, conv2( F, fliplr(P) ), 'valid' );
                sigma = sqrt( conv2( Prob, conv2( F, fliplr(P) ).^2, 'valid' ) );
                h_neigh = h_neigh + prob_hist(abs(proj(:)), sigma, binEdges); 
            end
            if size(P, 1) > 1
                proj = conv2( X, conv2( F, flipud(P) ), 'valid' );
                sigma = sqrt( conv2( Prob, conv2( F, flipud(P) ).^2, 'valid' ) );
                h_neigh = h_neigh + prob_hist(abs(proj(:)), sigma, binEdges); 
            end
            if all(size(P)>1)
                proj = conv2( X, conv2( F, rot90(P, 2) ), 'valid' );
                sigma = sqrt( conv2( Prob, conv2( F, rot90(P, 2) ).^2, 'valid' ) );
                h_neigh = h_neigh + prob_hist(abs(proj(:)), sigma, binEdges); 
            end
            
            h((projIndex-1)*settings.qBins + 1:projIndex*settings.qBins, 1) = h_neigh;
    end
end

function h = prob_hist( proj, sigma, binEdges )
    h = zeros( length(binEdges)-1, 1 );
    for i = 1:length(h)
        I = (proj >= binEdges(i)) & (proj < binEdges(i+1));
        h(i) = sum( sigma(I) );
    end
end

function result = mergeMinMax(Fmin, Fmax)
    Fmin = reshape(Fmin, 2*settings.qBins, []);
    Fmin = flipud(Fmin);
    result = Fmax + Fmin(:)';
end

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

end

end
