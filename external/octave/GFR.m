function F=GFR(IMAGE,NR,QF,channel)
% -------------------------------------------------------------------------
% Copyright (c) 2015 DDE Lab, Binghamton University, NY.
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
% Contact:     mboroum1@binghamton.edu   | July 2015
%              fridrich@binghamton.edu 
%          
%          http://dde.binghamton.edu/download/feature_extractors
% -------------------------------------------------------------------------
% This function extracts Gabor features for steganalysis of JPEG images 
% proposed in [1]. Parameters are exactly set as they are described in the
% paper. Total dimensionality of the features with 32 rotations: 17000.
% Note: Phil Sallee's Matlab JPEG toolbox(function jpeg_read)is needed for 
% running this function.
% -------------------------------------------------------------------------
% Input:  IMAGE .. path to the JPEG image
%         QF ..... JPEG quality factor (can be either 75 or 95) 
%         NR  .... number of rotations for Gabor kernel 
% Output: F ...... extracted Gabor features
% -------------------------------------------------------------------------
% [1] X. Song, F. Liu, C. Yang, X. Luo and Y. Zhang "Steganalysis of
%     Adaptive JPEG Steganography Using 2D Gabor Filters", Proceedings of
%     the 3rd ACM Workshop on Information Hiding and Multimedia Security, 
%     Pages 15-23, Portland, OR, June 2015.   
% -------------------------------------------------------------------------

I_STRUCT = jpeg_read(IMAGE);

% number of histogram bins
T = 4;
% quantization steps 
if QF==75
    q = [2 4 6 8];
elseif QF==95
    q = [0.5 1 1.5 2];
end

Rotations = (0:NR-1)*pi/NR;
sr=numel(Rotations);

% Standard deviations
Sigma = [0.5 0.75 1 1.25];
ss=numel(Sigma);

PhaseShift = [0 pi/2];
sp=numel(PhaseShift);

AspectRatio = 0.5;

% Decompress to spatial domain
% fun = @(x)x.data .*I_STRUCT.quant_tables{1};
if channel>1
    fun = @(x)x .*I_STRUCT.quant_tables{2};
else
    fun = @(x)x .*I_STRUCT.quant_tables{1};
end

I_spatial = blockproc(I_STRUCT.coef_arrays{channel},[8 8],fun);
% fun=@(x)idct2(x.data);
fun=@(x)idct2(x);
I_spatial = blockproc(I_spatial,[8 8],fun);

% Compute DCTR locations to be merged 
mergedCoordinates = cell(25, 1);
for i=1:5
    for j=1:5
        coordinates = [i,j; i,10-j; 10-i,j; 10-i,10-j];
        coordinates = coordinates(all(coordinates<9, 2), :);
        mergedCoordinates{(i-1)*5 + j} = unique(coordinates, 'rows');
    end
end

% Load Gabor Kernels
Kernel = cell(ss,sr,sp);
for S = Sigma
    for R = Rotations
        for P=PhaseShift
        Kernel{S==Sigma,R==Rotations,P==PhaseShift} = gaborkernel(S, R, P, AspectRatio);
        end
    end
end

% Compute features
modeFeaDim = numel(mergedCoordinates)*(T+1);
DimF=sp*ss*sr;
DimS=sp*ss*(sr/2+1);
FF = zeros(1, DimF*modeFeaDim, 'single');  
F = zeros(1, DimS*modeFeaDim, 'single');   

for mode_P=1:sp
    for mode_S = 1:ss  
        for mode_R = 1:sr

             modeIndex = (mode_P-1)*(sr*ss) + (mode_S-1)*sr+ mode_R;
                          
             R = conv2(I_spatial, Kernel{mode_S,mode_R,mode_P}, 'valid');
             R = abs(round(R / q(mode_S)));      
             R(R > T) = T;
            % feature extraction and merging        
            for merged_index=1:numel(mergedCoordinates)
                f_merged = zeros(1, T+1, 'single');
                for coord_index = 1:size(mergedCoordinates{merged_index}, 1);
                    r_shift = mergedCoordinates{merged_index}(coord_index, 1);
                    c_shift = mergedCoordinates{merged_index}(coord_index, 2);
                    R_sub = R(r_shift:8:end, c_shift:8:end);
                    f_merged = f_merged + hist(R_sub(:), 0:T);
                end
                F_index_from = (modeIndex-1)*modeFeaDim+(merged_index-1)*(T+1)+1;
                F_index_to = (modeIndex-1)*modeFeaDim+(merged_index-1)*(T+1)+T+1;
                FF(F_index_from:F_index_to) = f_merged / sum(f_merged);           
            end
      
        end
     % merging of symmetrical directions   
     MIndex=1:sr/2-1;
     MS=size(MIndex,2)+2;
     SI = (modeIndex-sr)*modeFeaDim;      
     
     Fout=FF(SI+1:  SI + MS* modeFeaDim );
     F_M=FF(SI+1:  SI + sr* modeFeaDim );

        for i=MIndex
           Fout(i*modeFeaDim+1:i*modeFeaDim+modeFeaDim)= ( F_M(i*modeFeaDim+1:i*modeFeaDim+modeFeaDim)+ F_M( (sr-i)*modeFeaDim+1: (sr-i)*modeFeaDim+modeFeaDim ) )/2;     %%% Needs Modification
        end
       
       ind = (mode_P-1)*ss + mode_S;
       F((ind-1)*MS*modeFeaDim+1  :(ind-1)*MS*modeFeaDim + MS*modeFeaDim )=Fout;     
       
    end
end

function kernel = gaborkernel(sigma, theta, phi, gamma)
lambda = sigma / 0.56;
gamma2 = gamma^2;
s = 1 / (2*sigma^2);
f = 2*pi/lambda;
% sampling points for Gabor function
[x,y]=meshgrid([-7/2:-1/2,1/2:7/2],[-7/2:-1/2,1/2:7/2]);
y = -y;
xp =  x * cos(theta) + y * sin(theta);
yp = -x * sin(theta) + y * cos(theta);
kernel = exp(-s*(xp.*xp + gamma2*(yp.*yp))) .* cos(f*xp + phi);
% normalization
kernel = kernel- sum(kernel(:))/sum(abs(kernel(:)))*abs(kernel);
