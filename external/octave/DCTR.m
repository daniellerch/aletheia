function F = DCTR(IMAGE, QF, channel)

I_STRUCT = jpeg_read(IMAGE);


%% Set parameters
% number of histogram bins
T = 4;

% compute quantization step based on quality factor
if QF<50,
    q = min(8 * (50 / QF), 100);
else
    q = max(8 * (2 - (QF/50)), 0.2);
end

%% Prepare for computing DCT bases
k=0:7;
l=0:7;
[k,l]=meshgrid(k,l);

A=0.5*cos(((2.*k+1).*l*pi)/16);
A(1,:)=A(1,:)./sqrt(2);
A=A';

%% Compute DCTR locations to be merged
mergedCoordinates = cell(25, 1);
for i=1:5
    for j=1:5
        coordinates = [i,j; i,10-j; 10-i,j; 10-i,10-j];
        coordinates = coordinates(all(coordinates<9, 2), :);
        mergedCoordinates{(i-1)*5 + j} = unique(coordinates, 'rows');
    end
end

%% Decompress to spatial domain
if channel>1
    fun = @(x)x .*I_STRUCT.quant_tables{2};
else
    fun = @(x)x .*I_STRUCT.quant_tables{1};
end

I_spatial = blockproc(I_STRUCT.coef_arrays{channel},[8 8],fun);



%fun = @(x)x.data .*I_STRUCT.quant_tables{1};
%I_spatial = blockproc(I_STRUCT.coef_arrays{1},[8 8],fun);
%fun=@(x)idct2(x.data);
fun=@(x)idct2(x);
I_spatial = blockproc(I_spatial,[8 8],fun)+128;

%% Compute features
modeFeaDim = numel(mergedCoordinates)*(T+1);
F = zeros(1, 64*modeFeaDim, 'single');
for mode_r = 1:8
    for mode_c = 1:8
        modeIndex = (mode_r-1)*8 + mode_c;
        % Get DCT base for current mode
        DCTbase = A(:,mode_r)*A(:,mode_c)';
        
        % Obtain DCT residual R by convolution between image in spatial domain and the current DCT base
        R = conv2(I_spatial-128, DCTbase, 'valid');
                
        % Quantization, rounding, absolute value, thresholding
        R = abs(round(R / q));      
        R(R > T) = T;
        
        % Core of the feature extraction
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
            F(F_index_from:F_index_to) = f_merged / sum(f_merged);
        end
    end
end

end

