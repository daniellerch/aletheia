function [STEGO,embedding_efficiency,changes] = NSF5_COLOR(cover, payload, stego)
% -------------------------------------------------------------------------
% Contact: jan@kodovsky.com | June 2011
% -------------------------------------------------------------------------
% This program simulates the embedding impact of the steganographic
% algorithm nsF5 [1] as if the best possible coding was used. Please, visit
% the webpage http://dde.binghamton.edu/download/nsf5simulator for more
% information.
% -------------------------------------------------------------------------
% Input:
%  COVER - cover image (grayscale JPEG image)
%  STEGO - resulting stego image that will be created
%  ALPHA - relative payload in terms of bits per nonzero AC DCT coefficient
%  SEED - PRNG seed for the random walk over the coefficients
% Output:
%  nzAC - number of nonzero AC DCT coefficients in the cover image
%  embedding_efficiency - bound on embedding efficiency used for simulation
%  changes - number of changes made
% -------------------------------------------------------------------------
% References:
% [1] J. Fridrich, T. Pevny, and J. Kodovsky, Statistically undetectable
%     JPEG steganography: Dead ends, challenges, and opportunities. In J.
%     Dittmann and J. Fridrich, editors, Proceedings of the 9th ACM
%     Multimedia & Security Workshop, pages 3-14, Dallas, TX, September
%     20-21, 2007.
% -------------------------------------------------------------------------
% Note: The program requires Phil Sallee's MATLAB JPEG toolbox available at
% http://www.philsallee.com/
% -------------------------------------------------------------------------

%%% load the cover image
COVER=jpeg_read(cover);
STEGO=COVER;
for index_color=1:3
	DCT=COVER.coef_arrays{index_color};
	ALPHA=1;
	if ALPHA>0
	    %%% embedding simulation
        embedding_efficiency = payload/invH(payload);  % bound on embedding efficiency
	    nAC = nnz(DCT); % number of nonzero AC DCT coefficients
	    nzAC = nnz(DCT)-nnz(DCT(1:8:end,1:8:end)); % number of nonzero AC DCT coefficients
        changes = ceil(payload*nzAC/embedding_efficiency); % number of changes nsF5 would make on bound
	    ALPHA=changes/nzAC;
	    embedding_efficiency = ALPHA/invH(ALPHA); % bound on embedding efficiency
	    changeable = (DCT~=0); % mask of all nonzero DCT coefficients in the image
	    changeable(1:8:end,1:8:end) = false; % do not embed into DC modes
	    changeable = find(changeable); % indexes of the changeable coefficients
	    % rand('state',SEED); % initialize PRNG using given SEED
	    changeable = changeable(randperm(nzAC)); % create a pseudorandom walk over nonzero AC coefficients
	    to_be_changed = changeable(1:changes); % coefficients to be changed
	    DCT(to_be_changed) = DCT(to_be_changed)-sign(DCT(to_be_changed)); % decrease the absolute value of the coefficients to be changed
	end
	STEGO.coef_arrays{index_color}=DCT;

end

try
    jpeg_write(STEGO, stego);
catch
    error('ERROR (problem with saving the stego image)')
end



	
function res = invH(y)
% inverse of the binary entropy function
to_minimize = @(x) (H(x)-y)^2;
res = fminbnd(to_minimize,eps,0.5-eps);

function res = H(x)
% binary entropy function
res = -x*log2(x)-(1-x)*log2(1-x);
