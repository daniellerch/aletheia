function beta_hat = SP(path, channel)
%
% Sample Pairs Analysis. Original Dumitrescu version as described in:
% Sorina Dumitrescu, Xiaolin Wu, Nasir D. Memon: "On steganalysis of random
% LSB embedding in continuous-tone images." In: Proc. of ICIP, 2002,
% Rochester, NY, pp.641~644.
%
% 2011 Copyright by Jessica Fridrich, fridrich@binghamton.edu,
% http:\\ws.binghamton.edu\fridrich


X = double(imread(path));
if(size(size(X))==2)
    X=X(:,:,channel)
end




% X = double(X);
[M,N] = size(X);

u = X(:,1:N-1);  % Only horizontal pairs are considered
v = X(:,2:N);

% X = { (u,v) | v even u<v or v odd u>v}
% Z = { (u,v) | u = v}
% W = { (u,v) | (u = 2k and v = 2k+1) or (u = 2k+1 and v = 2k)}
% the rest is V

Xc = length(find( (mod(v(:),2)==0 & u(:) < v(:)) | (mod(v(:),2)==1 & u(:) > v(:))));
Zc = length(find( u(:) == v(:) ));
Wc = length(find(floor(u(:)/2) == floor(v(:)/2) & u(:)~=v(:) ));
Vc = M*(N-1) - (Xc + Zc + Wc);

a = (Wc + Zc)/2;
b = 2*Xc - M*(N-1);
c = Vc + Wc - Xc;

D = b^2-4*a*c;
if a > 0
    p1 = (-b+sqrt(D))/(2*a);
    p2 = (-b-sqrt(D))/(2*a);
else
    p1 = -1; p2 = -1;
end

beta_hat = real(min(p1,p2));
