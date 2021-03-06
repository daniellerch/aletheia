function beta_hat = TRIPLES(path, channel)
%
% Triples message length estimator utilizing only the symmetry e_2m+1,2n+1
% = o_2m+1,2n+1 with m,n \in {-5,...,5}.
% This implementation follows the method description in:
% Andrew D. Ker: "A general framework for the structural steganalysis of 
% LSB replacement." In: Proc. 7th Information Hiding Workshop, LNCS vol. 
% 3727, pp. 296-311, Barcelona, Spain. Springer-Verlag, 2005.
%
% 2011 Copyright by Jessica Fridrich, fridrich@binghamton.edu,
% http:\\ws.binghamton.edu\fridrich
%

X = double(imread(path));
if(size(size(X))!=2)
    X=X(:,:,channel);
end

[M N] = size(X);
X  = double(X);
I  = 1 : M;
J  = 1 : N-2;
e  = zeros(25,25); o  = e;
c0 = zeros(11,11); c1 = c0; c2 = c0; c3 = c0;

L = X(I,J); C = X(I,J+1); R = X(I,J+2);  % Only row triples are considered
T1 = C(:) - L(:);  % s2 - s1
T2 = R(:) - C(:);  % s3 - s2

for m1 = -11 : 13   % We need subscripts 2n-1...2n+3 and the same for 2m-1...2m+3, which means m1,m2 \in {-11...13}
    for m2 = -11 : 13
        i1 = find(T1 == m1);
        i2 = i1(T2(i1) == m2);
        e(m1+12,m2+12) = sum(1 - mod(L(i2),2)); % (s1..s3), s2=s1+m1, s3=s2+m2, s1 even
        o(m1+12,m2+12) = sum(mod(L(i2),2));     % (s1..s3), s2=s1+m1, s3=s2+m2, s1 odd
    end
end

m = -5 : 5;
n = -5 : 5;
d0 = e(2*m+1+12,2*n+1+12)-o(2*m+1+12,2*n+1+12);
d1 = e(2*m+1+12,2*n+2+12)+e(2*m+12,2*n+2+12)+o(2*m+12,2*n+1+12)-o(2*m+1+12,2*n+12)-o(2*m+2+12,2*n+12)-e(2*m+2+12,2*n+1+12);
d2 = e(2*m+12,2*n+3+12)+o(2*m-1+12,2*n+2+12)+o(2*m+12,2*n+2+12)-o(2*m+2+12,2*n-1+12)-e(2*m+2+12,2*n+12)-e(2*m+3+12,2*n+12);
d3 = o(2*m-1+12,2*n+3+12)-e(2*m+3+12,2*n-1+12);

c0 = d0 + d1 + d2 + d3;
c1 = 3*d0 + d1 - d2 - 3*d3;
c2 = 3*d0 - d1 - d2 + 3*d3;
c3 = d0 - d1 + d2 - d3;

% Solving the quintic
epsilon = 0.000000001;
Left  = 0.6; Right = 7;
FL = quintic(Left,c0,c1,c2,c3);
FR = quintic(Right,c0,c1,c2,c3);
accuracy = 1;

if FL * FR > 0
    beta_hat = -1;
else  % Binary search
    while accuracy > epsilon
        y = (Left + Right) / 2;
        Fy = quintic(y,c0,c1,c2,c3);
        if Fy * FL <= 0
            Right = y;
            FR = Fy;
        else
            Left = y;
            FL = Fy;
        end
        accuracy = abs(Right - Left);
    end
end

q_hat = (Left + Right) / 2;
beta_hat = 1/2*(1 - 1/q_hat);

% t = 0.5:0.01:7;
% y = quintic(t,c0,c1,c2,c3);
% plot(t,y,'k')


function y = quintic(q,c0,c1,c2,c3)
% q can be a vector

[Cm,Cn] = size(c0);
y = 0;

for m = 1 : Cm
    for n = 1 : Cn 
        y = y + 2*c0(m,n)*c1(m,n) + q*(4*c0(m,n)*c2(m,n) + 2*c1(m,n)^2);
        y = y + q.^2*(6*c0(m,n)*c3(m,n) + 6*c1(m,n)*c2(m,n));
        y = y + q.^3*(4*c2(m,n)^2 + 8*c1(m,n)*c3(m,n)) + q.^4*10*c2(m,n)*c3(m,n) + q.^5*6*c3(m,n)^2;
    end
end
