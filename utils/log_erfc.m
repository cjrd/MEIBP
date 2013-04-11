function res = log_erfc(x)
% function res = log_erfc(x)
% implementation of Cody's algorithm
% this exactly implements his method from Math Comp 1969
%
% input
% x = array of positive values (at first)
% output
% res = log of erfc values at these points
% function res = log_cody(x)

% initialize res
res = zeros(size(x));

% first process the x values >= 0
index_pos = find(x>=0);
xpos = x(index_pos);
y = log_cody_pos(xpos);
res(index_pos) = y;

% then process the negative values
index_neg = find( x < 0);
xneg = x(index_neg);
y = log_cody_pos(-xneg);

% now convert this to erfc, subtract from 2 and take log
res(index_neg) = log(2- exp(y));

return

% subfunction to calculate log(erfc(x)) for x>=0
function res = log_cody_pos(x)

% set up coeff data - first for erf with |x| < 0.5
% n = 4 coeffs
% all these coeffs start with constant term
plow = [3.209377589138469e+3, 3.774852376853020e+2,...
    1.138641541510501e+2, ...
    3.161123743870565e+0, 1.857777061846031e-1];
qlow = [2.844236833439170e+3, 1.282616526077372e+3,...
    2.44024637934441e+2, ...
    2.360129095234412e+1, 1];

plow = fliplr(plow); qlow = fliplr(qlow);

% set up coeffs for erfc for 0.5 <= x <= 4
% n = 7 coeffs
pmid = [3.004592610201616e+2, 4.519189537118729e+2,...
    3.393208167343436e+2, ...
    1.529892850469404e+2, 4.316222722205673e+1,...
    7.211758250883093e+0, ...
    5.641955174789739e-1, -1.368648573827167e-7];

qmid = [3.004592609569832e+2, 7.909509253278980e+2,...
    9.313540948506096e+2, ...
    6.389802644656311e+2, 2.775854447439876e+2,...
    7.700015293522947e+1, ...
    1.278272731962942e+1, 1];
pmid = fliplr(pmid); qmid = fliplr(qmid);

% set up coeffs for erc for x>=4, this is n = 5 set
phigh = [-6.587491615298378e-4, -1.608378514874227e-2,...
    -1.257817261112292e-1, ...
    -3.603448999498044e-1, -3.053266349612323e-1,...
    -1.631538713730209e-2];
qhigh = [2.335204976268692e-3, 6.051834131244132e-2,...
    5.279051029514284e-1, ...
    1.872952849923461, 2.568520192289822, 1];
phigh = fliplr(phigh); qhigh = fliplr(qhigh);

% first find x values between 0 and +0.5
index_low = find(x < 0.5);
xlow = x(index_low); u = xlow.^2;
y = xlow.*polyval(plow,u)./polyval(qlow,u);

% now go from y = erf to y = log of erfc here
res(index_low) = log(1-y);

% then find positive x values between 0.5 and 4
index_mid = find( (x >= 0.5) & ( x<=4));
xmid = x(index_mid);
temp = polyval(pmid,xmid)./polyval(qmid,xmid);
res(index_mid) = -xmid.^2 + log(temp);

% finally do the values above 4
index_high = find(x>4);
xhigh = x(index_high);
u = 1./(xhigh); u = u.^2;

temp = polyval(phigh,u)./polyval(qhigh,u);
temp = 1/sqrt(pi) + temp.*u;
temp = log(temp) -xhigh.^2 - log(xhigh);

res(index_high) = temp;

return