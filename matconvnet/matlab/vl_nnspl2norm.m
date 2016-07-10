function y = vl_nnspl2norm(x, param, dzdy)
%VL_NNSPNORM CNN spaital normalization.
%   Y = VL_NNSPNORM(X, PARAM) computes the spatial normalization of
%   the data X with parameters PARAM = [PH PW ALPHA BETA, c]. Here PH and
%   PW define the size of the spatial neighbourhood used for
%   nomalization.
%
%   For each feature channel, the function computes the sum of squares
%   of X inside each rectangle, N2(i,j). It then divides each element
%   of X as follows:
%
%      Y(i,j) = X(i,j) / (1 + ALPHA * N2(i,j))^BETA.
%
%   DZDX = VL_NNSPNORM(X, PARAM, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


if numel(param) < 5, C = 1; else, C = param(5); end;

n2 = sum(sum(x.*x,1),2) ;
f = C + param(3) * n2 ;

if nargin <= 2 || isempty(dzdy)
  y = bsxfun(@times, f.^(-param(4)), x) ;
else
  t = sum(sum(bsxfun(@times, f.^(-param(4)-1) , dzdy .* x),1),2) ;
  y = bsxfun(@times, f.^(-param(4)), dzdy) - 2 * param(3)*param(4) * bsxfun(@times,x , t) ;
end
