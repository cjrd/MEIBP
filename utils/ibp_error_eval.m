function err_res = ibp_error_eval(X, ZAexp, test_mask, ZAtrue)
% function err_res = ibp_error_eval(Zexp, Aexp, test_mask, ZAtrue)
%
% calculates the l2 error of a linear Gaussian IBP model
%
% author: Colorado Reed, gmail address: colorado.j.reed

err_res = sum(sum((test_mask .* (X - ZAexp)).^2));



