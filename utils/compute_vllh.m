function vllh = compute_vllh(Xtr, model, aux)
% function vllh = compute_vllh(X, model, aux)
% compute variational likelihood component of vlb
%
% author: Colorado Reed, gmail address: colorado.j.reed
tmp1 = (model.Z * aux.Ex_A).^2;
tmp2 = model.Z * (aux.Ex_A.^2); 
if aux.has_test
    tmp1(aux.test_mask) = 0;
    tmp2(aux.test_mask) = 0;
end
vllh = 1/model.sigX^2*( sum(sum((Xtr * aux.Ex_A') .* model.Z))...
    - 0.5 * sum(sum(( model.Z .* ((1-aux.test_mask) * aux.Ex_Asq') )))...
    -1/2*(sum(tmp1(:)) - sum(tmp2(:)))...
    - 1/2*aux.trXX ) - aux.NDtrain/2*log(2*pi*model.sigX^2);