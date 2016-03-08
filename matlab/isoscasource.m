% --------------------------------------------------------------------------------------------------------------
% Description   : Since the source involves an integral over flux moving in
%                 other direction within the same enrgy group, iterations
%                 over source must be used. The scattering integral is
%                 evaluated based upon prior information or guesses as
%                 available, all of the fluxes for an energy group are
%                 evaluated, a new scattering integral is evaluated, and so
%                 on. 
%                 This 'isoscasource' assume the isotropic scattering. This
%                 approximation allows the source due to scattering to be
%                 expressed solely in terms of the scalar flux:
%                 S = (1/4pi)* sum(sig_s(g'-g)*flux(g))
% Input         : scalar flux of cells, and energy group number, cfg,
%                 zoneid, cross section
% 
% Output        : Scatter source in each cell 
% Created       : Mar 2016
% --------------------------------------------------------------------------------------------------------------
function isoscatter = isoscasource(inputflux, egroup, cfg, zone_id,msig)
isoscatter = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh); % Initialize scatter source 

for ieg=1:egroup
    for kk=1:cfg.zmesh            
        for jj=1:cfg.ymesh
            for ii=1:cfg.xmesh
                mid = zone_id(ii,jj,kk)+1; % Obtain mateiral ID. Note: zone id start at 0, so the array index plus 1.
                isoscatter(ii,jj,kk) = isoscatter(ii,jj,kk) + (1.0/4.0/pi)*(inputflux(ii,jj,kk,ieg).*msig(egroup,mid,1,ieg+1)); % Pn=0 
            end
        end
    end
end


