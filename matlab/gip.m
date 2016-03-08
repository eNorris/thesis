% --------------------------------------------------------------------------------------------------------------
% Description   : Group-organized cross section input program
%                 This program is crated based on the lengendary 'gip.f' 
%                 program. It reads nuclide-organized microscopic x-section
%                 data from either local data or from a file prepared by
%                 the 'NJOY' program. In photon transport, we assume there
%                 is no up-scatter. The output of gip is a 4D arry
%                 msig(enegygroup, zoneid, Pn order, 1+(cfg.ihm-cfg.ihs)). 
%                 This arry contains the x-section of (sig g, sig g->g,
%                 g-1->g, g-2->g, ..., 1->g)
% Created       : Feb 2016
% --------------------------------------------------------------------------------------------------------------

% Setup the group-organized x-section data with Pn expansion
display('Setup group-organized x-section data ...');

if(strcmp(cfg.njoy,'yes'))
    display('Sorry, still under development for NJOY processed x-section');
    return;
else
    display('Ok, read x-section locally');
    % Initialize the group organized x-section array (group, zoneid,Pn order,g'-g)
    msig = zeros(cfg.igm, round(cfg.mtm/(1+cfg.isct)), (1+cfg.isct), 2+(cfg.ihm-cfg.ihs));
    % Read x-section from cfg.xsection array
    for ii=1:cfg.igm
        for jj=1:round(cfg.mtm/(1+cfg.isct))
            for kk=1:1+cfg.isct  % Because the Pn start from 0th order
                for mm=1:2+(cfg.ihm-cfg.ihs)
                    msig(ii,jj,kk,mm) = cfg.xsection((jj-1)*(1+cfg.isct)+kk, cfg.iht+mm-1);
                end
            end
        end
    end
end

    