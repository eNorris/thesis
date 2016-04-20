% --------------------------------------------------------------------------------------------------------------
% Description   : Generate uncollided flux and the first collision source
%                 in each cell. The unclollided flux is simply calculated
%                 based on the attenuation law through the distance between
%                 a cell and the source and the optical mean free paths.
%                 To obtain the optical mean free paths, the ray tracing
%                 from the cell to the source is used. The ray tracing
%                 algorithm is described by J. Amanatides and A. Woo
%                 (1987).
%            
%        
%                 Orentation of the space is shown below 
%                  
%                                   /z     
%                                  /_______x        
%                                  |
%                                  |
%                                 y|
%                                 
%                                    
% Created       : April 2016
% --------------------------------------------------------------------------------------------------------------

% Start the 3D uncollided flux and first collision source calculation
display('Generating uncollided flux and first collistion source in each cell ...');

e1 = 1.0e30;  % a very big number
e2 = 1.0e-30; % a very small number
e3 = 1.0e-8;
e4 = 2.5e-9;

% Initialize mesh boundaries
% Fix me: only uniform mesh size used
rp = (0.0:cfg.xlen/cfg.xmesh:cfg.xlen);           % X direction
thp = (0.0:cfg.ylen/cfg.ymesh:cfg.ylen);          % Y direction
zp = (0.0:cfg.zlen/cfg.zmesh:cfg.zlen);           % Z direction

% Initialize uncollided flux array

uflux = zeros(cfg.xmesh,cfg.ymesh,cfg.zmesh,cfg.igm);

% Loop over the 3D cells
for kk=1:cfg.zmesh
    for jj=1:cfg.ymesh
        for ii=1:cfg.xmesh  
            for isp=1:cfg.source_number % Loop over source positions
                kc = kk;       % Initialize cell index in Z direction
                zc = (kk-0.5)*(cfg.zlen/cfg.zmesh); % Convert cell index to real position
                jc = jj;       % Initialize cell index in Y direction
                yc = (jj-0.5)*(cfg.ylen/cfg.ymesh);
                ic = ii;       % Initialize cell index in X direction
                xc = (ii-0.5)*(cfg.xlen/cfg.xmesh);
                
                xs = sx(isp)*(cfg.xlen/cfg.xmesh)-cfg.xlen/cfg.xmesh/2.0; % Convert source position index to real position
                ys = sy(isp)*(cfg.ylen/cfg.ymesh)-cfg.ylen/cfg.ymesh/2.0;
                zs = sz(isp)*(cfg.zlen/cfg.zmesh)-cfg.zlen/cfg.zmesh/2.0;
                
                % Determine distance and ray direction cosines from source
                % point to mesh point
                [dist,ds2,rmu,psi,eta] = dsqrd(xc,yc,zc,xs,ys,zs);
                
                rmu = -rmu; % Reverse ray direction cosines from mesh cell to source
                psi = -psi;
                eta = -eta;
                
                % Determine mesh cell boundary in each direction 
                if(rmu>0) 
                    icb = ic+1;
                else 
                    icb = ic;
                end
                if(psi>0)
                    jcb = jc+1;
                else
                    jcb = jc;
                end
                if(eta>0)
                    kcb = kc+1;
                else
                    kcb = kc;
                end
                
                % Performe the ray-tracing based on Amanatides-Woo
                % algorithm (1987)
                dmfp = zeros(cfg.igm);  % Initialize  mean free path array 
                mchk = 1;
                while mchk>0
                    %Determine distances to cell boundaries
                    if(abs(rmu)<e2)  % rmu~0 means ray is perpedicular to x axis
                        dx = e1;     % assign a very large number means infinity
                    else
                        dx = (rp(icb)-xc)/rmu;
                    end
                    if(abs(psi)<e2)  % psi~0 means ray is perpedicular to y axis
                        dy = e1;     % assign a very large number means infinity
                    else
                        dy = (thp(jcb)-yc)/psi;
                    end
                    if(abs(eta)<e2)  % eta~0 means ray is perpedicular to z axis
                        dz = e1;     % assign a very large number means infinity
                    else
                        dz = (zp(kcb)-zc)/eta;
                    end
                    
                    % Determine shortest distance in three directions
                    if(dx<dy && dx<dz)
                        dis = dx;
                        idx = 1;
                    elseif(dy<dz)
                        dis = dy;
                        idx =2;
                    else
                        dis = dz;
                        idx =3;
                    end
                    
                    % Calculate distance from cell to source
                    csds = (xc-xs)*(xc-xs);
                    csds = csds+(yc-ys)*(yc-ys);
                    csds = csds+(zc-zs)*(zc-zs);
                    csds = sqrt(csds);
                    if(csds<dis)  % Determine if cell is source cell
                        dis = csds;
                        mchk = 0;
                        idx = 0;
                    end
                    
                    % Update mean free path array
                    zid = zone_id(ic,jc,kc)+1;  % Note: zone id start at 0, so the index plus 1.
                    for ig=1:cfg.igm  % Loop over energy group
                        dmfp(ig) = dmfp(ig)+dis*msig(ig,zid,1,1); % Macroscopic cross section 
                    end
                                
                    % Update cell indices and positions
                    if(idx==1)
                        xc = rp(icb);
                        yc = yc+dis*psi;
                        zc = zc+dis*eta;
                        if(rmu>0)
                            ic = ic+1;
                            icb = ic+1;
                        else
                            ic = ic-1;
                            icb = ic;
                        end
                        if(ic<1 || ic>cfg.xmesh)
                            mchk = 0;
                        end
                    elseif(idx==2)
                        xc = xc+dis*rmu;
                        yc = thp(jcb);
                        zc = zc+dis*eta;
                        if(psi>0)
                            jc = jc+1;
                            jcb = jc+1;
                        else
                            jc = jc-1;
                            jcb = jc;
                        end
                        if(jc<1 || jc>cfg.ymesh)
                            mchk = 0;
                        end
                    elseif(idx==3)
                        xc = xc+dis*rmu;
                        yc = yc+dis*psi;
                        zc = zp(kcb);
                        if(eta>0) 
                            kc = kc+1;
                            kcb = kc+1;
                        else
                            kc = kc-1;
                            kcb = kc;
                        end
                        if(kc<1 || kc>cfg.zmesh)
                            mchk = 0;
                        end
                    end                
                end % End of while loop (i.e. finsih ray tracing from a cell to source)
                
                % Calculate uncollided flux in cell 
                for ig=1:cfg.igm
                    uflux(ii,jj,kk,ig) = cfg.source_activity(ig).*exp(-dmfp(ig))/ds2;    
                end               
                % Force the uncollided flux in the source cell to be equal
                % to the source activity
                uflux(sx(isp),sy(isp),sz(isp),ig) = cfg.source_activity(ig);     
                
            end % End of source loop
        end % End of X direction
    end % End of Y direction
end % End of Z direction

figure;
imagesc(log10(uflux(:,:,15,1))); 
caxis([-10, 6]);
title(['Uncollided Flux']); 
grid on; 
axis equal;
colorbar;
        
        
        
                