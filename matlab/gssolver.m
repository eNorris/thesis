% --------------------------------------------------------------------------------------------------------------
% Description   : Iterative solve Sn using Gauss-Seidel scheme
%                 The 3D cartisan transport equation is given as following
%                 mu*[phi_i+1/2-phi_i-1/2] + eta*[phi_j+1/2-phi_j-1/2] +
%                 szi*[phi_k+1/2-phi_k-1/2] + V*sigma_t*phi_i,j,k =
%                 V*S_i,j,k
%                 with "Diamond" difference model, the average flux of a
%                 cell is phi_i,j,k = [V*S_i,j,k +
%                 2mu*A*phi_i_in+2eta*B*phi_j_in+2*czi*C*phi_k_in]/[V*sigma_t+
%                 2mu*A+2eta*B+2czi*C]
%                 where V is the cell volume, S is the source in cell
%                 (i,j,k). A, B, C are face area of the cell.
%                 The 'direction-space' sweep process. For each direction,
%                 angluar flux is calculated for each cell. Until all
%                 direction finished, the cell average flux is calcualted
%                 and compared to the previous flux value. If converged,
%                 move to the next energy group unitll go through all the 
%                 energy group. 
%  
%                 Since the source involves an integral over flux moving in
%                 other direction within the same enrgy group, iterations
%                 over source must be used. The scattering integral is
%                 evaluated based upon prior information or guesses as
%                 available, all of the fluxes for an energy group are
%                 evaluated, a new scattering integral is evaluated, and so
%                 on. The iterations are deemed to have "converged" when
%                 the fluxes from two or more successive iterations are
%                 sufficiently close to each other. Problems that have no
%                 fission and no "upscatter" can be solved with one pass
%                 through the energys, and it is inefficient to do more. 
%
% Created       : Mar 2016
% --------------------------------------------------------------------------------------------------------------

% Start the iteratie solving 
display('Solving photon transport using Gauss-Seidel ...');

% Initialize the multigroup flux and angular flux with zeros
flux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh, cfg.igm);  % Scalar flux
angflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh, cfg.m, cfg.igm); % Angular flux
tempflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh); % temporary flux array

% Initialize the multigroup source with zeros except the point source 
extsource = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh, cfg.igm);
extsource((cfg.xmesh-1)/2+1, cfg.col_ylen/2, (cfg.zmesh-1)/2+1, cfg.igm) = 1.0e6;   % Fixed one-group point source 
%extsource((cfg.xmesh-1)/2+1, (cfg.ymesh-1)/2+1, (cfg.zmesh-1)/2+1, cfg.igm) = 1.0e6;   % Fixed one-group point source 

% % Setup vaccum boundary 
% Initialization, fluxi, fluxj, fluxk are outward flux crossing the cell
%fluxj=zeros(cfg.xmesh,1);
%fluxk=zeros(cfg.xmesh,cfg.ymesh);
%fluxj_pre=zeros(cfg.xmesh,1);
%fluxk_pre=zeros(cfg.xmesh,cfg.ymesh);

xinflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh);
xoutflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh);
yinflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh);
youtflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh);
zinflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh);
zoutflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh);

% Plot the quadrature
figure();
x = zeros(2, length(wt));
y = zeros(2, length(wt));
z = zeros(2, length(wt));
x(2,:) = emu;
y(2,:) = eta;
z(2,:) = xzi;
plot3(x, y, z, 'r');
hold on;
plot3([0; 1], [0; 0], [0; 0], 'k');
plot3([0; 0], [0; 1], [0; 0], 'k');
plot3([0; 0], [0; 0], [0; 1], 'k');
xlabel('x');
ylabel('y');
zlabel('z');
hold off;
title('Quadrature');

figure();
l = length(wt)/8;
x = zeros(2, l);
y = zeros(2, l);
z = zeros(2, l);
ll = (emu>0) & (eta > 0) & (xzi > 0);
x(2,:) = emu(ll);
y(2,:) = eta(ll);
z(2,:) = xzi(ll);
plot3(x, y, z, 'r');
hold on;
plot3([0; 1], [0; 0], [0; 0], 'k');
plot3([0; 0], [0; 1], [0; 0], 'k');
plot3([0; 0], [0; 0], [0; 1], 'k');
xlabel('x');
ylabel('y');
zlabel('z');
hold off;
title('Quadrature - Octant 1');

tic;  % Start the timer
% Start the source iteration
num_iter = 1;
maxdiff = 1.0;
for ieg=1:cfg.igm  % Loop over energy groups
    disp(['Energy group #', num2str(ieg)]);
    while (num_iter<=cfg.maxit && maxdiff>cfg.epsi) 
        figure;
        imagesc(log10(flux(:,:,15,ieg))); 
        caxis([-10, 6]);
        title(['Energy Group #',num2str(ieg),' Iteration #',num2str(num_iter)]); 
        grid on; 
        axis equal;
        colorbar;
        preflux = tempflux;  % Store the flux for the previous iteration result
        totsource = extsource(:,:,:,ieg)+isoscasource(flux,ieg,cfg,zone_id, msig); % Calculate source contributed by scatter
        tempflux = zeros(cfg.xmesh,cfg.ymesh, cfg.zmesh); % Clear flux content for new sweep
        for iang=1:cfg.m   % Loop over angles
          disp(['Angle group #', num2str(iang)]);
          
          
          if(emu(iang)<0 && eta(iang)<0 && xzi(iang)<0) % Octant #1
              for kk=cfg.zmesh:-1:1       % Sweep through the 3D mesh from corner       
                  for jj=cfg.ymesh:-1:1
                      for ii=cfg.xmesh:-1:1
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==cfg.xmesh) % At boundary setup vaccum boundary condition 
                              xinflux(ii,jj,kk) = 0;
                          else
                              xinflux(ii,jj,kk) = xoutflux(ii+1,jj,kk);
                              %if xinflux(ii,jj,kk) < 0
                              %    xinflux(ii,jj,kk) = 0;
                              %    angflux(ii,jj,kk,iang,ieg) = 0.5*outflux(ii-1,jj,kk);
                              %end
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              yinflux(ii,jj,kk) = 0;
                          else
                              yinflux(ii,jj,kk) = youtflux(ii,jj+1,kk);
                              %if xinflux(ii,jj,kk) < 0
                              %    xinflux(ii,jj,kk) = 0;
                              %    angflux(ii,jj,kk,iang,ieg) = 0.5*outflux(ii-1,jj,kk);
                              %end
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              %fluxk(ii,jj) = 0;    % fluxk is the inward flux in z direction
                              %fluxk_pre(ii,jj) =0;
                              zinflux(ii,jj,kk) = 0;
                          else
                              zinflux(ii,jj,kk) = zoutflux(ii,jj,kk+1);
                              %fluxk(ii,jj) = 2*angflux(ii,jj,kk,iang,ieg)-fluxk_pre(ii,jj);  %2*angflux(ii,jj,kk+1,iang,ieg)-fluxk_pre(ii,jj);
                              %if fluxk(ii,jj)<0
                              %    fluxk(ii,jj) = 0.0;
                              %    angflux(ii,jj,kk+1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              %end
                              %fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          
                          % Calculate the angular and total flux
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk) + ...
                                                        2*DA(jj,kk,iang)*xinflux(ii,jj,kk) + ...
                                                        2*DB(ii,kk,iang)*yinflux(ii,jj,kk) + ...
                                                        2*DC(ii,jj,iang)*zinflux(ii,jj,kk)) / ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                               
                          % Now we calculate the outgoing flux 
                          xoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - xinflux(ii,jj,kk);
                          youtflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - yinflux(ii,jj,kk);
                          zoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - zinflux(ii,jj,kk);
                          
                          % Flux fixup
                          if xoutflux(ii,jj,kk) < 0
                              xoutflux(ii,jj,kk) = 0;
                          end
                          if youtflux(ii,jj,kk) < 0
                              youtflux(ii,jj,kk) = 0;
                          end
                          if zoutflux(ii,jj,kk) < 0
                              zoutflux(ii,jj,kk) = 0;
                          end
                          
                          angflux(ii,jj,kk,iang,ieg) = (xinflux(ii,jj,kk) + yinflux(ii,jj,kk) + zinflux(ii,jj,kk) + ...
                                                        xoutflux(ii,jj,kk) + youtflux(ii,jj,kk) + zoutflux(ii,jj,kk))/6;
                               
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #1
             
          
          if(emu(iang)>0 && eta(iang)<0 && xzi(iang)<0) % Octant #2
              for kk=cfg.zmesh:-1:1       % Sweep through the 3D mesh from corner       
                  for jj=cfg.ymesh:-1:1
                      for ii=1:1:cfg.xmesh
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==1) % At boundary setup vaccum boundary condition 
                              xinflux(ii,jj,kk) = 0;
                          else
                              xinflux(ii,jj,kk) = xoutflux(ii-1,jj,kk);
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              yinflux(ii,jj,kk) = 0;
                          else
                              yinflux(ii,jj,kk) = youtflux(ii,jj+1,kk);
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              zinflux(ii,jj,kk) = 0;
                          else
                              zinflux(ii,jj,kk) = zoutflux(ii,jj,kk+1);
                          end
                          
                          % Calculate the angular and total flux
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk) + ...
                                                        2*DA(jj,kk,iang)*xinflux(ii,jj,kk) + ...
                                                        2*DB(ii,kk,iang)*yinflux(ii,jj,kk) + ...
                                                        2*DC(ii,jj,iang)*zinflux(ii,jj,kk)) / ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                               
                          % Now we calculate the outgoing flux 
                          xoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - xinflux(ii,jj,kk);
                          youtflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - yinflux(ii,jj,kk);
                          zoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - zinflux(ii,jj,kk);
                          
                          % Flux fixup
                          if xoutflux(ii,jj,kk) < 0
                              xoutflux(ii,jj,kk) = 0;
                          end
                          if youtflux(ii,jj,kk) < 0
                              youtflux(ii,jj,kk) = 0;
                          end
                          if zoutflux(ii,jj,kk) < 0
                              zoutflux(ii,jj,kk) = 0;
                          end
                          
                          angflux(ii,jj,kk,iang,ieg) = (xinflux(ii,jj,kk) + yinflux(ii,jj,kk) + zinflux(ii,jj,kk) + ...
                                                        xoutflux(ii,jj,kk) + youtflux(ii,jj,kk) + zoutflux(ii,jj,kk))/6;
                               
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #2
          
          if(emu(iang)<0 && eta(iang)>0 && xzi(iang)<0) % Octant #3
              for kk=1:1:cfg.zmesh       % Sweep through the 3D mesh from corner       
                  for jj=cfg.ymesh:-1:1
                      for ii=cfg.xmesh:-1:1
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==cfg.xmesh) % At boundary setup vaccum boundary condition 
                              xinflux(ii,jj,kk) = 0;
                          else
                              xinflux(ii,jj,kk) = xoutflux(ii+1,jj,kk);
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              yinflux(ii,jj,kk) = 0;
                          else
                              yinflux(ii,jj,kk) = youtflux(ii,jj+1,kk);
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              zinflux(ii,jj,kk) = 0;
                          else
                              zinflux(ii,jj,kk) = zoutflux(ii,jj,kk-1);
                          end
                          
                          % Calculate the angular and total flux
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk) + ...
                                                        2*DA(jj,kk,iang)*xinflux(ii,jj,kk) + ...
                                                        2*DB(ii,kk,iang)*yinflux(ii,jj,kk) + ...
                                                        2*DC(ii,jj,iang)*zinflux(ii,jj,kk)) / ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                               
                          % Now we calculate the outgoing flux 
                          xoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - xinflux(ii,jj,kk);
                          youtflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - yinflux(ii,jj,kk);
                          zoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - zinflux(ii,jj,kk);
                          
                          % Flux fixup
                          if xoutflux(ii,jj,kk) < 0
                              xoutflux(ii,jj,kk) = 0;
                          end
                          if youtflux(ii,jj,kk) < 0
                              youtflux(ii,jj,kk) = 0;
                          end
                          if zoutflux(ii,jj,kk) < 0
                              zoutflux(ii,jj,kk) = 0;
                          end
                          
                          angflux(ii,jj,kk,iang,ieg) = (xinflux(ii,jj,kk) + yinflux(ii,jj,kk) + zinflux(ii,jj,kk) + ...
                                                        xoutflux(ii,jj,kk) + youtflux(ii,jj,kk) + zoutflux(ii,jj,kk))/6;
                               
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #3
          
   
          if(emu(iang)>0 && eta(iang)>0 && xzi(iang)<0) % Octant #4
              for kk=1:1:cfg.zmesh       % Sweep through the 3D mesh from corner       
                  for jj=cfg.ymesh:-1:1
                      for ii=1:1:cfg.xmesh
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==1) % At boundary setup vaccum boundary condition 
                              xinflux(ii,jj,kk) = 0;
                          else
                              xinflux(ii,jj,kk) = xoutflux(ii-1,jj,kk);
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              yinflux(ii,jj,kk) = 0;
                          else
                              yinflux(ii,jj,kk) = youtflux(ii,jj+1,kk);
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              zinflux(ii,jj,kk) = 0;
                          else
                              zinflux(ii,jj,kk) = zoutflux(ii,jj,kk-1);
                          end
                          
                          % Calculate the angular and total flux
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk) + ...
                                                        2*DA(jj,kk,iang)*xinflux(ii,jj,kk) + ...
                                                        2*DB(ii,kk,iang)*yinflux(ii,jj,kk) + ...
                                                        2*DC(ii,jj,iang)*zinflux(ii,jj,kk)) / ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                               
                          % Now we calculate the outgoing flux 
                          xoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - xinflux(ii,jj,kk);
                          youtflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - yinflux(ii,jj,kk);
                          zoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - zinflux(ii,jj,kk);
                          
                          % Flux fixup
                          if xoutflux(ii,jj,kk) < 0
                              xoutflux(ii,jj,kk) = 0;
                          end
                          if youtflux(ii,jj,kk) < 0
                              youtflux(ii,jj,kk) = 0;
                          end
                          if zoutflux(ii,jj,kk) < 0
                              zoutflux(ii,jj,kk) = 0;
                          end
                          
                          angflux(ii,jj,kk,iang,ieg) = (xinflux(ii,jj,kk) + yinflux(ii,jj,kk) + zinflux(ii,jj,kk) + ...
                                                        xoutflux(ii,jj,kk) + youtflux(ii,jj,kk) + zoutflux(ii,jj,kk))/6;
                               
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #4         
          
          if(emu(iang)<0 && eta(iang)<0 && xzi(iang)>0) % Octant #5
              for kk=cfg.zmesh:-1:1       % Sweep through the 3D mesh from corner       
                  for jj=1:1:cfg.ymesh
                      for ii=cfg.xmesh:-1:1
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==cfg.xmesh) % At boundary setup vaccum boundary condition 
                              xinflux(ii,jj,kk) = 0;
                          else
                              xinflux(ii,jj,kk) = xoutflux(ii+1,jj,kk);
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              yinflux(ii,jj,kk) = 0;
                          else
                              yinflux(ii,jj,kk) = youtflux(ii,jj-1,kk);
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              zinflux(ii,jj,kk) = 0;
                          else
                              zinflux(ii,jj,kk) = zoutflux(ii,jj,kk+1);
                          end
                          
                          % Calculate the angular and total flux
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk) + ...
                                                        2*DA(jj,kk,iang)*xinflux(ii,jj,kk) + ...
                                                        2*DB(ii,kk,iang)*yinflux(ii,jj,kk) + ...
                                                        2*DC(ii,jj,iang)*zinflux(ii,jj,kk)) / ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                               
                          % Now we calculate the outgoing flux 
                          xoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - xinflux(ii,jj,kk);
                          youtflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - yinflux(ii,jj,kk);
                          zoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - zinflux(ii,jj,kk);
                          
                          % Flux fixup
                          if xoutflux(ii,jj,kk) < 0
                              xoutflux(ii,jj,kk) = 0;
                          end
                          if youtflux(ii,jj,kk) < 0
                              youtflux(ii,jj,kk) = 0;
                          end
                          if zoutflux(ii,jj,kk) < 0
                              zoutflux(ii,jj,kk) = 0;
                          end
                          
                          angflux(ii,jj,kk,iang,ieg) = (xinflux(ii,jj,kk) + yinflux(ii,jj,kk) + zinflux(ii,jj,kk) + ...
                                                        xoutflux(ii,jj,kk) + youtflux(ii,jj,kk) + zoutflux(ii,jj,kk))/6;
                               
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #5       
          
          if(emu(iang)>0 && eta(iang)<0 && xzi(iang)>0) % Octant #6
              for kk=cfg.zmesh:-1:1       % Sweep through the 3D mesh from corner       
                  for jj=1:1:cfg.ymesh
                      for ii=1:1:cfg.xmesh
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==1) % At boundary setup vaccum boundary condition 
                              xinflux(ii,jj,kk) = 0;
                          else
                              xinflux(ii,jj,kk) = xoutflux(ii-1,jj,kk);
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              yinflux(ii,jj,kk) = 0;
                          else
                              yinflux(ii,jj,kk) = youtflux(ii,jj-1,kk);
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              zinflux(ii,jj,kk) = 0;
                          else
                              zinflux(ii,jj,kk) = zoutflux(ii,jj,kk+1);
                          end
                          
                          % Calculate the angular and total flux
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk) + ...
                                                        2*DA(jj,kk,iang)*xinflux(ii,jj,kk) + ...
                                                        2*DB(ii,kk,iang)*yinflux(ii,jj,kk) + ...
                                                        2*DC(ii,jj,iang)*zinflux(ii,jj,kk)) / ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                               
                          % Now we calculate the outgoing flux 
                          xoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - xinflux(ii,jj,kk);
                          youtflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - yinflux(ii,jj,kk);
                          zoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - zinflux(ii,jj,kk);
                          
                          % Flux fixup
                          if xoutflux(ii,jj,kk) < 0
                              xoutflux(ii,jj,kk) = 0;
                          end
                          if youtflux(ii,jj,kk) < 0
                              youtflux(ii,jj,kk) = 0;
                          end
                          if zoutflux(ii,jj,kk) < 0
                              zoutflux(ii,jj,kk) = 0;
                          end
                          
                          angflux(ii,jj,kk,iang,ieg) = (xinflux(ii,jj,kk) + yinflux(ii,jj,kk) + zinflux(ii,jj,kk) + ...
                                                        xoutflux(ii,jj,kk) + youtflux(ii,jj,kk) + zoutflux(ii,jj,kk))/6;
                               
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #6  
          
          if(emu(iang)<0 && eta(iang)>0 && xzi(iang)>0) % Octant #7
              for kk=1:1:cfg.zmesh       % Sweep through the 3D mesh from corner       
                  for jj=1:1:cfg.ymesh
                      for ii=cfg.xmesh:-1:1
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==cfg.xmesh) % At boundary setup vaccum boundary condition 
                              xinflux(ii,jj,kk) = 0;
                          else
                              xinflux(ii,jj,kk) = xoutflux(ii+1,jj,kk);
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              yinflux(ii,jj,kk) = 0;
                          else
                              yinflux(ii,jj,kk) = youtflux(ii,jj-1,kk);
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              zinflux(ii,jj,kk) = 0;
                          else
                              zinflux(ii,jj,kk) = zoutflux(ii,jj,kk-1);
                          end
                          
                          % Calculate the angular and total flux
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk) + ...
                                                        2*DA(jj,kk,iang)*xinflux(ii,jj,kk) + ...
                                                        2*DB(ii,kk,iang)*yinflux(ii,jj,kk) + ...
                                                        2*DC(ii,jj,iang)*zinflux(ii,jj,kk)) / ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                               
                          % Now we calculate the outgoing flux 
                          xoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - xinflux(ii,jj,kk);
                          youtflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - yinflux(ii,jj,kk);
                          zoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - zinflux(ii,jj,kk);
                          
                          % Flux fixup
                          if xoutflux(ii,jj,kk) < 0
                              xoutflux(ii,jj,kk) = 0;
                          end
                          if youtflux(ii,jj,kk) < 0
                              youtflux(ii,jj,kk) = 0;
                          end
                          if zoutflux(ii,jj,kk) < 0
                              zoutflux(ii,jj,kk) = 0;
                          end
                          
                          angflux(ii,jj,kk,iang,ieg) = (xinflux(ii,jj,kk) + yinflux(ii,jj,kk) + zinflux(ii,jj,kk) + ...
                                                        xoutflux(ii,jj,kk) + youtflux(ii,jj,kk) + zoutflux(ii,jj,kk))/6;
                               
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #7  
          
          if(emu(iang)>0 && eta(iang)>0 && xzi(iang)>0) % Octant #8
              for kk=1:1:cfg.zmesh       % Sweep through the 3D mesh from corner       
                  for jj=1:1:cfg.ymesh
                      for ii=1:1:cfg.xmesh
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==1) % At boundary setup vaccum boundary condition 
                              xinflux(ii,jj,kk) = 0;
                          else
                              xinflux(ii,jj,kk) = xoutflux(ii-1,jj,kk);
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              yinflux(ii,jj,kk) = 0;
                          else
                              yinflux(ii,jj,kk) = youtflux(ii,jj-1,kk);
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              zinflux(ii,jj,kk) = 0;
                          else
                              zinflux(ii,jj,kk) = zoutflux(ii,jj,kk-1);
                          end
                          
                          % Calculate the angular and total flux
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk) + ...
                                                        2*DA(jj,kk,iang)*xinflux(ii,jj,kk) + ...
                                                        2*DB(ii,kk,iang)*yinflux(ii,jj,kk) + ...
                                                        2*DC(ii,jj,iang)*zinflux(ii,jj,kk)) / ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                               
                          % Now we calculate the outgoing flux 
                          xoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - xinflux(ii,jj,kk);
                          youtflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - yinflux(ii,jj,kk);
                          zoutflux(ii,jj,kk) = 2*angflux(ii,jj,kk,iang,ieg) - zinflux(ii,jj,kk);
                          
                          % Flux fixup
                          if xoutflux(ii,jj,kk) < 0
                              xoutflux(ii,jj,kk) = 0;
                          end
                          if youtflux(ii,jj,kk) < 0
                              youtflux(ii,jj,kk) = 0;
                          end
                          if zoutflux(ii,jj,kk) < 0
                              zoutflux(ii,jj,kk) = 0;
                          end
                          
                          angflux(ii,jj,kk,iang,ieg) = (xinflux(ii,jj,kk) + yinflux(ii,jj,kk) + zinflux(ii,jj,kk) + ...
                                                        xoutflux(ii,jj,kk) + youtflux(ii,jj,kk) + zoutflux(ii,jj,kk))/6;
                               
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #8  
          %}
          
        end % End of angle
        
        maxdiff = max(max(max(abs(tempflux-preflux)./tempflux))); % Claculate flux change
        errlist(num_iter) = maxdiff;
        disp(['Inner Iteartion #',num2str(num_iter)]);
       
        flux(:,:,:,ieg) = tempflux(:,:,:); % Store the result 
        
        
        num_iter = num_iter+1;
    end % End while loop of the inner iterations
    
    figure();
    plot(errlist);
    xlabel('Iteration #');
    ylabel('Max error');
    
    figure();
    semilogy(errlist);
    xlabel('Iteration #');
    ylabel('Max error');
    
    
end % End loop of the energy groups

timetorun = toc;
disp(['Time to run: ', num2str(timetorun)])

