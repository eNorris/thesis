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

% % Setup vaccum boundary 
% Initialization, fluxi, fluxj, fluxk are outward flux crossing the cell
  fluxj=zeros(cfg.xmesh,1);
  fluxk=zeros(cfg.xmesh,cfg.ymesh);
  fluxj_pre=zeros(cfg.xmesh,1);
  fluxk_pre=zeros(cfg.xmesh,cfg.ymesh);

% Start the source iteration
num_iter = 1;
maxdiff = 1.0;
for ieg=1:cfg.igm  % Loop over energy groups
    disp(['Energy group #', num2str(ieg)]);
    while (num_iter<=cfg.maxit && maxdiff>cfg.epsi) 
        figure;
        imagesc(log10(flux(:,:,15,ieg))); 
        title(['Energy Group #',num2str(ieg),' Iteration #',num2str(num_iter)]); 
        grid on; 
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
                              fluxi = 0;    % fluxi is the inward flux in x direction
                              fluxi_pre = 0;
                          else
                              fluxi = 2*angflux(ii+1,jj,kk,iang,ieg)-fluxi_pre;
                              if fluxi<0   %  Non negtive fix
                                  fluxi = 0.0;
                                  angflux(ii+1,jj,kk,iang,ieg) = 0.5*(fluxi+fluxi_pre);  % Update the angular flux
                              end
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              fluxj(ii) = 0;    % fluxj is the inward flux in y direction
                              fluxj_pre(ii) = 0;
                          else
                              fluxj(ii) = 2*angflux(ii,jj+1,kk,iang,ieg)-fluxj_pre(ii);
                              if fluxj(ii)<0
                                  fluxj(ii) = 0.0;
                                  angflux(ii,jj+1,kk,iang,ieg) = 0.5*(fluxj(ii)+fluxj_pre(ii));
                              end
                              fluxj_pre(ii) = fluxj(ii);
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              fluxk(ii,jj) = 0;    % fluxk is the inward flux in z direction
                              fluxk_pre(ii,jj) =0;
                          else
                              fluxk(ii,jj) = 2*angflux(ii,jj,kk+1,iang,ieg)-fluxk_pre(ii,jj);
                              if fluxk(ii,jj)<0
                                  fluxk(ii,jj) = 0.0;
                                  angflux(ii,jj,kk+1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              end
                              fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj(ii)+2*DC(ii,jj,iang)*fluxk(ii,jj))/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
%                           if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
%                               angflux(ii,jj,kk,iang,ieg)=0.0;
%                           end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                          
%                           if(jj==1 && kk==1)
%                               disp(['fluxi=',num2str(fluxi), 'fluxj=', num2str(fluxj(ii)), 'fluxk=', num2str(fluxk(ii,jj))]);
%                           end
%                           if(ii==31 && jj==2 && kk==13)
%                             disp(['ii=',num2str(ii), 'jj=', num2str(jj), 'kk=', num2str(kk),'  ', num2str(max(max(max(tempflux))))]);
%                             keyboard
%                           end
                          
                      end
                  end
              end
          end % End of octant #1
          
             
          if(emu(iang)>0 && eta(iang)<0 && xzi(iang)<0) % Octant #2
              for kk=cfg.zmesh:-1:1       % Sweep through the 3D mesh from corner       
                  for jj=cfg.ymesh:-1:1
                      for ii=1:1:cfg.xmesh
                          zid = zone_id(ii,jj,kk)+1;
                          if(ii==1) % At boundary setup vaccum boundary condition
                              fluxi = 0;
                              fluxi_pre =0;
                          else
                              fluxi = 2*angflux(ii-1,jj,kk,iang,ieg)-fluxi_pre;
                              if fluxi<0   %  Non negtive fix
                                  fluxi = 0.0;
                                  angflux(ii-1,jj,kk,iang,ieg) = 0.5*(fluxi+fluxi_pre);  % Update the angular flux
                              end
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              fluxj(ii) = 0;
                              fluxj_pre(ii) = 0;
                          else
                              fluxj(ii) = 2*angflux(ii,jj+1,kk,iang,ieg)-fluxj_pre(ii);
                              if fluxj(ii)<0
                                  fluxj(ii) = 0.0;
                                  angflux(ii,jj+1,kk,iang,ieg) = 0.5*(fluxj(ii)+fluxj_pre(ii));
                              end
                              fluxj_pre(ii) = fluxj(ii);
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              fluxk(ii,jj) = 0;
                              fluxk_pre(ii,jj) =0;
                          else
                              fluxk(ii,jj) = 2*angflux(ii,jj,kk+1,iang,ieg)-fluxk_pre(ii,jj);
                              if fluxk(ii,jj)<0
                                  fluxk(ii,jj) = 0.0;
                                  angflux(ii,jj,kk+1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              end
                              fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj(ii)+2*DC(ii,jj,iang)*fluxk(ii,jj))/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
%                           if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
%                               angflux(ii,jj,kk,iang,ieg)=0.0;
%                           end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #2
          
          if(emu(iang)<0 && eta(iang)>0 && xzi(iang)<0) % Octant #3
              for kk=cfg.zmesh:-1:1       % Sweep through the 3D mesh from corner       
                  for jj=1:1:cfg.ymesh
                      for ii=cfg.xmesh:-1:1
                          zid = zone_id(ii,jj,kk)+1;
                          if(ii==cfg.xmesh) % At boundary setup vaccum boundary condition
                              fluxi = 0;
                              fluxi_pre =0;
                          else
                              fluxi = 2*angflux(ii+1,jj,kk,iang,ieg)-fluxi_pre;
                              if fluxi<0   %  Non negtive fix
                                  fluxi = 0.0;
                                  angflux(ii+1,jj,kk,iang,ieg) = 0.5*(fluxi+fluxi_pre);  % Update the angular flux
                              end
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              fluxj(ii) = 0;
                              fluxj_pre(ii) =0;
                          else
                              fluxj(ii) = 2*angflux(ii,jj-1,kk,iang,ieg)-fluxj_pre(ii);
                              if fluxj(ii)<0
                                  fluxj(ii) = 0.0;
                                  angflux(ii,jj-1,kk,iang,ieg) = 0.5*(fluxj(ii)+fluxj_pre(ii));
                              end
                              fluxj_pre(ii) = fluxj(ii);
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              fluxk(ii,jj) = 0;
                              fluxk_pre(ii,jj) = 0;
                          else
                              fluxk(ii,jj) = 2*angflux(ii,jj,kk+1,iang,ieg)-fluxk_pre(ii,jj);
                              if fluxk(ii,jj)<0
                                  fluxk(ii,jj) = 0.0;
                                  angflux(ii,jj,kk+1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              end
                              fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj(ii)+2*DC(ii,jj,iang)*fluxk(ii,jj))/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
%                           if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
%                               angflux(ii,jj,kk,iang,ieg)=0.0;
%                           end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #3
          
   
          if(emu(iang)>0 && eta(iang)>0 && xzi(iang)<0) % Octant #4
              for kk=cfg.zmesh:-1:1       % Sweep through the 3D mesh from corner       
                  for jj=1:1:cfg.ymesh
                      for ii=1:1:cfg.xmesh
                          zid = zone_id(ii,jj,kk)+1;
                          if(ii==1) % At boundary setup vaccum boundary condition
                              fluxi = 0;
                              fluxi_pre =0;
                          else
                              fluxi = 2*angflux(ii-1,jj,kk,iang,ieg)-fluxi_pre;
                              if fluxi<0   %  Non negtive fix
                                  fluxi = 0.0;
                                  angflux(ii-1,jj,kk,iang,ieg) = 0.5*(fluxi+fluxi_pre);  % Update the angular flux
                              end
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              fluxj(ii) = 0;
                              fluxj_pre(ii) = 0;
                          else
                              fluxj(ii) = 2*angflux(ii,jj-1,kk,iang,ieg)-fluxj_pre(ii);
                              if fluxj(ii)<0
                                  fluxj(ii) = 0.0;
                                  angflux(ii,jj-1,kk,iang,ieg) = 0.5*(fluxj(ii)+fluxj_pre(ii));
                              end
                              fluxj_pre(ii) = fluxj(ii);
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              fluxk(ii,jj) = 0;
                              fluxk_pre(ii,jj) =0;
                          else
                              fluxk(ii,jj) = 2*angflux(ii,jj,kk+1,iang,ieg)-fluxk_pre(ii,jj);
                              if fluxk(ii,jj)<0
                                  fluxk(ii,jj) = 0.0;
                                  angflux(ii,jj,kk+1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              end
                              fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj(ii)+2*DC(ii,jj,iang)*fluxk(ii,jj))/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
%                           if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
%                               angflux(ii,jj,kk,iang,ieg)=0.0;
%                           end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #4         
          
          if(emu(iang)<0 && eta(iang)<0 && xzi(iang)>0) % Octant #5
              for kk=1:1:cfg.zmesh       % Sweep through the 3D mesh from corner       
                  for jj=cfg.ymesh:-1:1
                      for ii=cfg.xmesh:-1:1
                          zid = zone_id(ii,jj,kk)+1;
                          if(ii==cfg.xmesh) % At boundary setup vaccum boundary condition
                              fluxi = 0;
                              fluxi_pre =0;
                          else
                              fluxi = 2*angflux(ii+1,jj,kk,iang,ieg)-fluxi_pre;
                              if fluxi<0   %  Non negtive fix
                                  fluxi = 0.0;
                                  angflux(ii+1,jj,kk,iang,ieg) = 0.5*(fluxi+fluxi_pre);  % Update the angular flux
                              end
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              fluxj(ii) = 0;
                              fluxj_pre(ii) = 0;
                          else
                              fluxj(ii) = 2*angflux(ii,jj+1,kk,iang,ieg)-fluxj_pre(ii);
                              if fluxj(ii)<0
                                  fluxj(ii) = 0.0;
                                  angflux(ii,jj+1,kk,iang,ieg) = 0.5*(fluxj(ii)+fluxj_pre(ii));
                              end
                              fluxj_pre(ii) = fluxj(ii);
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              fluxk(ii,jj) = 0;
                              fluxk_pre(ii,jj) = 0;
                          else
                              fluxk(ii,jj) = 2*angflux(ii,jj,kk-1,iang,ieg)-fluxk_pre(ii,jj);
                              if fluxk(ii,jj)<0
                                  fluxk(ii,jj) = 0.0;
                                  angflux(ii,jj,kk-1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              end
                              fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj(ii)+2*DC(ii,jj,iang)*fluxk(ii,jj))/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
%                           if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
%                               angflux(ii,jj,kk,iang,ieg)=0.0;
%                           end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #5       
          
          if(emu(iang)>0 && eta(iang)<0 && xzi(iang)>0) % Octant #6
              for kk=1:1:cfg.zmesh       % Sweep through the 3D mesh from corner       
                  for jj=cfg.ymesh:-1:1
                      for ii=1:1:cfg.xmesh
                          zid = zone_id(ii,jj,kk)+1;
                          if(ii==1) % At boundary setup vaccum boundary condition
                              fluxi = 0;
                              fluxi_pre =0;
                          else
                              fluxi = 2*angflux(ii-1,jj,kk,iang,ieg)-fluxi_pre;
                              if fluxi<0   %  Non negtive fix
                                  fluxi = 0.0;
                                  angflux(ii-1,jj,kk,iang,ieg) = 0.5*(fluxi+fluxi_pre);  % Update the angular flux
                              end
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              fluxj(ii) = 0;
                              fluxj_pre(ii) = 0;
                          else
                              fluxj(ii) = 2*angflux(ii,jj+1,kk,iang,ieg)-fluxj_pre(ii);
                              if fluxj(ii)<0
                                  fluxj(ii) = 0.0;
                                  angflux(ii,jj+1,kk,iang,ieg) = 0.5*(fluxj(ii)+fluxj_pre(ii));
                              end
                              fluxj_pre(ii) = fluxj(ii);
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              fluxk(ii,jj) = 0;
                              fluxk_pre(ii,jj) =0;
                          else
                              fluxk(ii,jj) = 2*angflux(ii,jj,kk-1,iang,ieg)-fluxk_pre(ii,jj);
                              if fluxk(ii,jj)<0
                                  fluxk(ii,jj) = 0.0;
                                  angflux(ii,jj,kk-1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              end
                              fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj(ii)+2*DC(ii,jj,iang)*fluxk(ii,jj))/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
%                           if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
%                               angflux(ii,jj,kk,iang,ieg)=0.0;
%                           end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #6  
          
          if(emu(iang)<0 && eta(iang)>0 && xzi(iang)>0) % Octant #7
              for kk=1:1:cfg.zmesh       % Sweep through the 3D mesh from corner       
                  for jj=1:1:cfg.ymesh
                      for ii=cfg.xmesh:-1:1
                          zid = zone_id(ii,jj,kk)+1;
                          if(ii==cfg.xmesh) % At boundary setup vaccum boundary condition
                              fluxi = 0;
                              fluxi_pre =0;
                          else
                              fluxi = 2*angflux(ii+1,jj,kk,iang,ieg)-fluxi_pre;
                              if fluxi<0   %  Non negtive fix
                                  fluxi = 0.0;
                                  angflux(ii+1,jj,kk,iang,ieg) = 0.5*(fluxi+fluxi_pre);  % Update the angular flux
                              end
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              fluxj(ii) = 0;
                              fluxj_pre(ii) = 0;
                          else
                              fluxj(ii) = 2*angflux(ii,jj-1,kk,iang,ieg)-fluxj_pre(ii);
                              if fluxj(ii)<0
                                  fluxj(ii) = 0.0;
                                  angflux(ii,jj-1,kk,iang,ieg) = 0.5*(fluxj(ii)+fluxj_pre(ii));
                              end
                              fluxj_pre(ii) = fluxj(ii);
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              fluxk(ii,jj) = 0;
                              fluxk_pre(ii,jj) = 0;
                          else
                              fluxk(ii,jj) = 2*angflux(ii,jj,kk-1,iang,ieg)-fluxk_pre(ii,jj);
                              if fluxk(ii,jj)<0
                                  fluxk(ii,jj) = 0.0;
                                  angflux(ii,jj,kk-1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              end
                              fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj(ii)+2*DC(ii,jj,iang)*fluxk(ii,jj))/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
%                           if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
%                               angflux(ii,jj,kk,iang,ieg)=0.0;
%                           end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #7  
          
          if(emu(iang)>0 && eta(iang)>0 && xzi(iang)>0) % Octant #8
              for kk=1:1:cfg.zmesh       % Sweep through the 3D mesh from corner       
                  for jj=1:1:cfg.ymesh
                      for ii=1:1:cfg.xmesh
                          zid = zone_id(ii,jj,kk)+1;
                          if(ii==1) % At boundary setup vaccum boundary condition
                              fluxi = 0;
                              fluxi_pre =0;
                          else
                              fluxi = 2*angflux(ii-1,jj,kk,iang,ieg)-fluxi_pre;
                              if fluxi<0   %  Non negtive fix
                                  fluxi = 0.0;
                                  angflux(ii-1,jj,kk,iang,ieg) = 0.5*(fluxi+fluxi_pre);  % Update the angular flux
                              end
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              fluxj(ii) = 0;
                              fluxj_pre(ii) = 0;
                          else
                              fluxj(ii) = 2*angflux(ii,jj-1,kk,iang,ieg)-fluxj_pre(ii);
                              if fluxj(ii)<0
                                  fluxj(ii) = 0.0;
                                  angflux(ii,jj-1,kk,iang,ieg) = 0.5*(fluxj(ii)+fluxj_pre(ii));
                              end
                              fluxj_pre(ii) = fluxj(ii);
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              fluxk(ii,jj) = 0;
                              fluxk_pre(ii,jj) = 0;
                          else
                              fluxk(ii,jj) = 2*angflux(ii,jj,kk-1,iang,ieg)-fluxk_pre(ii,jj);
                              if fluxk(ii,jj)<0
                                  fluxk(ii,jj) = 0.0;
                                  angflux(ii,jj,kk-1,iang,ieg) = 0.5*(fluxk(ii,jj)+fluxk_pre(ii,jj));
                              end
                              fluxk_pre(ii,jj) = fluxk(ii,jj);
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj(ii)+2*DC(ii,jj,iang)*fluxk(ii,jj))/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
%                           if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
%                               angflux(ii,jj,kk,iang,ieg)=0.0;
%                           end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #8  
          
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

