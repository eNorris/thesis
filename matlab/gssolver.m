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
flux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh, cfg.igm);
angflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh, cfg.m, cfg.igm);
tempflux = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh); % temporary flux array

% Initialize the multigroup source with zeros except the point source 
extsource = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh, cfg.igm);
extsource((cfg.xmesh-1)/2+1, cfg.col_ylen/2, (cfg.zmesh-1)/2+1, cfg.igm) = 1.0e3;   % Fixed one-group point source 

% Setup vaccum boundary 
fluxi=zeros(cfg.ymesh,cfg.zmesh);
fluxj=zeros(cfg.xmesh,cfg.zmesh);
fluxk=zeros(cfg.xmesh,cfg.ymesh);

% Start the source iteration
num_iter = 1;
maxdiff = 1.0;
for ieg=1:cfg.igm  % Loop over energy groups
    disp(['Energy group #', num2str(ieg)]);
    while (num_iter<=cfg.maxit && maxdiff>cfg.epsi) 
        % Plot the solution
        figure();
        imagesc(flux(:,:,15,ieg)); 
        title(['Energy Group #',num2str(ieg),' - Iteration #',num2str(num_iter)]); 
        grid on; 
        colorbar;
        
        preflux = tempflux;  % Store the flux for the previous iteration result
        totsource = extsource(:,:,:,ieg)+isoscasource(flux,ieg,cfg,zone_id, msig); % Calculate source contributed by scatter
        tempflux = zeros(cfg.xmesh,cfg.ymesh, cfg.zmesh); % Clear flux content for new sweep
        for iang=1:cfg.m   % Loop over angles
          if(emu(iang)<0 && eta(iang)<0 && xzi(iang)<0) % Octant #1
              for kk=cfg.zmesh:-1:1       % Sweep through the 3D mesh from corner       
                  for jj=cfg.ymesh:-1:1
                      for ii=cfg.xmesh:-1:1
                          zid = zone_id(ii,jj,kk)+1; %Note: zone id start at 0, so the array index plus 1.
                          if(ii==cfg.xmesh) % At boundary setup vaccum boundary condition 
                              fluxi = 0;    % fluxi is the out flux in x direction
                              fluxi_pre =0;
                          else
                              fluxi = 2*angflux(ii+1,jj,kk)-fluxi_pre;
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              fluxj = 0;    % fluxj is the out flux in y direction
                              fluxj_pre =0;
                          else
                              fluxj = 2*angflux(ii,jj+1,kk)-fluxj_pre;
                              fluxj_pre = fluxj;
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              fluxk = 0;    % fluxk is the out flux in z direction
                              fluxk_pre =0;
                          else
                              fluxk = 2*angflux(ii,jj,kk+1)-fluxk_pre;
                              fluxk_pre = fluxk;
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj+2*DC(ii,jj,iang)*fluxk)/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                          if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
                              angflux(ii,jj,kk,iang,ieg)=0.0;
                          end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
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
                              fluxi = 2*angflux(ii-1,jj,kk)-fluxi_pre;
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              fluxj = 0;
                              fluxj_pre =0;
                          else
                              fluxj = 2*angflux(ii,jj+1,kk)-fluxj_pre;
                              fluxj_pre = fluxj;
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              fluxk = 0;
                              fluxk_pre =0;
                          else
                              fluxk = 2*angflux(ii,jj,kk+1)-fluxk_pre;
                              fluxk_pre = fluxk;
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj+2*DC(ii,jj,iang)*fluxk)/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                          if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
                              angflux(ii,jj,kk,iang,ieg)=0.0;
                          end
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
                              fluxi = 2*angflux(ii+1,jj,kk)-fluxi_pre;
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              fluxj = 0;
                              fluxj_pre =0;
                          else
                              fluxj = 2*angflux(ii,jj-1,kk)-fluxj_pre;
                              fluxj_pre = fluxj;
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              fluxk = 0;
                              fluxk_pre =0;
                          else
                              fluxk = 2*angflux(ii,jj,kk+1)-fluxk_pre;
                              fluxk_pre = fluxk;
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj+2*DC(ii,jj,iang)*fluxk)/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                          if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
                              angflux(ii,jj,kk,iang,ieg)=0.0;
                          end
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
                              fluxi = 2*angflux(ii-1,jj,kk)-fluxi_pre;
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              fluxj = 0;
                              fluxj_pre =0;
                          else
                              fluxj = 2*angflux(ii,jj-1,kk)-fluxj_pre;
                              fluxj_pre = fluxj;
                          end
                          if(kk==cfg.zmesh) % At boundary setup vaccum boundary condition
                              fluxk = 0;
                              fluxk_pre =0;
                          else
                              fluxk = 2*angflux(ii,jj,kk+1)-fluxk_pre;
                              fluxk_pre = fluxk;
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj+2*DC(ii,jj,iang)*fluxk)/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                          if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
                              angflux(ii,jj,kk,iang,ieg)=0.0;
                          end
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
                              fluxi = 2*angflux(ii+1,jj,kk)-fluxi_pre;
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              fluxj = 0;
                              fluxj_pre =0;
                          else
                              fluxj = 2*angflux(ii,jj+1,kk)-fluxj_pre;
                              fluxj_pre = fluxj;
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              fluxk = 0;
                              fluxk_pre =0;
                          else
                              fluxk = 2*angflux(ii,jj,kk-1)-fluxk_pre;
                              fluxk_pre = fluxk;
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj+2*DC(ii,jj,iang)*fluxk)/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                          if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
                              angflux(ii,jj,kk,iang,ieg)=0.0;
                          end
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
                              fluxi = 2*angflux(ii-1,jj,kk)-fluxi_pre;
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==cfg.ymesh) % At boundary setup vaccum boundary condition
                              fluxj = 0;
                              fluxj_pre =0;
                          else
                              fluxj = 2*angflux(ii,jj+1,kk)-fluxj_pre;
                              fluxj_pre = fluxj;
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              fluxk = 0;
                              fluxk_pre =0;
                          else
                              fluxk = 2*angflux(ii,jj,kk-1)-fluxk_pre;
                              fluxk_pre = fluxk;
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj+2*DC(ii,jj,iang)*fluxk)/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                          if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
                              angflux(ii,jj,kk,iang,ieg)=0.0;
                          end
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
                              fluxi = 2*angflux(ii+1,jj,kk)-fluxi_pre;
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              fluxj = 0;
                              fluxj_pre =0;
                          else
                              fluxj = 2*angflux(ii,jj-1,kk)-fluxj_pre;
                              fluxj_pre = fluxj;
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              fluxk = 0;
                              fluxk_pre =0;
                          else
                              fluxk = 2*angflux(ii,jj,kk-1)-fluxk_pre;
                              fluxk_pre = fluxk;
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj+2*DC(ii,jj,iang)*fluxk)/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                          if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
                              angflux(ii,jj,kk,iang,ieg)=0.0;
                          end
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
                              fluxi = 2*angflux(ii-1,jj,kk)-fluxi_pre;
                              fluxi_pre = fluxi;
                          end                          
                          if(jj==1) % At boundary setup vaccum boundary condition
                              fluxj = 0;
                              fluxj_pre =0;
                          else
                              fluxj = 2*angflux(ii,jj-1,kk)-fluxj_pre;
                              fluxj_pre = fluxj;
                          end
                          if(kk==1) % At boundary setup vaccum boundary condition
                              fluxk = 0;
                              fluxk_pre =0;
                          else
                              fluxk = 2*angflux(ii,jj,kk-1)-fluxk_pre;
                              fluxk_pre = fluxk;
                          end
                          angflux(ii,jj,kk,iang,ieg) = (totsource(ii,jj,kk)*vol(ii,jj,kk)+2*DA(jj,kk,iang)*fluxi+2*DB(ii,kk,iang)*fluxj+2*DC(ii,jj,iang)*fluxk)/ ...
                                   (vol(ii,jj,kk)*msig(ieg, zid,1,1)+2*(DA(jj,kk,iang)+DB(ii,kk,iang)+DC(ii,jj,iang)));
                          if angflux(ii,jj,kk,iang,ieg)<0 % Negtive flux fix
                              angflux(ii,jj,kk,iang,ieg)=0.0;
                          end
                          tempflux(ii,jj,kk) = tempflux(ii,jj,kk)+wt(iang)*angflux(ii,jj,kk,iang,ieg); % Scalar flux is the summation of angular flux
                      end
                  end
              end
          end % End of octant #8  
          
        end % End of angle
        
        maxdiff = max(max(max(abs(tempflux-preflux)./tempflux))); % Claculate flux change
        disp(['Inner Iteartion #',num2str(num_iter)]);
       
        flux(:,:,:,ieg) = tempflux(:,:,:); % Store the result 
        
        
        num_iter = num_iter+1;
    end % End while loop of the inner iterations
    
    
end % End loop of the energy groups

