% --------------------------------------------------------------------------------------------------------------
% Description   : Set space mesh array
% Created       : Feb 2016
% --------------------------------------------------------------------------------------------------------------

% Setup the mesh interval for x, y, z dimension
% Use odd number to make sure the source is NOT on the mesh boundary
display('Setup mesh ...');
cfg.xmesh = 59;
cfg.ymesh = 99;
cfg.zmesh = ceil(5/(cfg.source_front_gap*2));

x_index = (0:cfg.xlen/cfg.xmesh:cfg.xlen);
y_index = (0:cfg.ylen/cfg.ymesh:cfg.ylen);
z_index = (0:cfg.zlen/cfg.zmesh:cfg.zlen);

% Derive the cell face area, volume, which are used in the flux calcualtion
for mm = 1:cfg.m               % Loop over angle
    vu = abs(emu(mm));
    ve = abs(eta(mm));
    vz = abs(xzi(mm));
    for ii=1:cfg.xmesh         % Loop over space variable
        for jj=1:cfg.ymesh
            for kk=1:cfg.zmesh
                xx(ii) = x_index(ii+1)-x_index(ii);
                yy(jj) = y_index(jj+1)-y_index(jj);
                zz(kk) = z_index(kk+1)-z_index(kk);
                DA(jj,kk,mm) = vu*yy(jj)*zz(kk);
                DB(ii,kk,mm) = ve*xx(ii)*zz(kk);
                DC(ii,jj,mm) = vz*xx(ii)*yy(jj);
                vol(ii,jj,kk) = xx(ii)*yy(jj)*zz(kk);
            end
        end
    end
end
    

% Initialize zone id, fill with air
zone_id = zeros(cfg.xmesh, cfg.ymesh, cfg.zmesh, 'uint8');                  % 0--vacuum or air
                                                                            % 1--water
                                                                            % 2--tungsten or lead
% Setup zone id for the collimator
left_x = (cfg.xmesh-1)/2+1-round(cfg.col_xlen/2/(cfg.xlen/cfg.xmesh));             % collimator x left index
right_x = (cfg.xmesh-1)/2+1+round(cfg.col_xlen/2/(cfg.xlen/cfg.xmesh));            % collimator x right index
left_gap_x = (cfg.xmesh-1)/2+1-round(cfg.source_left_gap/(cfg.xlen/cfg.xmesh));    % collimator x gap left index
right_gap_x = (cfg.xmesh-1)/2+1+round(cfg.source_right_gap/(cfg.xlen/cfg.xmesh));  % collimator x gap right index

top_y = 1;                                                                         % collimator y direction 
bottom_y = round(cfg.col_ylen/(cfg.ylen/cfg.ymesh));
top_gap_y = round((cfg.col_ylen/2-cfg.source_top_gap)/(cfg.ylen/cfg.ymesh));       % the source is at the center of the collimator

front_z = (cfg.zmesh-1)/2+1-round(cfg.col_zlen/2/(cfg.zlen/cfg.zmesh));            % collimator z front index
back_z = (cfg.zmesh-1)/2+1+round(cfg.col_zlen/2/(cfg.zlen/cfg.zmesh));             % collimator x right index
front_gap_z = (cfg.zmesh-1)/2+1-round(cfg.source_front_gap/(cfg.zlen/cfg.zmesh));  % collimator x gap left index
back_gap_z = (cfg.zmesh-1)/2+1+round(cfg.source_back_gap/(cfg.zlen/cfg.zmesh));    % collimator x gap right index

zone_id(left_x:right_x, top_y:bottom_y, front_z:back_z) = 2;                       % set collimator zone as 2

zone_id(left_gap_x:right_gap_x, top_gap_y:bottom_y, front_gap_z:back_gap_z) = 0;   % set the hole in collimator zone 0

%zone_id((cfg.xmesh-1)/2+1, cfg.col_ylen/2, (cfg.zmesh-1)/2+1) = 2;                 % source point

% Setup zone id for the phantom
% Ideally the phantom zone id should be automatic generated from CT image
% Here we just simply set up a water cylinder phantom
center_x = (cfg.xmesh-1)/2+1;
center_y = round(cfg.col_ylen/2+cfg.sid/(cfg.ylen/cfg.ymesh));
radius = 17.5; 
for iz = 1:cfg.zmesh
    for iy = 1:cfg.ymesh
        for ix = 1:cfg.xmesh
            if(ix-center_x)^2 + (iy-center_y)^2 <= radius^2;
                zone_id(ix, iy, iz) = 1;
            end
        end
    end
end

% figure; imagesc(zone_id(:,:,15));
% grid on

% figure, imagesc(zone_id(:,:,13));
% grid on



