% --------------------------------------------------------------------------------------------------------------
% Description   : Set directional quadrature set
%                 Flux solution by the method of Sn requires a set of
%                 discrete directions and weights. Flux evaluation is
%                 performed only in these directions, and integrals over
%                 directional variables required in the transport formation
%                 are estimated by weighted sums. The directions are
%                 ordered according to increasing values of cosine with
%                 respect to the X axis (EMU), and then with respect to the
%                 y axis (ETA). Thus, all upward directions precede all
%                 downward directions. EMU's with equal ETA are are grouped
%                 together, and such a set is called an "ETA level". A
%                 third cosine, XZI, measured with respect to the z axis,
%                 is required in the calculation, but it is calculated
%                 internally. After the user has supplied a sweep of EMU
%                 for each value of ETA in either the positive or the
%                 negative hemisphere, another identical sweep must follow
%                 it. The first will be assigned negative values of XZI,
%                 while the second will have positive values. A weight, W,
%                 is given for each direction. The following fig shows the
%                 orientation
%                                   /z     
%                                  /_______x        
%                                  |
%                                  |
%                                 y|
%                                 
%                                    
% Created       : Feb 2016
% --------------------------------------------------------------------------------------------------------------

display('Initialize quandraure set ...');

wt = zeros(cfg.m,1);          % Weight for each direction
emu = zeros(cfg.m,1);         % cosine with respect to X axis
eta = zeros(cfg.m,1);         % cosine with respect to Y axis
xzi = zeros(cfg.m,1);         % cosine with respect to Z axis

if(cfg.sn==2)                 % N=2 total 8 directions in 8 octants
    emu(1) = -0.577350;  
    emu(2) =  0.577350;
    emu(3) = -0.577350;
    emu(4) =  0.577350;
    emu(5:8) = emu(1:4);
    
    eta(1:4) = -0.577350;
    eta(5:8) = -eta(1:4);
    
    xzi(1:2) = -sqrt(1-emu(1:2).^2-eta(1:2).^2);
    xzi(3:4) = -xzi(1:2);
    xzi(5:8) = xzi(1:4);
    
    wt(1:8) = 1.0/8.0;
end

if(cfg.sn==4)                 %N=4 total 24 directions in 8 octants
    emu(1) = -0.868890;  
    emu(2) = -0.350021;
    emu(3) = -0.350021;
    emu(4) = 0.350021;
    emu(5) = 0.350021;
    emu(6) = 0.868890; 
    emu(7:12) = emu(1:6);
    emu(13:24) = emu(1:12);
    
    eta(1) = -0.350021;  
    eta(2) = -0.350021;
    eta(3) = -0.868890;
    eta(4) = -0.868890;
    eta(5) = -0.350021;
    eta(6) = -0.350021; 
    eta(7:12) = eta(1:6);
    eta(13:24) = -eta(1:12);
    
    xzi(1:6) = -sqrt(1-emu(1:6).^2-eta(1:6).^2);
    xzi(7:12) = -xzi(1:6);
    xzi(13:24) = xzi(1:12);
    
    wt(1:24) = 0.333333/8.0;
end

if(cfg.sn==6)                %N=6 total 48 directions in 8 octants
    emu(1) = -0.926181;  
    emu(2) = -0.681508;
    emu(3) = -0.681508;
    emu(4) = -0.266636;
    emu(5) = -0.266636;
    emu(6) = -0.266636; 
    emu(7) =  0.266636;
    emu(8) =  0.266636;
    emu(9) =  0.266636; 
    emu(10) = 0.681508;
    emu(11) = 0.681508;
    emu(12) = 0.926181;  
    emu(13:24) = emu(1:12);
    emu(25:48) = emu(1:24);
    
    eta(1) = -0.266636;  
    eta(2) = -0.266636;
    eta(3) = -0.681508;
    eta(4) = -0.266636;
    eta(5) = -0.681508;
    eta(6) = -0.926181; 
    eta(7) = -0.926181;  
    eta(8) = -0.681508;
    eta(9) = -0.266636;
    eta(10) = -0.681508;
    eta(11) = -0.266636;
    eta(12) = -0.266636;
    
    eta(13:24) = eta(1:12);
    eta(25:48) = -eta(1:24);
    
    xzi(1:12) = -sqrt(1-emu(1:12).^2-eta(1:12).^2);
    xzi(13:24) = -xzi(1:12);
    xzi(25:48) = xzi(1:24);
    
    wt(1) = 0.176126;
    wt(2) = 0.157207;
    wt(3) = 0.157207;
    wt(4) = 0.176126;
    wt(5) = 0.157207;
    wt(6) = 0.176126;
    wt(7) = 0.176126;
    wt(8) = 0.157207;
    wt(9) = 0.176126;
    wt(10) = 0.157207;
    wt(11) = 0.157207;
    wt(12) = 0.176126;
    
    wt(13:24) = wt(1:12);
    wt(25:48) = wt(1:24);
end
