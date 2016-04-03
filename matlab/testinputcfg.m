% --------------------------------------------------------------------------------------------------------------
% Description   : Test configuration script for 3D discrete ordinate photon transport simulation
% Created       : Feb 2016
% --------------------------------------------------------------------------------------------------------------

%%%%%%%%%%%%%%
% SIMULATION %
%%%%%%%%%%%%%%
cfg.results_basename = 'test_input';	                           % Base of all filenames for results
cfg.callback_output = 'WriteRawView';                              % output type (e.g., WriteRawView, WriteHiqView)

%%%%%%%%%%%%%%%%%%%%%%%%%
% SOURCE AND COLLIMATOR %
%%%%%%%%%%%%%%%%%%%%%%%%%
cfg.callback_spectrum = 'GetSpectrum';                                 % Spectrum reader callback
cfg.spectrum_filename = 'nsp_2.dat';                                   % Filename of spectrum to use (includes kVp) 
% cfg.recompute_spectrum=0;                                            % Re-compute spectrum every view ?
cfg.source_fanangle = (50/180)*pi;                                     % X-ray fan angle in X-Y plane (degree)
cfg.source_coneangle = (2/180)*pi;                                     % X-ray cone angel in Z direction (degree) determined by the Z collimation
cfg.source_top_gap = 1;                                                % Air gap between x-ray source and collimator default=1 cm
cfg.source_bottom_gap = 5;                                             % Air gap between x-ray source and collimator default=5 cm
cfg.source_left_gap = cfg.source_bottom_gap*tan(cfg.source_fanangle/2);   % Air gap between x-ray source and collimator 
cfg.source_right_gap = cfg.source_left_gap;
cfg.source_front_gap = cfg.source_bottom_gap*tan(cfg.source_coneangle/2);
cfg.source_back_gap = cfg.source_front_gap;
cfg.col_xlen = 10;                                                        % Collimator x length cm
cfg.col_ylen = 10;                                                        % Collimator y length cm
cfg.col_zlen = 5;                                                         % Collimator z length cm

%%%%%%%%%%%%
% GEOMETRY %
%%%%%%%%%%%%
cfg.sid = 54;                                                      % Source-to-isocenter distance (cm)
cfg.sdd = 95;                                                      % Source-to-detector distance (cm)
cfg.xlen = 2*cfg.sdd*tan(cfg.source_fanangle/2);                     % Length in x direction
cfg.ylen = cfg.sdd-cfg.source_bottom_gap+cfg.col_ylen;                            % Length in y direction assume the W thickness 5cm
cfg.zlen = 5;                                                      % Length in z direction default=10cm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BOWTIE & FLAT FILTER FILTRATION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cfg.bowtie_type = 'medium';                                        % bowtie type small, medium, large
% recompute_bowtie=0;                                            % Recompute (dynamic) bowtie every view
cfg.flat_filters = {'cu', 0.002};                                  % Material-depth (cm) pairs in front of source, e.g. {'w',0.0001,'cu',0.002}
% recompute_filtration=0;                                        % Recompute filtration every view


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DIRECTONAL QUNDRATURE SET %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cfg.sn = 6;                                                      % N in Sn - quadrature order
cfg.m = cfg.sn*(cfg.sn+2);                                       % Total number of directions in all 8 octants

%%%%%%%%%%%%%%%%%%%%%%%%%%
% CROSS SECTION DATA SET %
%%%%%%%%%%%%%%%%%%%%%%%%%%
cfg.njoy = 'no';                                                 % If 'yes' read x-section data from a file; if 'no' read local x-section data
cfg.njoyfile = '';                                               % NJOY generated x-section data file name;
cfg.igm = 1;                                                     % Number of energy group
cfg.iht = 3;                                                     % Table position of the total x-section
cfg.ihs = 4;                                                     % Table position of the self-scatter x-section
cfg.ihm = 4;                                                     % x-section table length per energy group (usually ihm = iht+igm)
cfg.ms = 0;                                                      % x-section mixing table length (ms = 0 no mixing)
cfg.mtm = 6;                                                     % Total number of materials including all Pn's (each Pn is considered a nuclide)
cfg.isct = 1;                                                    % Maximum order of Legendre (i.e. scattering) expansion of x-section (lowest order is 0)
cfg.xsection = [0.0 0.0  0.0002 0.0002;                             % Air x-section at 60 keV nearly void P0
                0.0 0.0  0.0000 0.0002;                             % Air x-section P1 expansion
                0.0 0.0  0.2059 0.1770;                             % Water x-section at 60 keV  P0
                0.0 0.0  0.0000 0.1770;                             % Water x-section P1 expansion
                0.0 0.0 71.4753 0.0000;                             % Tunsgten x-section at 60 keV P0
                0.0 0.0 71.4753 0.0000];                            % Tunsgten x-section P1 expansion
            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOURCE ITERATION DATA SET%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cfg.epsi = 5.0e-5;                                                   % Convergence criterion on inner iterations (default = 5.e-5)
cfg.maxit = 10;                                                     % Maximum number of inner iterations (default = 20)




