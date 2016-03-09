% Discrete ordinates computed tomography organ dose simulator (DOCTORS)
% Version 1.0
%
% Aim
%   Calculate organ dose and scatter of a CT scan
%
% Inputs
%   Configfilename: the name of the file containing all simulation
%   parameters, geometry, phantom/patient, scan protocol, 
% Optional inputs  
%   cfg: allows to specify a cfg structure directly instead of reading it from
%        a configuration file. 
%   adjust: this contains a string with commands to adjust the configuration
%           specified by the configfile, for example :
%           adjust = 'cfg.phantom_filename=\'HeadPhantom.pp\';cfg.start_z=0;
%                    cfg.phantom_scale=10;'
%   preadjust: same as adjust, but these changes are executed 'before'
%               sourcing the configfile ; this can be useful for example if
%               the configfile executes some conditional statements that
%               depend on the value of other cfg parameters
%   silent: If a value for this argument is provided, then the header
%               is not printed and the compute time estimates are not printed
% Outputs
%   cfg: the configuration structure read from configfilename
% 
% Last modified by Xin Liu on Mar, 2016

%function doctors(configfilename)
clear all;
close all;

%   %
%   % Show syntax
%   %
%   if (~exist('cfg'))
%     if (isempty('configfilename'))
%       disp(' ------------------------------------------- ');
%       disp('|          Doctors 1.0                      |');
%       disp('|          Liu Laboratory                   |');
%       disp('|     Missouri S&T, Rolla, MO               |');
%       disp(' ------------------------------------------- ');
%       disp(' ');
%       disp('Usage: doctors <configfilename>');
%       disp('   or: doctors(configfilename, cfg, adjust, preadjust, silent)');
%       cfg = [];
%       return;
%     end
%   end

  %
  % Disclaimer
  %
    disp(' ------------------------------------------- ');
    disp('|          Doctors 1.0                      |');
    disp('|         Liu Laboratory                    |');
    disp('|     Missouri S&T, Rolla, MO               |');
    disp(' ------------------------------------------- ');
  
  %
  % Load configuration file
  %
    configfilename = 'testinputcfg.m';
    disp(['Loading configuration file...',configfilename]);
    testinputcfg; % Load input parameters
    
    quad;         % Setup directional quandrature
    
    geom;         % Setup mesh, zone id, cell face area, cell volume
    
    t = zone_id(:,:,15);
    figure();
    contourf(t, 3);
    
    gip;          % Setup group-organized x-section data
    
    gssolver;     % Iterative solve flux by Gauss-Seidel method
    


  
 
