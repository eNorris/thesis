function [dist,ds2,xdc,ydc,zdc] = dsqrd(x2,y2,z2,x1,y1,z1)
%  Determine distance and ray direction cosines from source point to mesh
%  point
%
% Input:
%    x2,y2,z2: Mesh point.
%    x1,y1,z1: Source point.
% Output:
%    dist: Distance between source point and mesh point
%    ds2: Distance squared multiplied by 4pi
%    xdc,ydc,zdc: Direction cosines of vector from point (x1, y1, z1) to (x2,y2,z2)
% Created: 
%    April, 2016

e1 = 1.0e-8;
e2 = 2.5e-9;

ds2 = (x2-x1)*(x2-x1);
ds2 = ds2+(y2-y1)*(y2-y1);
ds2 = ds2+(z2-z1)*(z2-z1);
if(ds2<e1) 
    ds2 = e2;
end
dist = sqrt(ds2);
ds2 = 4*pi*ds2;

% Determine ray direction cosine from source to mesh point
xdc = (x2-x1)/dist;
ydc = (y2-y1)/dist;
zdc = (z2-z1)/dist;

end


    