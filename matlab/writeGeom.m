function [ ] = writeGeom( geo, config, filename )

fileId = fopen(filename, 'w');
fprintf(fileId, '3\n');

fprintf(fileId, '%i\n', config.xmesh);
fprintf(fileId, '%i\n', config.ymesh);
fprintf(fileId, '%i\n', config.zmesh);

x_index = (0:config.xlen/config.xmesh:config.xlen);
y_index = (0:config.ylen/config.ymesh:config.ylen);
z_index = (0:config.zlen/config.zmesh:config.zlen);

for i = 1:config.xmesh
    fprintf(fileId, '%0.6f\n', x_index(i));
end

for i = 1:config.ymesh
    fprintf(fileId, '%0.6f\n', y_index(i));
end

for i = 1:config.zmesh
    fprintf(fileId, '%0.6f\n', z_index(i));
end

for i = 1:config.xmesh
    for j = 1:config.ymesh
        for k = 1:config.zmesh
            fprintf(fileId, '%i\n', geo(i,j,k));
        end
    end
end

fclose(fileId);

end

