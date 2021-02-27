% create the computational grid
Nx = 128;      % number of grid points in the x (row) direction
Ny = 256;      % number of grid points in the y (column) direction
Nz = 256;      % number of grid points in the z (depth) direction
dx = 10;    % grid point spacing in the x direction [m]
dy = 50e-6;    % grid point spacing in the y direction [m]
dz = 50e-6;    % grid point spacing in the z direction [m]
kgrid = makeGrid(Nx, dx, Ny, dy,Nz,dz);
% define the medium properties
medium.sound_speed = 1500;  % [m/s]
medium.density = 1000;                  %  [kg/m^3]
% define an initial pressure using makeBall
radius = 5;
source.p0 = makeBall(Nx, Ny, Nz, Nx/2, Ny/2, Nz/2, radius);

% create initial pressure distribution
source_radius = 2;              % [grid points]
source.p0 = zeros(Nx, 1);
source.p0(Nx/2 - source_radius:Nx/2 + source_radius) = 1;

% define a Ball sensor mask
num_sensor_points = 50;
sensor.mask = makeBall(Nx, Ny, Nz, Nx/2, Ny/2, Nz/2, source_radius);

% define a single sensor point
%source_sensor_distance = 10;    % [grid points]
%sensor.mask = zeros(Nx, 1);
%sensor.mask(Nx/2 + source_sensor_distance) = 1;
%sensor.mask = [-10e-3, 10e-3];

% run the simulation
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor,'RecordMovie',false);