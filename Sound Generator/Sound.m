% size of the computational grid
Nx = 64;    % number of grid points in the x (row) direction
x = 1e-3;   % size of the domain in the x direction [m]
dx = x/Nx;  % grid point spacing in the x direction [m]

% define the properties of the propagation medium
medium.sound_speed = 1500;      % [m/s]

% size of the initial pressure distribution
source_radius = 2;              % [grid points]

% distance between the centre of the source and the sensor
source_sensor_distance = 10;    % [grid points]

% time array
dt = 2e-9;                      % [s]
t_end = 300e-9;                 % [s]

% computation settings
input_args = {'DataCast', 'single','RecordMovie',true};

% create the computational grid
kgrid = kWaveGrid(Nx, dx, Nx, dx);

% create initial pressure distribution
source.p0 = makeDisc(Nx, Nx, Nx/2, Nx/2, source_radius);

% define a single sensor point
sensor.mask = zeros(Nx, Nx);
sensor.mask(Nx/2 - source_sensor_distance, Nx/2) = 1;

% run the simulation
sensor_data_2D = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
