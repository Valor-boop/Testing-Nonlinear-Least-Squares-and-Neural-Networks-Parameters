function a3q1_20060920
% Code for CISC371, Fall 2021, Assignment #3, Question #1

    % Options to silence LSQNONLIN
    optnls = optimset('Display','none');
 
    % Load the GPS data
    load('xgps.txt');
    load('ygps.txt');

    % Define as anonymous functions or 
    % use global variables and append their code after this function

    % Transpose matrix so have position vectors 
    xgps = xgps';

    % Mean location of the satellites
    w0 = mean(xgps, 2)
    
    % Equation 12.1 
    g = @(w, i) sqrt(w'*w - 2*xgps(:, i)'*w + xgps(:, i)'*xgps(:, i));
    
    % Find the receiver location
    error = @(w) [g(w, 1) - ygps(1); g(w, 2) - ygps(2); g(w, 3) - ygps(3); g(w, 4) - ygps(4); g(w, 5) - ygps(5); g(w, 6) - ygps(6)];
    
    % Attempt different starting points
    estimate_1 = lsqnonlin(error, w0)
    estimate_2 = lsqnonlin(error, 0.5*w0)
    estimate_3 = lsqnonlin(error, 2*w0)

    
    % Display the receiver location for lookup
    % Initial value of w0
    disp('Estimate 1: w0')
    fprintf('%7.1f %7.1f %7.1f\n', w0);
    disp('Cartesian coordinates of the GPS receiver are:');
    fprintf('%7.1f %7.1f %7.1f\n', estimate_1);
    disp('Earth centered Earth-fixed (ECEF) coordinates:');
    fprintf('%7.1f %7.1f %7.1f\n', ecef2lla(estimate_1'));

    % Display the receiver location for lookup
    % Initial value of 0.5 * w0
    disp('Estimate 2: 0.5*w0')
    fprintf('%7.1f %7.1f %7.1f\n', 0.5*w0);
    disp('Cartesian coordinates of the GPS receiver are:');
    fprintf('%7.1f %7.1f %7.1f\n', estimate_2);
    disp('Earth centered Earth-fixed coordinates:');
    fprintf('%7.1f %7.1f %7.1f\n', ecef2lla(estimate_2'));

    % Display the receiver location for lookup
    % Initial value of 2 * w0
    disp('Estimate 3: 2*w0')
    fprintf('%7.1f %7.1f %7.1f\n', 2*w0);
    disp('Cartesian coordinates of the GPS receiver are:');
    fprintf('%7.1f %7.1f %7.1f\n', estimate_3);
    disp('Earth centered Earth-fixed coordinates:');
    fprintf('%7.1f %7.1f %7.1f\n', ecef2lla(estimate_3'));
