function iristest
% IRISTEST will load, test, and plot the MATLAB
% repository IRIS data

    % Load the data from a MATLAB repository
    load fisheriris;

    % Use sepal sizes in a design matrix
    xmat = [meas(:,[3:4]) , ones(numel(meas(:,1)), 1)];
    % Use species "setosa" as the binary label, 1 or 0
    yvec = strcmp(species, 'setosa');
    rmat = @(alpha)[cos(alpha) - sin(alpha); sin(alpha) cos(alpha)]
    x = rmat(-pi/7)
    % Place data in global variable for optimization
    global ANNDATA;
    ANNDATA = [];
    ANNDATA.xmat = xmat;
    ANNDATA.yvec = yvec;

    % IRIS data: initial hyperplane
    w0 = [ 1; 0 ;1];
    eta_1 = 0.03
    % Use MATLAB descent method to solve the problem
    wmin = fminunc(@log1cell, w0, optimset('Display','none'));

    % Classify the data using the optimal hyperplane
    cvec = ANNDATA.xmat*wmin >= 0;
    cok = 100*(1 - sum(ANNDATA.yvec - cvec)/numel(ANNDATA.yvec));

    % Plot the data
    clf;
    ph=gscatter(xmat(:,1), xmat(:,2), yvec, "rb", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    % Show the separating hyperplane as a level curve at zero
    axv = axis;
    gden = 100;
    [xg,yg] = meshgrid(linspace(axv(1), axv(2), gden), ...
        linspace(axv(3), axv(4), gden));
    fimp =@(x1,x2) wmin(1)*x1 + wmin(2)*x2 + wmin(3);
    hold on;
    contour(xg,yg,fimp(xg,yg), [0 0], 'color', 'k', 'LineWidth', 2);
    hold off;
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title(sprintf('IRIS data: %d\\%% correct', cok), ...
        'Interpreter', 'latex', 'FontSize', 14);

    [wann fann iann] = steepfixed(@annfun, ...
        w0, eta_1, imax, gnorm)
end

function [fval,g1form] = log1cell(wvec)
% [FVAL,G1FORM]=LOGCELL(WVEC) computes the objective function and the
% gradient 1-form for weights WVEC in a cell with a logistic activation.
% Data are in the global variable ANNDATA. The objective is the
% sum of squared residual errors.
% Here, N is the number of independent entries of the augmented weight
% vector
%
% INPUTS:
%         WVEC    - Nx1 augmented weight vector
% OUTPUTS:
%         FVAL    - scalar, objective value for this weight vector
%         G1FORM  - 1xN gradient as a 1-form
% NEEDS:
%         ANNDATA - global variable with observations and labels

    % Declare the global variable and find sizes
    global ANNDATA
    [M,N] = size(ANNDATA.xmat);
    L = size(ANNDATA.yvec, 2);
    
    % Anonymous functions for the logistic activation and derivative
    phi_log =@(u) 1./(1 + exp(-u));
    psi_log =@(u) phi_log(u).*(1 - phi_log(u));

    % Find the inner product vector
    uvec = ANNDATA.xmat*wvec;

    % Find the score, or activation
    zvec = phi_log(uvec);

    % Compute the residual-error vector and the objective value
    rvec = ANNDATA.yvec(:,1) - zvec;
    fval = 0.5*rvec'*rvec;
    
    % Two ways to compute the gradient: loop and vectorized
    doloop = 1;
    if doloop
        % Gradient matrix has a 1-form for each observation
        gmat = zeros(size(ANNDATA.xmat));
        for ix = 1:size(ANNDATA.xmat, 1)
            gmat(ix,:) = -rvec(ix)*psi_log(uvec(ix))*ANNDATA.xmat(ix,:);
        end
        % Gradient is the sum of observation gradients
        g1form = sum(gmat, 1);
    else
        % Gradient 1-form from a diagonal matrix of derivatives
        psimat = diag(psi_log(uvec));
        g1form = -rvec'*psimat*ANNDATA.xmat;
    end
end


function [tmin,fmin,ix]=steepfixed(objgradf,w0,s,imax_in,eps_in)
% [WMIN,FMIN,IX]=STEEPFIXED(OBJGRADF,W0,S,IMAX,F)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point W0 and using constant stepsize S. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of W
%         S        - stepsize, positive scalar value
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         WMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

    % Set convergence criteria to those supplied, if available
    if nargin >= 4 & ~isempty(imax_in)
        imax = imax_in;
    else
        imax = 50000;
    end

    if nargin >= 5 & ~isempty(eps_in)
        epsilon = eps_in;
    else
        epsilon = 1e-6;
    end

    % Initialize: search vector, objective, gradient
    tmin = w0;
    [fmin, gval] = objgradf(tmin);
    ix = 0;
    while (norm(gval)>epsilon & ix<imax)

    %set current tmin with stepsize, direction vec
    tmin = tmin + s* -gval';
    
    %set current d using wmin
    [fmin, gval] = objgradf(tmin);
   
    ix = ix + 1;
    end
end
