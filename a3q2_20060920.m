function a3q2_20060920
% A3Q2: CISC371, Assignment 3, Question 2
% CISC371, Fall 2021, A3Q2: IRIS sepal data for species I. versicolor

    % Use Fisher's iris data: sepals and I. versicolor species
    load fisheriris;
    xmat = [-2 1 ; 0 -1; 0 0; 0 1; 2 -1]
    yvec = [-1, 1, 1, 1, -1]'
    
    % Clear the Matlab data from the workspace
    clear meas species;

    % Set the size of the ANN: L layers, expanded weight vector
    Lnum = 2;
    wvecN = 9

    % Set the auxiliary data structures for functions and gradients
    global ANNDATA
    ANNDATA.lnum = Lnum;
    ANNDATA.xmat = [xmat ones(size(xmat, 1), 1)];
    ANNDATA.yvec = yvec;

    % Set the starting point: fixed weight vector
    w0 = [1; 0 ;2 ;1 ;0 ;1 ;1 ;0 ;1];

    % Set the learning rate and related parameters
    % Learning rate 1
    eta_1 = 0.03;
    % Learning rate 2
    %eta_2 = 1;
    % Learning rate 3
    %eta_3 = 0.000001;
    imax  = 5000;
    gnorm = 1e-3;

  
    
    % Builtin neural network
    figure(2);
    disp('   ... doing NET...');
    net2layer = configure(feedforwardnet(3), xmat', yvec');
    net2layer.trainParam.showWindow = 0;
    [mlnet, mltrain] = train(net2layer, xmat', yvec');
    ynet = (mlnet(xmat')>0.5)*2 - 1;
    % Plot and proceed
    ph=gscatter(xmat(:,1), xmat(:,2), ynet, "mc", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: MATLAB network', ...
        'interpreter', 'latex', 'fontSize', 14);
    pause(0.5);

    % Hard-coded logistic activation, 1 hidden layer
    % eta = 0.01 (learning rate 1) 
    figure(3);
    disp('   ... doing ANN response...');
    [wann fann iann] = steepfixed(@annfun, ...
        w0, eta_1, imax, gnorm)
    yann = annclass(wann)
    yann
    cok = 100*(1 - sum(abs(ANNDATA.yvec - yann))/numel(ANNDATA.yvec));
    % Plot and pause
    disp(sprintf('ANN (%d), W is', iann));
    disp(wann');
    fprintf('ETA: %1.2f\n', eta_1);
    fprintf('Descent: %d%% correct\n', cok);
    figure(2);
    ph=gscatter(xmat(:,1), xmat(:,2), yann, "mc", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: custom network', ...
        'interpreter', 'latex', 'fontSize', 14);

    % generate confusion matrix, learning rate = 0.01
    figure(4);
    c = confusionmat(yvec, double(yann))
    confusionchart(c)
end


function [fval, gform] = annfun(wvec)
% FUNCTION [FVAL,GFORM]=ANNFUN(WVEC) computes the response of a simple
% neural network that has 1 hidden layer of sigmoids and a linear output
% neuron. WVEC is the initial estimate of the weight vector. FVAL is the
% scalar objective evaluation a GFORM is the gradient 1-form (row vector).
% ANNDATA is a global structure containing data and labels.
%
% INPUTS:
%         WVEC    -  weight vector, output then hidden
% GLOBALS:
%         ANNDATA -  Structure containing
%                    lnum - number of hidden units to compute
%                    xmat - MxN matrix, each row is a data vector
%                    yvec - Mx1 column vector, each label is +1 or -1
% OUTPUTS:
%         FVAL    - 1xM row vector of sigmoid responses

    global ANNDATA
    % Problem size: original data, intermediate data
    [m, n] = size(ANNDATA.xmat);
    l = ANNDATA.lnum;

    % Separate output weights and hidden weights; latter go into a matrix
    wvec1= wvec(1:(l + 1));
    wvecH = reshape(wvec((l+2):end), n, l);

    % Compute the hidden responses as long row vectors, 1 per hidden neuron;
    % then, append 1's to pass the transfer functions to the next layer
    phi2mat = (1./(1+exp(-(ANNDATA.xmat*wvecH)))); %apply logistic activation function to inputs
    phi2mat(:,end+1) = 1; % %Append a 1 to the input for layer 1 

    % Compute the output transfer function: linear in hidden responses
    phi2vec = phi2mat*wvec1;

    % ANN quantization is Heaviside step function of transfer function
    q2vec = phi2vec >= 0;

    % Residual is difference between label and network output
    rvec = ANNDATA.yvec - q2vec;

    % Objective is sum of squares of residual errors
    fval = 0.5*sum((rvec).^2);

    % If required, compute and return the gradient 1-form
    if nargout > 1
        % Compute the hidden differential responses, ignoring the appended 1's
        phi_log = @(u) 1./(1+exp(-u));
        psi_log = @(u) phi_log(u).*(1 - phi_log(u));
        psimat = psi_log(ANNDATA.xmat * wvecH);
 
        % Set up the hidden gradients, then loop through the data vectors
        hidgrad = [];
        for jx = 1:m
            % Find the product of the derivative vector and Jacobian matrix
            imat = eye(2);
            thisJmat = [diag(psimat(jx, :))]; % Equation 14.15: Jacobian matrix of the activations of the hidden layer
            thisDmat = kron(imat, ANNDATA.xmat(jx, :)); % Equation 14.8: Calculate gradient
            thisJDaug = [thisJmat*thisDmat; zeros(1,6)]; % Equation 14.17: Derivative of outputs of hidden layer
            hidgrad = [hidgrad ; wvec1'*thisJDaug];
        end

        % Differentiate the residual error and scale the gradient matrix
        grad12mat = -diag(rvec)*[phi2mat hidgrad];

        % Net gradient is the sum of the gradients of each data vector
        gform = sum(grad12mat);
    end
end

function [rvec, xfmat] = annclass(wvec)
% FUNCTION RVEC=ANNCLASS(WVEC) computes the response of a simple neural
% network that has 1 hidden layer of logistic cells and a linear output.
% ANNDATA is a global structure containing data and labels.
%
% INPUTS:
%         WVEC  -  weight vector, output then hidden
% GLOBALS:
%         ANNDATA -  Structure containing
%                    lnum - number of hidden units to compute
%                    xmat - MxN matrix, each row is a data vector
%                    yvec - Mx1 column vector, each label is 0 or 1
% OUTPUTS:
%         RVEC  - Mx1 vector of linear responses to data
%         XFMAT - Mx(L+1) array of hidden-layer responses to data

    % Problem size: original data, intermediate data
    global ANNDATA
    [m,n] = size(ANNDATA.xmat);
    l = ANNDATA.lnum;

    % Separate output weights and hidden weights; latter go into a matrix
    wvec2 = wvec(1:(l + 1));
    wvecH = reshape(wvec((l+2):end), n, l);

    % Compute the hidden responses as long row vectors, 1 per hidden neuron;
    % then, append 1's to pass the transfer functions to the next layer
    xfmat = (1./(1+exp(-(ANNDATA.xmat*wvecH))));
    xfmat(:,end+1) = 1;

    % Compute the transfer function: linear in hidden responses
    hidxfvec = xfmat*wvec2;

    % ANN response is Heaviside step function of transfer function
    rvec = (hidxfvec >= 0);
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
