function [ycfmean,ySTufmean,ySTcfdraw,ySTufdraw,epsdraw,ySTdifdraw,Acoefs,junknown,R,jknown]...
    = varcondfcast(beta,C,y,w,varargin)
% assume a N-dimensional VAR with P lags:
% y_t = Phi1 y_{t-1} +...+ Phi_ y_{t-P} + Gamma w_t + C eps_t
% where
% eps_t - orthogonal shocks, i.i.d N(0,I)
% C - produces the covariance of VAR residuals u_t = C eps_t
% w_t - exogenous variables, typically just a 1 (so Gamma is the constant)
% 
% This function computes conditional forecasts with the above VAR
% arguments:
% beta - VAR parameters matrix [Phi1 Phi2 ... PhiP Gamma]
% C - factor of the variance matrix of VAR residuals: C*C' = var(u)
% y - data on which we condition
%     need at least P first observations on all variables to initialize;
%     after that, only data on which we condition are present, rest is set to NaN
% w - exogenous variables, need to be known in all periods
% optional arguments: 
% (1) Radd, (2) radd - matrices defining additional restrictions 
%     on the shocks of the form: Radd eps = radd
% (3) S - last period with full info, forecast computation begins with S+1
% returns:
% ycfmean - a copy of y with NaNs replaced by the mean of the conditional forecast
%   the subsequent returns have size (T-S) and only contain
%   data from S+1 to T, i.e. data preceding S are dropped
% ySTufmean - the mean of the unconditional forecast
% ySTcfdraw - a draw from the conditional predictive density
% ySTufdraw - a draw from the unconditional predictive density
% epsdraw - a draw of the epsilons (from S+1 to T)
% and lots of other stuff
%
% DEPENDS: nothing
% SUBFUNCTIONS: varcompan
% SEE: (and please cite if you use this code for your project)
% Jarocinski, Marek (2010), Conditional Forecasts and Uncertainty about
% Forecast Revisions in Vector Autoregressions, Economics Letters 108(3) pp.257-259.
% Marek Jarocinski, first version: 30 July 2007
% modified 06 May 2010: exogenous variables and unit roots allowed


% measure inputs
[T,N] = size(y);
[N,NPpW] = size(beta);
W = size(w,2);
NP = NPpW - W;
P = NP/N;
% determine S - last period with full info
if length(varargin)>2
    S = varargin{3};
else
    % find S+1 - the earliest date with missing data
    [c,S1] = find(isnan(y'),1);
    if isempty(S1)
        error('varcondfcast error: no missing values in the condition set!');
    end
    S = S1-1;
end
TmS = T-S;

%%%
%%% stacked vectors
%%%
%%% convention: 'ST' denotes variables from S+1 to T
% stacked y's (S+1)...T
yST = reshape(y((S+1):end,:)',TmS*N,1);
% ylagst
ylagst = reshape(flipud(y(S-P+1:S,:))',NP,1);

%%%
%%% Build matrices R and ySTufmean
%%%
% allocate space for R, ySTufmean
ySTufmean = nan(TmS*N,1);
R = zeros(TmS*N,TmS*N);
% companion matrix
F = varcompan(beta(:,1:NP));
% horizon h = 1
% initialize rowsh (rows corresponding to horizon h)
rowsh = 1:N;
% initialize R
R(rowsh,1:N) = C;
% initialize powers of F
Fj = F;
% compute the first piece of ySTufmean
ySTufmean(rowsh,1) = beta*[ylagst; w(S+1,:)'];
% horizon h = 2:(T-S)
if TmS>1
    for h = 2:TmS
        % update rowsh
        rowsh = rowsh + N;
        % copy repeating entries in R
        R(rowsh,N+1:rowsh(N)) = R(rowsh-N,1:rowsh(N)-N);
        % fill in the new block in R
        R(rowsh,1:N) = Fj(1:N,1:N)*C;
        % compute the next power of F
        Fj = F*Fj;
        % update ylagst
        ylagst(N+1:end) = ylagst(1:N*(P-1));
        ylagst(1:N) = ySTufmean(rowsh-N);
        % compute next piece of ySTufmean
        ySTufmean(rowsh) = beta*[ylagst; w(S+h,:)'];
    end
end
%%%
%%% done building R and ySTufmean
%%%

% selector for known and unknown variables
jknown = find(~isnan(yST));
junknown = find(isnan(yST));
k = N*TmS; % total number of data points (after S)
% matrices defining restrictions
Rtil = R(jknown,:);
r = yST(jknown) - ySTufmean(jknown);
% add restrictions?
if ~isempty(varargin)
    Radd = varargin{1}; radd = varargin{2};
    Rtil = [Rtil; Radd];
    r = [r; radd];
end
q = size(Rtil,1);
% mean and factorized variance of the conditional distribution of shocks
[U,D,V] = svd(Rtil);
D = D(1:q,1:q);
M = V(:,1:q)*diag(diag(D).^(-1))*U';
condmean = M*r;
epsdraw = condmean + V(:,q+1:end)*randn(k-q,1);

% recover y's
ySTcfmean = ySTufmean + R*condmean;
ySTcfdraw = ySTufmean + R*epsdraw;
ySTufdraw = ySTufmean + R*randn(k,1);

ySTufmean = reshape(ySTufmean,N,TmS)';
ySTcfmean = reshape(ySTcfmean,N,TmS)';
ySTcfdraw = reshape(ySTcfdraw,N,TmS)';
ySTufdraw = reshape(ySTufdraw,N,TmS)';
epsdraw = reshape(epsdraw,N,TmS)';

ycfmean = [y(1:S,:); ySTcfmean];

% compute the difference btw. conditional and unconditional forecasts
% taking into account their correlation
if nargout > 5
    epsdifdraw = condmean + V(:,1:q)*randn(q,1);
    ySTdifdraw = R*epsdifdraw;
    ySTdifdraw = reshape(ySTdifdraw,N,TmS)';
end

if nargout > 6 % compute the matrix of coefficients of known data in cf
    Acoefs = R*M; % TmS*N x q
end

end

function F = varcompan(phi)
% MJ
% build matrix F (sparse)
% Output:  F - np x np matrix
% Input: phi - n x np matrix of VAR coeffs (without constant)
% Reference: Hamilton, p.259

[n pn]=size(phi);
F=spalloc(pn,pn,(n+1)*pn);
F(1:n,:)=phi;
F(n+1:pn,1:pn-n)=speye(pn-n);
end
