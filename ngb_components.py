# type: ignore

from hpsklearn import components
from hyperopt import hp
from hyperopt.pyll import scope
from ngboost import NGBRegressor, distns, scores, learners


@scope.define
def sklearn_NGBRegressor(*args, **kwargs):
    return NGBRegressor(*args, **kwargs)


def _ng_boosting_Dist(name):
    return hp.pchoice(
        name,
        [
            (0.7, distns.Normal),
            (0.1, distns.Exponential),
            (0.1, distns.LogNormal),
            (0.1, distns.Poisson),
        ],
    )


def _ng_boosting_hp_space(
    name_func,
    Dist=None,
    Score=scores.LogScore,
    Base=learners.default_tree_learner,
    natural_gradient=True,
    n_estimators=None,
    learning_rate=None,
    minibatch_frac=None,
    col_sample=None,
    verbose=True,
    verbose_eval=100,
    tol=None,
    random_state=None,
):
    hp_space = dict(
        Dist=(_ng_boosting_Dist(name_func("Dist")) if Dist is None else Dist),
        Score=Score,
        Base=Base,
        natural_gradient=natural_gradient,
        n_estimators=(
            components._boosting_n_estimators(name_func("n_estimators"))
            if n_estimators is None
            else n_estimators
        ),
        learning_rate=(
            components._grad_boosting_learning_rate(name_func("learning_rate"))
            if learning_rate is None
            else learning_rate
        ),
        minibatch_frac=(
            components._grad_boosting_subsample(name_func("minibatch_frac"))
            if minibatch_frac is None
            else minibatch_frac
        ),
        col_sample=(
            components._grad_boosting_subsample(name_func("col_sample"))
            if col_sample is None
            else col_sample
        ),
        verbose=verbose,
        verbose_eval=verbose_eval,
        tol=components._svm_tol(name_func("tol")) if tol is None else tol,
        random_state=components._random_state(name_func("rstate"), random_state),
    )
    return hp_space


def ngb_regression(name, **kwargs):
    def _name(msg):
        return "%s.%s_%s" % (name, "ng_boosting", msg)

    hp_space = _ng_boosting_hp_space(_name, **kwargs)
    return scope.sklearn_NGBRegressor(**hp_space)
