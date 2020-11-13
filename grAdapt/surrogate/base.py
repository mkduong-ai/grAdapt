# Python Standard Libraries
# decorators
from abc import abstractmethod


class Surrogate:
    """Surrogate model object for the regression
       
       Parameters
       ----------
       model : Regression model object
            Can be a scikit-learn object. It must support
                .fit(X, y)
                .predict(X, return_std) (maybe?)
            
       surrogate_params : list or tuple
            parameters passed for the surrogate_model
       

       Attributes
       ----------
       self.model : model
       self.surrogate_params : surrogate_params


       Examples
       --------
       >>> surrogate = Surrogate(surrogate_params)
       >>> surrogate.fit(X, y)
       >>> surrogate.eval_gradient(x, surrogate_grad_params)
    """

    def __init__(self, model, surrogate_params):
        self.model = model
        self.surrogate_params = surrogate_params

    @abstractmethod
    def fit(self, X, y):
        """ Fits x points to the y-coordinate
        
            Parameters
            ----------
            X : array-like of size (n, m) for n points of d dimensions
            y : array-like of shape (n_samples,)
            
            
            Returns
            -------
            None
        """
        # self.model.fit(X, y)
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **args):
        return NotImplementedError

    @abstractmethod
    def eval_gradient(self, x, surrogate_grad_params):
        """Evaluates the gradient of the surrogate model at point x
        
        Parameters
        ----------
        x : 1D array-like
        
        
        Returns
        -------
        grad : 1D array-like
            same shape as x
        """

        raise NotImplementedError
