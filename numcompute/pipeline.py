"""
Pipeline abstraction for NumCompute.

This module provides a lightweight pipeline for chaining transformers
and optionally ending with an estimator that implements `fit` and
`predict`.
"""

from __future__ import annotations


class Pipeline:
    """
    Chain multiple preprocessing steps and an optional final estimator.

    Parameters
    ----------
    steps : list of tuple
        Sequence of (name, step_object) pairs.

        Intermediate steps must implement:
        - fit(X)
        - transform(X)

        The final step may be either:
        - a transformer implementing fit(X), transform(X), or
        - an estimator implementing fit(X, y) and predict(X)

    Attributes
    ----------
    steps : list of tuple
        Pipeline steps as (name, object) pairs.
    named_steps : dict
        Dictionary mapping step names to step objects.

    Examples
    --------
    >>> pipe = Pipeline([
    ...     ("scale", StandardScaler()),
    ...     ("encode", OneHotEncoder())
    ... ])
    >>> X_out = pipe.fit_transform(X)
    """

    def __init__(self, steps):
        if not steps or len(steps) == 0:
            raise ValueError("steps must contain at least one pipeline step.")

        self.steps = steps
        self.named_steps = {}

        for name, step in self.steps:
            if not isinstance(name, str) or not name:
                raise ValueError("Each step name must be a non-empty string.")
            if name in self.named_steps:
                raise ValueError(f"Duplicate step name: '{name}'.")
            self.named_steps[name] = step

    def _validate_transformer(self, step, step_name: str):
        """
        Validate that a step supports fit and transform.
        """
        if not hasattr(step, "fit") or not hasattr(step, "transform"):
            raise TypeError(
                f"Step '{step_name}' must implement both fit(X) and transform(X)."
            )

    def _validate_estimator(self, step, step_name: str):
        """
        Validate that a final estimator supports fit and predict.
        """
        if not hasattr(step, "fit") or not hasattr(step, "predict"):
            raise TypeError(
                f"Final estimator '{step_name}' must implement fit(X, y) and predict(X)."
            )

    def fit(self, X, y=None):
        """
        Fit the pipeline.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like or None, default=None
            Optional target values. Needed when the final step is an estimator.

        Returns
        -------
        self
            Fitted pipeline.

        Notes
        -----
        - All intermediate steps are treated as transformers.
        - If the final step has a `predict` method and y is provided, it is
          treated as an estimator.
        """
        X_current = X

        if len(self.steps) == 1:
            name, step = self.steps[0]
            if hasattr(step, "predict") and y is not None:
                self._validate_estimator(step, name)
                step.fit(X_current, y)
            else:
                self._validate_transformer(step, name)
                step.fit(X_current)
            return self

        for name, step in self.steps[:-1]:
            self._validate_transformer(step, name)
            step.fit(X_current)
            X_current = step.transform(X_current)

        final_name, final_step = self.steps[-1]

        if hasattr(final_step, "predict") and y is not None:
            self._validate_estimator(final_step, final_name)
            final_step.fit(X_current, y)
        else:
            self._validate_transformer(final_step, final_name)
            final_step.fit(X_current)

        return self

    def transform(self, X):
        """
        Apply all transformer steps in sequence.

        Parameters
        ----------
        X : array-like
            Input features.

        Returns
        -------
        transformed_X : array-like
            Transformed output.

        Raises
        ------
        TypeError
            If the final step is an estimator without a transform method.
        """
        X_current = X

        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"Step '{name}' does not implement transform(X), so the pipeline "
                    "cannot be used for transform()."
                )
            X_current = step.transform(X_current)

        return X_current

    def fit_transform(self, X, y=None):
        """
        Fit the pipeline and return transformed output.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like or None, default=None
            Optional target values.

        Returns
        -------
        transformed_X : array-like
            Transformed data.

        Raises
        ------
        TypeError
            If the final step is an estimator without a transform method.
        """
        self.fit(X, y=y)
        return self.transform(X)

    def predict(self, X):
        """
        Run all transformer steps, then predict using the final estimator.

        Parameters
        ----------
        X : array-like
            Input features.

        Returns
        -------
        predictions : array-like
            Predicted output from the final estimator.

        Raises
        ------
        TypeError
            If the final step does not implement predict(X).
        """
        X_current = X

        if len(self.steps) == 0:
            raise ValueError("Pipeline contains no steps.")

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"Step '{name}' does not implement transform(X), required before prediction."
                )
            X_current = step.transform(X_current)

        final_name, final_step = self.steps[-1]
        if not hasattr(final_step, "predict"):
            raise TypeError(
                f"Final step '{final_name}' does not implement predict(X)."
            )

        return final_step.predict(X_current)