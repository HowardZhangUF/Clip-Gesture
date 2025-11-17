from collections import deque
import numpy as np

class TemporalSmoother:
    """
    Persistence + threshold smoother for stream predictions.
    """
    def __init__(self, window=21, persist=6, tau=0.55, labels=None):
        self.win = deque(maxlen=window)
        self.current = "Other"
        self.persist = persist
        self.tau = tau
        self.labels = labels

    def update(self, sims_vec):
        """
        sims_vec: 1D numpy array of class scores (cosine similarities).
        Returns smoothed label.
        """
        sims = sims_vec if isinstance(sims_vec, np.ndarray) else np.asarray(sims_vec)
        top_idx = sims.argmax()
        conf = sims[top_idx]
        pred = self.labels[top_idx] if (self.labels and conf >= self.tau) else "Other"

        self.win.append(pred)
        if pred == self.current:
            return self.current

        # switch only if new label has persisted enough in the window
        if list(self.win).count(pred) >= self.persist:
            self.current = pred
        return self.current
