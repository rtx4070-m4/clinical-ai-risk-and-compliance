
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_cm(model, X, y, path):
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_roc(model, X, y, path):
    RocCurveDisplay.from_estimator(model, X, y)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
